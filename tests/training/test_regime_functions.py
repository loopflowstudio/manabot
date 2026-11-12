"""Regime functions run over the same Run and stage bodies as saved regimes."""

from pathlib import Path

import pytest
import torch

from experiments.two_models import two_models
from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.checkpoint import load_checkpoint
from manabot.training.execution import execute_regime
from manabot.training.records import TrainingRun
from manabot.training.regime import (
    Learning,
    RecoveryPolicy,
    TrainingRegime,
    TrainSelfPlay,
)
import manabot.training.run as run_handle
from manabot.training.run import RegimeCall, Run, execute, rehearse
from manabot.training.stages import self_play
from manabot.verify.store import VerifyStore
import managym

AGENT = ModelConfig(hidden_dim=8, num_attention_heads=2)
LEARNING = Learning(epochs=1, minibatches=1, retained_fraction=1, ema=0.9)


def _settings(updates: int) -> dict[str, object]:
    return {
        "updates": updates,
        "streams": 2,
        "transitions": 16,
        "learning_seats": "one",
        "batch_closure": "exact-quota",
        "learning": LEARNING,
    }


def one_learner(run: Run, *, updates: int) -> None:
    """One model trained by self-play; initialized like a saved regime's."""
    model = run.model("policy", AGENT, ObservationConfig(), seed=run.seed)
    self_play(run, model, decks=MatchConfig(), **_settings(updates))  # type: ignore[arg-type]


def failing(run: Run, *, updates: int) -> None:
    model = run.model("policy", AGENT, ObservationConfig())
    self_play(run, model, decks=MatchConfig(), **_settings(updates))  # type: ignore[arg-type]
    raise RuntimeError("regime code failed after its stage")


def _weights(path: str) -> dict[str, torch.Tensor]:
    agent, _ = load_checkpoint(path)
    return agent.state_dict()


def _assert_same(left: str, right: str) -> None:
    a, b = _weights(left), _weights(right)
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_function_self_play_matches_the_saved_stage(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    regime = TrainingRegime(
        id="saved",
        world=managym.WORLD_VERSION,
        match=MatchConfig(),
        agent=AGENT,
        schedule_clock="iteration_fraction",
        stages=[
            TrainSelfPlay(id="self-play", operation="train_self_play", **_settings(2))  # type: ignore[arg-type]
        ],
    )
    with VerifyStore(tmp_path / "training.sqlite") as store:
        saved = execute_regime(regime, 41, tmp_path / "saved", store)
        function = execute(
            RegimeCall.of(one_learner, updates=2),
            out=tmp_path / "function",
            seed=41,
            store=store,
        )
    assert saved.status == function.status == "completed"
    (left,), (right,) = saved.stages, function.record.stages
    assert right.operation == "train_self_play"
    assert (left.updates, left.games, left.learner_transitions) == (
        right.updates,
        right.games,
        right.learner_transitions,
    )
    for weights in ("initial_raw", "raw", "ema"):
        _assert_same(left.artifacts[weights]["path"], right.artifacts[weights]["path"])
    checkpoints = function.checkpoints["policy"]
    assert checkpoints.raw == right.artifacts["raw"]
    assert checkpoints.ema == right.artifacts["ema"]


def test_two_models_train_one_against_the_other_frozen(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    call = RegimeCall.of(
        two_models,
        decks=MatchConfig(),
        updates=1,
        learning=LEARNING,
        reference=AGENT,
        learner=ModelConfig(hidden_dim=16, num_attention_heads=2),
        streams=2,
        transitions=16,
    )
    result = execute(call, out=tmp_path / "run", seed=7)
    assert result.status == "completed"
    record = result.record
    assert [stage.id for stage in record.stages] == ["export-reference", "self-play"]
    assert record.regime is None and record.regime_call is not None
    assert record.regime_call.function == "experiments.two_models:two_models"
    assert record.regime_call.arguments == call.arguments
    assert record.regime_call.arguments_sha256 == canonical_sha256(call.arguments)
    assert record.regime_call.arguments["execution"]["device"] == "cpu"
    reference, learner = result.checkpoints["reference"], result.checkpoints["learner"]
    assert reference.raw is not None and learner.raw is not None
    trained = record.stages[1]
    assert trained.inputs["frozen_opponent"]["sha256"] == reference.raw["sha256"]
    assert learner.raw == trained.artifacts["raw"]
    frozen, _ = load_checkpoint(reference.raw["path"])
    student, _ = load_checkpoint(learner.raw["path"])
    assert frozen.config == AGENT
    assert student.config.hidden_dim == 16
    # The reference is never trained: its export is its initialization.
    assert record.stages[0].artifacts["raw"] == reference.raw
    exported = TrainingRun.model_validate_json((tmp_path / "run/run.json").read_text())
    assert exported.regime_call == record.regime_call


def test_arguments_are_validated_before_a_run_opens() -> None:
    with pytest.raises(ValueError, match="updates"):
        RegimeCall.of(one_learner, updates="many")
    with pytest.raises(ValueError, match="extra"):
        RegimeCall.of(one_learner, updates=1, unknown=2)
    call = RegimeCall.of(two_models, decks=MatchConfig(), updates=3)
    # Defaults are resolved into the record, not only the arguments written.
    assert call.arguments["learning"]["gradient"] == "damped_policy"
    assert RegimeCall.load(call.path, call.arguments).arguments == call.arguments


def test_rehearsal_runs_one_tiny_update_and_rejects_bad_code(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    record = rehearse(RegimeCall.of(one_learner, updates=500))
    assert record.status == "completed"
    assert [stage.updates for stage in record.stages] == [1]
    assert not Path(record.stages[0].artifacts["raw"]["path"]).exists()
    with pytest.raises(RuntimeError, match="regime code failed"):
        rehearse(RegimeCall.of(failing, updates=500))


def test_function_regime_resumes_at_an_update_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    call = RegimeCall.of(one_learner, updates=3)
    recovery = RecoveryPolicy(checkpoint_updates=1)
    export = run_handle.export_training_run

    def fail_after_second_update(
        run_id: str, owner: VerifyStore, path: str | Path
    ) -> TrainingRun:
        run = export(run_id, owner, path)
        if run.status == "running" and run.stages and run.stages[-1].updates == 2:
            raise KeyboardInterrupt("stopped after a committed update")
        return run

    with VerifyStore(tmp_path / "training.sqlite") as store:
        whole = execute(
            call, out=tmp_path / "whole", seed=5, store=store, recovery=recovery
        )
        with monkeypatch.context() as patch:
            patch.setattr(run_handle, "export_training_run", fail_after_second_update)
            with pytest.raises(KeyboardInterrupt):
                execute(
                    call, out=tmp_path / "failed", seed=5, store=store, recovery=recovery
                )
        failed_id = store.con.execute(
            "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        failed = store.training_run(failed_id)
        assert failed.status == "interrupted"
        assert failed.stages[0].updates == 2
        with pytest.raises(ValueError, match="recipe or seed"):
            execute(
                RegimeCall.of(one_learner, updates=4),
                out=tmp_path / "other",
                seed=5,
                store=store,
                recovery=recovery,
                resume_from=failed.id,
            )
        resumed = execute(
            call,
            out=tmp_path / "resumed",
            seed=5,
            store=store,
            recovery=recovery,
            resume_from=failed.id,
        )
    assert resumed.status == "completed"
    assert resumed.record.parent_run_id == failed.id
    (stage,) = resumed.record.stages
    assert stage.updates == 3
    _assert_same(
        whole.record.stages[0].artifacts["raw"]["path"], stage.artifacts["raw"]["path"]
    )
