"""Regime functions run over the same Run and stage bodies as saved regimes."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
import re

import pytest
import torch

from experiments.two_models import two_models
from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.checkpoint import load_checkpoint
from manabot.training.execution import execute_regime
from manabot.training.records import TrainingRun
from manabot.training.regime import (
    ArtifactReference,
    Execution,
    Learning,
    PretrainedPolicy,
    RecoveryPolicy,
    TrainingRegime,
    TrainSelfPlay,
)
import manabot.training.run as run_handle
from manabot.training.run import Run, execute, load_regime, regime_record, smoke
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
            partial(one_learner, updates=2),
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
    regime = partial(
        two_models,
        decks=MatchConfig(),
        updates=1,
        learning=LEARNING,
        reference=AGENT,
        learner=ModelConfig(hidden_dim=16, num_attention_heads=2),
        streams=2,
        transitions=16,
    )
    result = execute(regime, out=tmp_path / "run", seed=7)
    assert result.status == "completed"
    record = result.record
    assert [stage.id for stage in record.stages] == ["export-reference", "self-play"]
    identity = record.regime_function
    assert record.regime is None and identity is not None
    assert identity.function == "experiments.two_models:two_models"
    assert identity.arguments_sha256 == canonical_sha256(identity.arguments)
    # Defaults are recorded with the bound arguments.
    assert identity.arguments["execution"]["device"] == "cpu"
    assert identity.arguments["updates"] == 1
    reference, learner = result.checkpoints["reference"], result.checkpoints["learner"]
    assert reference.raw is not None and learner.raw is not None
    trained = record.stages[1]
    assert trained.inputs["frozen_opponent"]["sha256"] == reference.raw["sha256"]
    assert learner.raw == trained.artifacts["raw"]
    assert trained.arguments is not None
    assert (trained.arguments["model"], trained.arguments["opponent"]) == (
        "learner",
        "reference",
    )
    assert trained.arguments["settings"]["behavior"] == "frozen"
    frozen, _ = load_checkpoint(reference.raw["path"])
    student, _ = load_checkpoint(learner.raw["path"])
    assert frozen.config == AGENT
    assert student.config.hidden_dim == 16
    # The reference is never trained: its export is its initialization.
    assert record.stages[0].artifacts["raw"] == reference.raw
    exported = TrainingRun.model_validate_json((tmp_path / "run/run.json").read_text())
    assert exported.regime_function == identity


def test_arguments_are_checked_before_a_run_opens() -> None:
    def record(regime: object) -> object:
        return regime_record(regime, wall_seconds=60, recovery=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="int"):
        record(partial(one_learner, updates="many"))
    with pytest.raises(TypeError, match="unknown"):
        record(partial(one_learner, updates=1, unknown=2))
    with pytest.raises(TypeError, match="not bound"):
        record(one_learner)
    with pytest.raises(TypeError, match="module-level"):
        record(lambda run: None)
    identity = record(partial(two_models, decks=MatchConfig(), updates=3))
    assert identity.arguments["learning"]["gradient"] == "damped_policy"
    rebound = load_regime(identity.function, identity.arguments)
    assert record(rebound).arguments == identity.arguments


def test_smoke_writes_one_tiny_update_and_rejects_bad_code(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    result = smoke(partial(one_learner, updates=500), out=tmp_path / "smoke")
    assert result.status == "completed"
    assert [stage.updates for stage in result.record.stages] == [1]
    raw = result.checkpoints["policy"].raw
    assert raw is not None and raw["path"].startswith(str(tmp_path / "smoke"))
    agent, _ = load_checkpoint(raw["path"])
    assert agent.config == AGENT
    with pytest.raises(RuntimeError, match="regime code failed"):
        smoke(partial(failing, updates=500), out=tmp_path / "failing")


def test_function_regime_resumes_at_an_update_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    monkeypatch.setattr(run_handle, "PROGRESS_EXPORT_SECONDS", 0.0)
    monkeypatch.setattr(run_handle, "PROGRESS_EXPORT_SHARE", float("inf"))
    regime = partial(one_learner, updates=3)
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
            regime, out=tmp_path / "whole", seed=5, store=store, recovery=recovery
        )
        with monkeypatch.context() as patch:
            patch.setattr(run_handle, "export_training_run", fail_after_second_update)
            with pytest.raises(KeyboardInterrupt):
                execute(
                    regime,
                    out=tmp_path / "failed",
                    seed=5,
                    store=store,
                    recovery=recovery,
                )
        failed_id = store.con.execute(
            "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        failed = store.training_run(failed_id)
        assert failed.status == "interrupted"
        assert failed.stages[0].updates == 2
        with pytest.raises(ValueError, match="recipe or seed"):
            execute(
                partial(one_learner, updates=4),
                out=tmp_path / "other",
                seed=5,
                store=store,
                recovery=recovery,
                resume_from=failed.id,
            )
        resumed = execute(
            regime,
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


def two_stages(run: Run, *, updates: int) -> None:
    """One model trained in two consecutive self-play stages."""
    model = run.model("policy", AGENT, ObservationConfig(), seed=run.seed)
    for _ in range(2):
        self_play(run, model, decks=MatchConfig(), **_settings(updates))  # type: ignore[arg-type]


def test_a_resumed_continuation_records_and_trains_as_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    monkeypatch.setattr(run_handle, "PROGRESS_EXPORT_SECONDS", 0.0)
    monkeypatch.setattr(run_handle, "PROGRESS_EXPORT_SHARE", float("inf"))
    regime = partial(two_stages, updates=2)
    recovery = RecoveryPolicy(checkpoint_updates=1)
    export = run_handle.export_training_run

    def fail_in_second_stage(
        run_id: str, owner: VerifyStore, path: str | Path
    ) -> TrainingRun:
        run = export(run_id, owner, path)
        if (
            run.status == "running"
            and run.stages
            and run.stages[-1].id == "self-play-2"
            and run.stages[-1].updates == 1
        ):
            raise KeyboardInterrupt("stopped inside the continued stage")
        return run

    with VerifyStore(tmp_path / "training.sqlite") as store:
        whole = execute(
            regime, out=tmp_path / "whole", seed=5, store=store, recovery=recovery
        )
        with monkeypatch.context() as patch:
            patch.setattr(run_handle, "export_training_run", fail_in_second_stage)
            with pytest.raises(KeyboardInterrupt):
                execute(
                    regime,
                    out=tmp_path / "failed",
                    seed=5,
                    store=store,
                    recovery=recovery,
                )
        failed_id = store.con.execute(
            "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        resumed = execute(
            regime,
            out=tmp_path / "resumed",
            seed=5,
            store=store,
            recovery=recovery,
            resume_from=failed_id,
        )
    assert resumed.status == "completed"
    assert [s.id for s in resumed.record.stages] == ["self-play", "self-play-2"]
    for left, right in zip(whole.record.stages, resumed.record.stages, strict=True):
        assert left.arguments == right.arguments
        _assert_same(left.artifacts["raw"]["path"], right.artifacts["raw"]["path"])
    assert resumed.record.stages[1].arguments["settings"]["initial"] == "self-play"


def orphaned_continuation(run: Run) -> None:
    model = run.model("policy", AGENT, ObservationConfig())
    model.trained_by = "self-play"
    self_play(run, model, decks=MatchConfig(), **_settings(1))  # type: ignore[arg-type]


def interleaved_continuation(run: Run) -> None:
    first = run.model("first", AGENT, ObservationConfig())
    second = run.model("second", AGENT, ObservationConfig())
    for model in (first, second, first):
        self_play(run, model, decks=MatchConfig(), **_settings(1))  # type: ignore[arg-type]


@pytest.mark.parametrize("regime", [orphaned_continuation, interleaved_continuation])
def test_a_continuation_without_learner_or_snapshot_is_refused(
    tmp_path: Path, regime: Callable[[Run], None]
) -> None:
    with pytest.raises(ValueError, match="neither its live learner"):
        execute(regime, out=tmp_path / "run", seed=5)


_REFERENCE: ArtifactReference = {"path": "unused.pt", "sha256": "0" * 64, "bytes": 1}


def pretrained_over_published(run: Run) -> None:
    model = run.model("policy", AGENT, ObservationConfig())
    run.export(model, decks=MatchConfig())
    pretrained = PretrainedPolicy(
        source_run=_REFERENCE, source_stage="self-play", checkpoint=_REFERENCE
    )
    self_play(run, model, decks=MatchConfig(), pretrained=pretrained, **_settings(1))  # type: ignore[arg-type]


def weights_over_trained(run: Run) -> None:
    model = run.model("policy", AGENT, ObservationConfig())
    self_play(run, model, decks=MatchConfig(), **_settings(1))  # type: ignore[arg-type]
    self_play(
        run, model, decks=MatchConfig(), initial_weights=_REFERENCE, **_settings(1)
    )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("regime", "message"),
    [
        (
            pretrained_over_published,
            "load pretrained into model 'policy', which holds weights from stage 'export-policy'",
        ),
        (
            weights_over_trained,
            "load initial_weights into model 'policy', which holds weights from stage 'self-play'",
        ),
    ],
)
def test_a_starting_state_over_a_model_with_weights_is_refused(
    tmp_path: Path, regime: Callable[[Run], None], message: str
) -> None:
    torch.set_num_threads(1)
    with pytest.raises(ValueError, match=re.escape(message)):
        execute(regime, out=tmp_path / "run", seed=5)


def test_a_saved_regime_smokes_through_the_same_stages(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    regime = TrainingRegime(
        id="saved",
        world=managym.WORLD_VERSION,
        match=MatchConfig(),
        agent=AGENT,
        stages=[
            TrainSelfPlay(id="first", operation="train_self_play", **_settings(400)),  # type: ignore[arg-type]
            TrainSelfPlay(
                id="second",
                operation="train_self_play",
                initial="first",
                **_settings(400),  # type: ignore[arg-type]
            ),
        ],
    )
    with VerifyStore(tmp_path / "training.sqlite") as store:
        record = execute_regime(
            regime, 0, tmp_path / "smoke", store, smoke_execution=Execution()
        )
    assert record.status == "completed"
    assert [(s.id, s.updates) for s in record.stages] == [
        ("first", 1),
        ("second", 1),
    ]
    # A saved regime's smoke run records the saved regime, unchanged.
    assert record.regime_function is None
    assert record.regime == regime
    assert record.regime_digest == canonical_sha256(regime.model_dump(mode="json"))
    assert record.stages[1].inputs.keys() >= {"first/raw", "first/optimizer"}
    assert record.selected_artifact == record.stages[-1].artifacts["raw"]
    load_checkpoint(record.stages[-1].artifacts["raw"]["path"])
