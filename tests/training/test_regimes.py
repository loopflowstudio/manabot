"""Training contracts, credit boundaries, and experiment controls."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.training.execution import execute_regime, validate_regime
from manabot.training.models import StageRecord, TrainingRegime, TrainingRun
from manabot.training.references import reference_distribution
from manabot.training.selection import selected_rows
from manabot.verify.store import VerifyStore

ROOT = Path(__file__).resolve().parents[2]


def recipe(name="direct-self-play"):
    value = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes" / f"{name}.json").read_text()
    )
    value.agent.semantic_pack = "ur-lessons-vs-gw-allies"
    return value


def test_all_recipes_validate_and_reject_future_references():
    for path in (ROOT / "experiments/regimes").glob("*.json"):
        validate_regime(TrainingRegime.model_validate_json(path.read_text()))
    raw = recipe().model_dump()
    raw["stages"][0]["initial"] = "policy-1"
    with pytest.raises(ValueError, match="earlier"):
        TrainingRegime.model_validate(raw)
    raw = recipe().model_dump()
    raw["stages"][0]["operation"] = "belief"
    with pytest.raises(ValueError):
        TrainingRegime.model_validate(raw)
    raw = recipe()
    raw.world = "invalid"
    with pytest.raises(ValueError, match="world"):
        validate_regime(raw)


def test_reference_has_full_support_and_is_permutation_invariant():
    obs = {
        "actions_valid": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        "actions": torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
    }
    probs = reference_distribution(obs, "action_type_uniform")
    torch.testing.assert_close(probs, torch.tensor([[0.25, 0.25, 0.5, 0.0]]))
    permutation = torch.tensor([2, 0, 3, 1])
    moved = reference_distribution(
        {k: v[:, permutation] for k, v in obs.items()}, "action_type_uniform"
    )
    torch.testing.assert_close(moved, probs[:, permutation])
    assert selected_rows(torch.zeros(4), 0.25, 0.01).numel() == 0


def test_store_exports_committed_run_and_preserves_old_schema(tmp_path):
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = TrainingRun(
            id="fixture",
            regime=recipe(),
            regime_digest="test",
            seed=1,
            seed_streams={},
            identities={},
            stages=[StageRecord(id="policy-0")],
        )
        store.save_training_run(run)
        run.status = "failed"
        assert store.training_run(run.id).status == "pending"
        store.save_training_run(run)
        assert store.training_run(run.id).status == "failed"
        assert store.con.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_deadline_retains_failed_attempt(tmp_path):
    value = recipe()
    value.wall_seconds = 1e-9
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(TimeoutError):
            execute_regime(value, 197, tmp_path / "run", store)
        result = json.loads((tmp_path / "run/run.json").read_text())
        assert result["status"] == "interrupted"
        assert result["stages"][0]["status"] == "interrupted"
        assert result["stages"][0]["cumulative_seconds"] is None
        assert result["seconds"] > 0


def test_named_treatments_execute_real_optimizer_batches():
    from types import SimpleNamespace

    from manabot.env import ObservationSpace
    from manabot.infra.hypers import AgentSpec
    from manabot.model.agent import Agent
    from manabot.sim.net_opponent import RolloutBatch
    from manabot.training.objectives import update_iteration

    torch.set_num_threads(1)
    space = ObservationSpace()
    agent = Agent(space, AgentSpec(hidden_dim=8, num_attention_heads=2))
    obs = space.encoder.allocate(2)
    obs["actions_valid"][:, :2] = 1
    obs["agent_player_valid"][:] = 1
    obs["opponent_player_valid"][:] = 1
    obs["actions"][:, 0, 0] = 1
    obs["actions"][:, 1, 1] = 1
    with torch.no_grad():
        logits, values = agent({k: torch.as_tensor(v) for k, v in obs.items()})
        probs = logits.softmax(-1).numpy()
    batch = RolloutBatch(
        obs={k: v[:, None] for k, v in obs.items()},
        actions=np.zeros((2, 1), dtype=np.int64),
        logprobs=np.log(probs[:, 0, None]),
        rewards=np.array([[0.0], [1.0]], dtype=np.float32),
        dones=np.array([[False], [True]]),
        values=values.numpy().reshape(2, 1),
        next_obs={k: v[:1] for k, v in obs.items()},
        next_done=np.ones(1, dtype=bool),
        probabilities=probs[:, None],
    )
    from manabot.sim.net_opponent import NetOpponentTrainer

    trainer = SimpleNamespace(
        agent=agent,
        experiment=SimpleNamespace(device="cpu"),
        optimizer=torch.optim.Adam(agent.parameters()),
        _obs_to_tensors=NetOpponentTrainer._obs_to_tensors,
    )
    for path in (ROOT / "experiments/regimes").glob("*.json"):
        stage = TrainingRegime.model_validate_json(path.read_text()).stages[0]
        if stage.operation != "train_self_play":
            continue
        result = update_iteration(
            trainer, batch, stage.learning, 0.5, np.random.default_rng(7)
        )
        assert result["optimizer_exposures"] > 0
        assert np.isfinite(result["loss"])


def test_runtime_failure_is_a_retained_run(tmp_path, monkeypatch):
    from manabot.training import execution

    def unavailable(*args, **kwargs):
        raise RuntimeError("runtime fingerprint unavailable")

    monkeypatch.setattr(execution, "runtime_fingerprints", unavailable)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(RuntimeError, match="fingerprint"):
            execute_regime(recipe(), 197, tmp_path / "run", store)
        saved = json.loads((tmp_path / "run/run.json").read_text())
        assert saved["status"] == "failed"
        assert "fingerprint unavailable" in saved["error"]
        assert saved["stages"] == []
        assert store.training_run(saved["id"]).status == "failed"


def test_manifest_export_failure_is_retained_in_canonical_store(tmp_path, monkeypatch):
    from manabot.training import execution

    def no_space(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(execution, "export_training_run", no_space)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(OSError, match="disk full"):
            execute_regime(recipe(), 197, tmp_path / "run", store)
        row = store.con.execute("SELECT id FROM training_runs").fetchone()
        saved = store.training_run(row[0])
        assert saved.status == "failed"
        assert "disk full" in saved.error


def test_modified_models_are_revalidated_before_execution():
    value = recipe()
    value.stages[0].transitions = 0
    with pytest.raises(ValueError):
        validate_regime(value)


def test_cli_rejects_ambiguous_regimes_and_keeps_presets(monkeypatch):
    from typer.testing import CliRunner

    from manabot import cli

    runner = CliRunner()
    for flags in (["--preset", "local"], ["--set", "train.num_steps=2"]):
        result = runner.invoke(
            cli.app, ["train", "--regime", "unused.json", "--out", "unused", *flags]
        )
        assert result.exit_code != 0
        assert "cannot be combined" in result.output
    assert runner.invoke(cli.app, ["train", "--seed", "2"]).exit_code != 0
    calls = []
    monkeypatch.setattr(
        cli, "_run_train", lambda preset, overrides: calls.append((preset, overrides))
    )
    assert runner.invoke(cli.app, ["train"]).exit_code == 0
    assert calls == [(cli.DEFAULT_TRAIN_PRESET, [])]


def test_failed_stage_keeps_artifacts_and_exported_config(tmp_path, monkeypatch):
    from manabot.training import execution

    closed = []
    close = execution.Experiment.close

    def track_close(experiment):
        closed.append(experiment)
        close(experiment)

    monkeypatch.setattr(execution.Experiment, "close", track_close)
    value = recipe()
    value.agent.hidden_dim = 8
    value.agent.num_attention_heads = 2
    value.stages = value.stages[:1]
    value.stages[0].transitions = 4
    value.stages[0].updates = 1
    value.stages[0].learning.epochs = 1

    def reject_checkpoint(*args, **kwargs):
        raise ValueError("checkpoint admission failed")

    monkeypatch.setattr(execution, "load_checkpoint_agent", reject_checkpoint)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(ValueError, match="admission"):
            execute_regime(value, 197, tmp_path / "run", store)
        saved = json.loads((tmp_path / "run/run.json").read_text())
        assert saved["stages"][0]["status"] == "failed"
        assert saved["stages"][0]["seconds"] > 0
        assert saved["regime"]["agent"]["hidden_dim"] == 8
        assert (tmp_path / "run/policy-0-raw.pt").exists()
        # Rejected bytes remain on disk, never advertised as an admitted artifact.
        assert "raw" not in saved["stages"][0]["artifacts"]
        assert saved["stages"][0]["rejected_artifacts"]["raw"]["sha256"]
        assert len(closed) == 1


def test_continuation_rejects_mutated_artifact_bytes(tmp_path, monkeypatch):
    from manabot.training import execution

    value = recipe()
    value.agent.hidden_dim = 8
    value.agent.num_attention_heads = 2
    for stage in value.stages:
        stage.transitions = 4
        stage.updates = 1
        stage.learning.epochs = 1
    export = execution.export_training_run

    def corrupt_completed_input(run_id, store, out):
        run = export(run_id, store, out)
        if len(run.stages) == 1 and run.stages[0].status == "completed":
            artifact = Path(run.stages[0].artifacts["raw"]["path"])
            artifact.write_bytes(b"changed bytes")
        return run

    monkeypatch.setattr(execution, "export_training_run", corrupt_completed_input)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(ValueError, match="input artifact changed"):
            execute_regime(value, 197, tmp_path / "run", store)
        saved = json.loads((tmp_path / "run/run.json").read_text())
        assert saved["stages"][0]["status"] == "completed"
        assert saved["stages"][1]["status"] == "failed"
        assert saved["selected_artifact"] is None


def test_self_play_continuation_keeps_adam_and_exports_each_stage(
    tmp_path, monkeypatch
):
    from manabot.training import execution

    clock = execution.time.perf_counter
    offset = 0.0
    admitted = []
    export = execution.export_training_run

    def delayed_export(run_id, store, out):
        nonlocal offset
        run = export(run_id, store, out)
        completed = [stage for stage in run.stages if stage.status == "completed"]
        if len(completed) > len(admitted):
            admitted.append(completed[-1].cumulative_seconds)
            # Model slow persistence after admission without sleeping or training more.
            offset += 5.0
        return run

    monkeypatch.setattr(execution.time, "perf_counter", lambda: clock() + offset)
    monkeypatch.setattr(execution, "export_training_run", delayed_export)
    value = recipe()
    value.agent.hidden_dim = 8
    value.agent.num_attention_heads = 2
    for stage in value.stages:
        stage.transitions = 4
        stage.updates = 1
        stage.learning.epochs = 1
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(value, 197, tmp_path / "run", store)
        assert run.status == "completed"
        assert run.selected_artifact == run.stages[-1].artifacts["raw"]
        steps = []
        for stage in run.stages:
            state = torch.load(stage.artifacts["optimizer"]["path"], weights_only=True)
            steps.append(max(int(item["step"]) for item in state["state"].values()))
            assert stage.learner_transitions == 16
        assert steps[0] > 0
        assert steps[1] == 2 * steps[0]
        assert [stage.cumulative_seconds for stage in run.stages] == admitted
        assert admitted[0] >= run.setup_seconds + run.stages[0].seconds
        assert admitted[1] >= admitted[0] + 5.0 + run.stages[1].seconds
        assert run.seconds >= admitted[1] + 5.0
        saved = store.training_run(run.id)
        assert [stage.cumulative_seconds for stage in saved.stages] == admitted


def _fixture_run(updates: int) -> TrainingRun:
    stage = StageRecord(id="policy-0")
    stage.diagnostics = [
        {"update": n, "loss": n / 7, "rows": [n] * 8} for n in range(updates)
    ]
    return TrainingRun(
        id="fixture",
        regime=recipe(),
        regime_digest="test",
        seed=1,
        seed_streams={},
        identities={},
        stages=[stage],
    )


def test_update_save_work_does_not_grow_with_recorded_updates(tmp_path: Path) -> None:
    """Saving after one more update writes the same rows at any history length."""

    def writes_for_next_update(updates: int) -> int:
        run = _fixture_run(updates)
        with VerifyStore(tmp_path / f"{updates}.sqlite") as store:
            store.save_training_run(run)
            run.stages[0].diagnostics.append({"update": updates})
            before = store.con.total_changes
            store.save_training_run(run)
            written = store.con.total_changes - before
            assert store.training_run(run.id) == run
            return written

    assert writes_for_next_update(5) == writes_for_next_update(500)


def test_saved_diagnostics_follow_rewrites_truncation_and_old_layout(
    tmp_path: Path,
) -> None:
    run = _fixture_run(3)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        store.save_training_run(run)
        run.stages[0].diagnostics[-1]["coordinates"] = {"updates": 3}
        store.save_training_run(run)
        assert store.training_run(run.id) == run
        # Search stages keep one summary row and replace it on every save.
        run.stages[0].diagnostics = [{"games": 9}]
        store.save_training_run(run)
        assert store.training_run(run.id) == run

        # A store written before diagnostics had their own table keeps them inline.
        old = _fixture_run(4)
        old.id = "old-layout"
        store.con.execute(
            "INSERT INTO training_runs VALUES (?, ?)",
            (old.id, old.model_dump_json(exclude={"stages"})),
        )
        store.con.execute(
            "INSERT INTO training_stages VALUES (?, ?, ?)",
            (old.id, "policy-0", old.stages[0].model_dump_json()),
        )
        assert store.training_run(old.id) == old
        old.stages[0].diagnostics.append({"update": 4})
        store.save_training_run(old)
        assert store.training_run(old.id) == old
