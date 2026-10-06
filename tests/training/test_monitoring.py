"""Dashboard replay and tiny real training preserve metric and learning meaning."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.infra.hypers import AgentSpec, MatchHypers
from manabot.model.agent import Agent
from manabot.sim.distill import VISIT_COUNT_KEY, load_shards
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.search_supervised import (
    CHOSEN_ACTION_TARGET,
    SearchSupervisedEpochStats,
    evaluate_search_supervised,
    train_search_supervised,
)
from manabot.training import execution, monitoring
from manabot.training.execution import execute_regime
from manabot.training.models import (
    CollectSearch,
    StageRecord,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
    TrainSupervised,
)
from manabot.training.monitoring import (
    Dashboard,
    append_history,
    default_panels,
    training_dashboard,
)
from manabot.verify.store import VerifyStore
from manabot.verify.util import INTERACTIVE_DECK
from tests.sim.test_search_supervised import _dataset


def _recipe() -> TrainingRegime:
    value = TrainingRegime.model_validate_json(
        Path("experiments/regimes/direct-self-play.json").read_text()
    )
    value.stages = value.stages[:1]
    stage = value.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 2
    stage.transitions = 16
    stage.streams = 2
    stage.learning.epochs = 1
    value.agent.hidden_dim = 8
    value.agent.num_attention_heads = 2
    value.schedule_clock = "iteration_fraction"
    return value


class RetainedSink:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    @property
    def step(self) -> int:
        return len(self.rows)

    def log(self, data: dict[str, object], *, step: int) -> None:
        assert step == len(self.rows)
        self.rows.append(data)


def test_backfill_preserves_missing_coordinates_and_resumes_suffix() -> None:
    run = TrainingRun(
        id="saved-fixture",
        regime=_recipe(),
        regime_digest="fixture",
        seed=197,
        seed_streams={},
        identities={},
        stages=[StageRecord(id="policy-0", diagnostics=[{"policy_loss": -0.4}])],
    )
    dashboard = training_dashboard(run)
    assert dashboard.summary["history_incomplete"] is True
    assert "progress/training_seconds" not in dashboard.rows[0]
    assert dashboard.rows[0]["rl/policy_loss"] == -0.4
    assert dashboard.rows[0]["availability/rl/value_loss"] is False
    sink = RetainedSink()
    append_history(dashboard, sink)
    append_history(dashboard, sink)
    assert len(sink.rows) == 1
    run.stages[0].diagnostics.append({"policy_loss": -0.3})
    append_history(training_dashboard(run), sink)
    assert len(sink.rows) == 2
    with pytest.raises(ValueError, match="longer"):
        append_history(dashboard, sink)


def test_fixed_validation_callback_and_leak_rejection() -> None:
    torch.set_num_threads(1)
    dataset = _dataset()
    mask = dataset["game_index"] == 0
    fixed = {k: v[mask] for k, v in dataset.items()}
    seen: list[SearchSupervisedEpochStats] = []

    def record(stats: SearchSupervisedEpochStats, model: Agent) -> None:
        assert isinstance(model, Agent)
        seen.append(stats)

    _, _, _, history = train_search_supervised(
        dataset,
        epochs=2,
        validation_games={0, 1},
        batch_size=32,
        fixed_validation_dataset=fixed,
        on_epoch=record,
    )
    assert seen == history
    assert all(s.fixed_validation is not None for s in seen)
    assert all(np.isfinite(s.train_policy_kl) for s in seen)
    with pytest.raises(ValueError, match="excluded"):
        train_search_supervised(
            dataset, epochs=1, validation_games={1}, fixed_validation_dataset=fixed
        )


def test_monitor_exports_do_not_change_learning_and_survive_reload(
    tmp_path: Path,
) -> None:
    recipe = _recipe()
    with VerifyStore(tmp_path / "runs.sqlite") as store:
        baseline = execute_regime(recipe, 197, tmp_path / "plain", store)
        monitored = execute_regime(
            recipe, 197, tmp_path / "monitored", store, checkpoint_seconds=1e-9
        )
        assert store.training_run(monitored.id) == monitored
    assert len(monitored.monitoring_checkpoints) == 2
    for checkpoint in monitored.monitoring_checkpoints:
        assert checkpoint.error is None and checkpoint.artifact is not None
        load_checkpoint_agent(checkpoint.artifact["path"])
    plain, _ = load_checkpoint_agent(baseline.selected_artifact["path"])
    tracked, _ = load_checkpoint_agent(monitored.selected_artifact["path"])
    for key, weights in plain.state_dict().items():
        torch.testing.assert_close(weights, tracked.state_dict()[key], rtol=0, atol=0)
    dashboard = training_dashboard(monitored)
    assert dashboard.summary["history_incomplete"] is False
    assert len(dashboard.rows) == 2
    assert (
        dashboard.rows[1]["progress/learner_transitions"]
        > dashboard.rows[0]["progress/learner_transitions"]
    )
    assert dashboard.rows[0]["availability/rl/policy_loss"] is True


def test_default_wandb_panels_and_offline_roundtrip() -> None:
    dashboard = Dashboard(
        run_id="offline-fixture",
        config={},
        summary={},
        rows=[
            {
                "progress/observation": 0,
                "distillation/train_cross_entropy": 0.3,
                "distillation/growing_cross_entropy": 0.4,
                "distillation/fixed_cross_entropy": 0.5,
            }
        ],
    )
    reloaded = Dashboard.model_validate_json(dashboard.model_dump_json())
    assert reloaded == dashboard
    panels = default_panels(reloaded)
    assert len(panels) == 1
    assert "dashboard/Teacher cross-entropy (nats)" in panels


def test_global_game_ids_preserve_fixed_membership_as_shards_grow(
    tmp_path: Path,
) -> None:
    source = _dataset()
    paths: list[Path] = []
    for round_index, offset in enumerate((0, 100)):
        data = {key: value.copy() for key, value in source.items()}
        data["game_index"] += offset
        path = tmp_path / f"shard-{round_index}.npz"
        np.savez_compressed(
            path, **data, provenance=np.array(json.dumps({"round": round_index}))
        )
        paths.append(path)
    first = load_shards(paths[:1], globally_unique_games=True)
    growing = load_shards(paths, globally_unique_games=True)
    np.testing.assert_array_equal(
        growing["game_index"][: len(first["game_index"])], first["game_index"]
    )
    assert set(growing["game_index"]) == set(range(8)) | set(range(100, 108))
    with pytest.raises(ValueError, match="overlap"):
        load_shards([paths[0], paths[0]], globally_unique_games=True)


def test_executor_streams_fixed_and_growing_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _dataset()
    mask = source["game_index"] == 0
    sample = {key: value[mask].copy() for key, value in source.items()}
    sample[VISIT_COUNT_KEY] = (sample["actions_valid"] > 0).astype(np.float32)

    def collect(**kwargs: object) -> dict[str, object]:
        path, game = kwargs["out_path"], kwargs["game_offset"]
        assert isinstance(path, Path) and isinstance(game, int)
        data = {key: value.copy() for key, value in sample.items()}
        data["game_index"][:] = game
        np.savez_compressed(path, **data)
        return {
            "terminated": [True],
            "truncated": [False],
            "decisions": len(data["game_index"]),
        }

    monkeypatch.setattr(execution, "generate_selfplay_shard", collect)
    recipe = TrainingRegime(
        id="distillation-fixture",
        world="w4",
        match=MatchHypers(
            hero_deck=dict(INTERACTIVE_DECK), villain_deck=dict(INTERACTIVE_DECK)
        ),
        agent=AgentSpec(hidden_dim=8, num_attention_heads=2),
        stages=[
            CollectSearch(id="data-0", operation="collect_search", games=2),
            TrainSupervised(
                id="policy-0",
                operation="train_supervised",
                datasets=["data-0"],
                epochs=1,
            ),
            CollectSearch(id="data-1", operation="collect_search", games=10),
            TrainSupervised(
                id="policy-1",
                operation="train_supervised",
                datasets=["data-0", "data-1"],
                initial="policy-0",
                epochs=1,
            ),
        ],
    )
    with VerifyStore(tmp_path / "run.sqlite") as store:
        run = execute_regime(
            recipe, 197, tmp_path / "run", store, checkpoint_seconds=1e-9
        )
    assert run.fixed_validation is not None
    assert run.fixed_validation.games == [0]
    assert run.updates_through() == 2
    assert [checkpoint.updates for checkpoint in run.monitoring_checkpoints] == [1, 2]
    for stage in (run.stages[1], run.stages[3]):
        assert stage.diagnostics[0]["fixed_validation"]["policy_loss"] >= 0
        assert stage.diagnostics[0]["validation"]["policy_loss"] >= 0
        assert stage.diagnostics[0]["coordinates"]["optimizer_exposures"] > 0
    assert training_dashboard(run).summary["history_incomplete"] is False


def test_publisher_uses_remote_prefix_and_rejects_rewritten_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Remote(RetainedSink):
        url = "https://wandb.invalid/fixture"

        def __init__(self) -> None:
            super().__init__()
            self.summary: dict[str, object] = {}

        def define_metric(self, name: str, **kwargs: object) -> None:
            pass

        def finish(self) -> None:
            pass

    remote = Remote()

    def initialize(**kwargs: object) -> Remote:
        assert kwargs["resume"] == "allow"
        assert kwargs["mode"] == "online"
        return remote

    monkeypatch.setattr(monitoring.wandb, "init", initialize)
    dashboard = Dashboard(
        run_id="fixture",
        config={},
        summary={},
        rows=[{"progress/observation": 0, "rl/entropy": 0.5}],
    )
    monitoring.publish_dashboard(dashboard, tmp_path)
    monitoring.publish_dashboard(dashboard, tmp_path)
    assert remote.step == 1
    assert any(name.startswith("dashboard/") for name in remote.rows[0])
    dashboard.rows.append({"progress/observation": 1, "rl/entropy": 0.4})
    monitoring.publish_dashboard(dashboard, tmp_path)
    assert remote.step == 2
    dashboard.rows[0]["rl/entropy"] = 0.6
    with pytest.raises(ValueError, match="published prefix"):
        monitoring.publish_dashboard(dashboard, tmp_path)


def test_fixed_reference_target_does_not_change_training_objective() -> None:
    torch.set_num_threads(1)
    dataset = _dataset()
    fixed = {key: value[dataset["game_index"] == 0] for key, value in dataset.items()}
    ordinary, _, _, _ = train_search_supervised(dataset, epochs=1, validation_games={0})
    monitored, _, _, history = train_search_supervised(
        dataset,
        epochs=1,
        validation_games={0},
        fixed_validation_dataset=fixed,
        fixed_validation_target_kind=CHOSEN_ACTION_TARGET,
    )
    for name, value in ordinary.state_dict().items():
        torch.testing.assert_close(value, monitored.state_dict()[name], rtol=0, atol=0)
    expected = evaluate_search_supervised(
        monitored,
        fixed,
        np.arange(len(fixed["game_index"])),
        policy_temperature=0.05,
        policy_target_kind=CHOSEN_ACTION_TARGET,
    )
    assert history[0].fixed_validation == expected
