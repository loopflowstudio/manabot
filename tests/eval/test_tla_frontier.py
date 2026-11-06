"""The c5fba4 frontier protocol pins two sizes and a milestone evaluation schedule."""

from collections import Counter
from decimal import Decimal
from pathlib import Path

from etude.experiment import (
    Checkpoint,
    Checkpoints,
    ExperimentState,
    TrainingRun,
    read_experiment,
)
from etude.jobs import evaluation_spec
from etude.run.records import ArtifactRef
import pytest

from experiments import tla_frontier
from experiments.tla_frontier import (
    EXPERIMENT,
    HEURISTIC,
    LARGE,
    REGIME,
    ROOT,
    SMALL,
    SizeLabel,
    comparison_schedule,
    eval,
    frozen,
    runs,
)
from manabot.env.observation import ObservationSpace
from manabot.model.actor_critic import ActorCritic
from manabot.training.regime import TrainingRegime, TrainSelfPlay


def _checkpoint(label: SizeLabel, update: int) -> Checkpoint:
    run = f"tla-frontier-c5fba4-{label}-seed21501"
    return Checkpoint(
        run=run,
        experiment=EXPERIMENT,
        ordinal=update + 1,
        artifact=ArtifactRef(
            uri=f"file:///fixture/{label}/updates-{update}.pt",
            sha256="a" * 64,
            size_bytes=1,
        ),
        clocks={"updates": update},
        line=run,
    )


def _state(reached: dict[SizeLabel, int], *, finished: bool = True) -> ExperimentState:
    milestones = (0, *frozen()[1].milestones)
    return ExperimentState(
        experiment=EXPERIMENT,
        runs=tuple(
            TrainingRun(
                name=f"tla-frontier-c5fba4-{label}-seed21501",
                experiment=EXPERIMENT,
                state="finished" if finished else "running",
                checkpoints=Checkpoints(
                    _checkpoint(label, update)
                    for update in sorted(
                        {u for u in milestones if u <= end}
                        | ({end} if finished else set())
                    )
                ),
            )
            for label, end in reached.items()
        ),
    )


def _compute_parameters(regime: TrainingRegime) -> int:
    """Parameters outside the 4096-bucket hashed lookup tables."""
    agent = ActorCritic(ObservationSpace(regime.observation), regime.agent)
    return sum(
        p.numel()
        for p in agent.parameters()
        if not (p.dim() == 2 and p.shape[0] == 4096)
    )


def _recipe(regime: TrainingRegime) -> dict[str, object]:
    """The regime without its identity and compute-network size."""
    size = {"hidden_dim", "attention_layers", "num_attention_heads"}
    return regime.model_dump(mode="json", exclude={"id": True, "agent": size})


def test_sizes_differ_only_in_compute_network() -> None:
    regimes, _ = frozen()
    small, large = regimes[SMALL], regimes[LARGE]
    assert _recipe(small) == _recipe(large)
    assert (small.agent.hidden_dim, small.agent.attention_layers) == (64, 2)
    assert (large.agent.hidden_dim, large.agent.attention_layers) == (256, 4)
    ratio = _compute_parameters(large) / _compute_parameters(small)
    assert 10 <= ratio <= 100


def test_frozen_bytes_declare_the_protocol() -> None:
    regimes, protocol = frozen()
    regime = regimes[SMALL]
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    assert stage.learning.gradient == "damped_policy"
    assert stage.game_errors is not None and stage.game_errors.recover_capacity
    assert stage.updates > max(protocol.milestones)
    assert regime.wall_seconds == stage.execution.wall_seconds
    assert regime.world == protocol.runtime_world.world_version
    observation = regime.observation
    assert observation.policy_input_version == 9
    assert observation.action_overflow == HEURISTIC["action_overflow"]
    assert observation.max_actions == HEURISTIC["max_actions"]
    source = regime.deck_source
    assert source is not None and source.entry_ids is not None
    assert source.sampling == "uniform-independent-v1"
    assert len(protocol.pairs) == 16
    assert all(max(p.first, p.second) < len(source.decks) for p in protocol.pairs)
    assert protocol.max_paid_usd == Decimal("100")
    assert 2 * protocol.training_usd + protocol.evaluation_usd <= protocol.max_paid_usd


def test_frozen_rejects_a_changed_regime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = tmp_path / "w64.json"
    changed.write_bytes(
        REGIME.read_bytes().replace(b'"hidden_dim": 64', b'"hidden_dim": 128')
    )
    monkeypatch.setattr(tla_frontier, "REGIME", changed)
    with pytest.raises(ValueError, match="regime differs"):
        frozen()


def test_milestones_play_both_sizes_against_heuristic_and_each_other() -> None:
    schedule = comparison_schedule(_state({SMALL: 1300, LARGE: 450}), frozen()[1])
    playable = [c for c in schedule if c.matchup is not None]
    names = {c.matchup.name for c in playable if c.matchup is not None}
    assert "w64 u0 vs heuristic" in names and "w256 u0 vs heuristic" in names
    assert "w256 u400 vs w64 u400" in names
    assert "w256 u600 vs w64 u600" not in names
    assert "w64 u1200 vs heuristic" in names
    assert {"w64 final u1300 vs heuristic", "w256 final u450 vs heuristic"} <= names
    assert {"w64 final u1300 vs w64 u0", "w256 final u450 vs w256 u0"} <= names
    assert "w256 final vs w64 final" in names
    # Unreached milestones stay expected for analysis.
    assert any(
        c.update == 2400 and c.tag == "head-to-head" and c.matchup is None
        for c in schedule
    )
    tags = Counter(c.tag for c in playable)
    assert tags["head-to-head"] == 6  # 25, 50, 100, 200, 300, 400
    assert len(names) == len(playable)


def test_running_runs_evaluate_reached_milestones_only() -> None:
    state = _state({SMALL: 100, LARGE: 50}, finished=False)
    schedule = comparison_schedule(state, frozen()[1])
    names = {c.matchup.name for c in schedule if c.matchup is not None}
    assert "w256 u50 vs w64 u50" in names
    assert not any("final" in name for name in names)
    (job,) = eval(state)
    assert job.max_usd == Decimal("2")


def test_no_checkpoints_schedules_nothing() -> None:
    assert eval(ExperimentState(experiment=EXPERIMENT)) == []


def test_training_runs_admit_milestones_and_wall_budget() -> None:
    launched = runs("HEAD")
    assert [run.job.name for run in launched] == [
        "tla-frontier-c5fba4-w64",
        "tla-frontier-c5fba4-w256",
    ]
    assert {run.job.until for run in launched} == {"updates=3000"}


def test_discovery_keeps_evaluation_callback_importable() -> None:
    experiment = read_experiment(ROOT / "experiments/tla_frontier.py")
    job = experiment.eval(_state({SMALL: 25, LARGE: 25}, finished=False))[0]
    callback = evaluation_spec(job).play
    assert callback.module == "experiments.tla_frontier"
    assert callable(callback.load(job.repo))
