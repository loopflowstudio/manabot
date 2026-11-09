"""MTG-252 readout arms vary only the critic readout of the re-pinned 183 recipe."""

import json
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
import torch

from experiments import learning_signal, value_readout_252
from experiments.learning_signal import ELEMENTAL_PAIRS
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.experiment.job import LearnerContinuation, StagedExport, TrainingJob
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    RewardConfig,
    ValueAggregation,
)
from manabot.model.actor_critic import ActorCritic
from manabot.recipes.builders import with_agent
from manabot.training.rollout.collection import SeatRoutedCollector


def test_arms_change_only_the_value_readout() -> None:
    baseline = value_readout_252.base()
    for aggregation in value_readout_252.ARMS:
        regime = value_readout_252.arm(aggregation)
        assert regime.agent.value_aggregation == aggregation
        assert regime.stages == baseline.stages
        assert regime.observation == baseline.observation
    wider = with_agent(
        baseline,
        id="wider",
        agent=baseline.agent.model_copy(
            update={"value_aggregation": "value_token", "hidden_dim": 128}
        ),
    )
    with pytest.raises(ValueError, match="more than agent.value_aggregation"):
        value_readout_252.require_single_axis(baseline, wider)


@pytest.mark.parametrize("aggregation", value_readout_252.ARMS)
def test_the_depth_arm_changes_only_the_attention_depth(
    aggregation: ValueAggregation,
) -> None:
    shallow = value_readout_252.arm(aggregation, 150)
    deep = value_readout_252.deeper(aggregation, 4, 150)
    assert deep.agent.attention_layers == 4 != shallow.agent.attention_layers
    assert deep.agent.value_aggregation == aggregation
    assert deep.stages == shallow.stages
    assert deep.observation == shallow.observation


def test_depth_runs_are_scored_and_meet_their_shallow_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        value_readout_252, "DEEPER", {("masked_mean", 25201): "deep25201"}
    )
    state = ExperimentState(
        experiment=value_readout_252.EXPERIMENT,
        runs=(
            _stored("deep25201", 20, 40, 75, 150),
            _stored(value_readout_252.FINALS["masked_mean", 25201], 150),
        ),
    )
    matchups = value_readout_252.evaluation_matchups(state)
    deep = [m for m in matchups if "depth4" in m.name and "heuristic" in m.tags]
    assert len(deep) == 4
    (versus,) = [m for m in matchups if "depth" in m.tags]
    assert versus.name == "masked_mean v depth4 s25201 u150"

    # A continuation republishes its initial state at 150 under a new run key.
    # Existing matchup names must keep their originally scored checkpoint.
    monkeypatch.setattr(
        value_readout_252, "DEEPER_THIRDS", {("masked_mean", 25201): "continued25201"}
    )
    continued = ExperimentState(
        experiment=state.experiment,
        runs=(*state.runs, _stored("continued25201", 150, 200, 250, 300)),
    )
    later = {m.name: m for m in value_readout_252.evaluation_matchups(continued)}
    for original in matchups:
        assert later[original.name] == original


def test_depth_continuation_and_fresh_token_share_the_scoring_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        value_readout_252,
        "DEEPER",
        {("masked_mean", 25201): "mean150", ("value_token", 25201): "token300"},
    )
    monkeypatch.setattr(
        value_readout_252, "DEEPER_THIRDS", {("masked_mean", 25201): "mean300"}
    )
    state = ExperimentState(
        experiment=value_readout_252.EXPERIMENT,
        runs=(
            _stored("mean150", 20, 40, 75, 150),
            _stored("mean300", 200, 250, 300),
            _stored("token300", 20, 40, 75, 150, 200, 250, 300),
            *(
                _stored(value_readout_252.THIRDS[aggregation, 25201], 300)
                for aggregation in ("masked_mean", "value_token")
            ),
        ),
    )
    matchups = value_readout_252.evaluation_matchups(state)
    deep = [m for m in matchups if "depth4" in m.name and "heuristic" in m.tags]
    assert len(deep) == 14
    names = {m.name for m in matchups}
    assert "masked_mean v depth4 s25201 u300" in names
    assert "value_token v value_token depth4 s25201 u300" in names
    assert "masked_mean depth4 v value_token depth4 s25201 u150" in names
    assert "masked_mean depth4 v value_token depth4 s25201 u300" in names


def test_depth_runs_restore_the_learner_and_export_scored_updates(
    tmp_path: Path,
) -> None:
    export = StagedExport(key="k", sha256="0" * 64, bytes=1)
    continuation = LearnerContinuation(
        from_update=150,
        source_stage="self-play",
        source_run=export,
        raw=export,
        ema=export,
        optimizer=export,
    )
    staged = tmp_path / "continuations.json"
    parent = value_readout_252.DEEPER["masked_mean", 25201]
    staged.write_text(json.dumps({parent: continuation.model_dump(mode="json")}))
    for aggregation, inputs, expected in (
        ("masked_mean", str(staged), (200, 250)),
        ("value_token", "", (20, 40, 75, 150, 200, 250)),
    ):
        (run,) = value_readout_252.runs(
            revision="HEAD",
            seeds="25201",
            arms=aggregation,
            depth=4,
            updates=300,
            hours=3.0,
            continuations=inputs,
            recover_capacity="yes",
        )
        job = TrainingJob.model_validate_json(run.job.config)
        assert job.regime.agent.attention_layers == 4
        assert job.regime.agent.value_aggregation == aggregation
        assert job.updates == 300
        assert job.milestones == expected
        assert job.regime.stages[0].game_errors.recover_capacity
        assert job.continuation == (continuation if inputs else None)


def test_static_costs_and_initialization_sharing() -> None:
    costs = {cost.aggregation: cost for cost in value_readout_252.costs()}
    masked, token, readout = (costs[arm] for arm in value_readout_252.ARMS)
    # 183: two player objects, 60 cards and 40 permanents per seat.
    assert masked.objects == 2 + 2 * 60 + 2 * 40
    assert masked.added_parameters == 0 and masked.shares_initialization
    assert token.added_parameters == 64
    assert token.attention_pairs == 2 * (masked.objects + 1) ** 2
    # Query 64 + in-projection 3*64*64+3*64 + out-projection 64*64+64.
    assert readout.added_parameters == 16_704
    assert readout.attention_pairs == masked.attention_pairs + masked.objects
    assert readout.shares_initialization
    # The token's draw precedes later input encoders, shifting their initial values.
    assert not token.shares_initialization


def test_every_arm_runs_under_the_same_seeds() -> None:
    grid = value_readout_252.grid([25201, 25202], updates=150)
    for seed in (25201, 25202):
        assert sorted(r.aggregation for r in grid if r.seed == seed) == sorted(
            value_readout_252.ARMS
        )
    assert all(r.regime.stages[0].updates == 150 for r in grid)
    with pytest.raises(ValueError, match="distinct seeds"):
        value_readout_252.grid([25201, 25201])


def test_pairings_depend_on_the_run_seed_not_the_arm() -> None:
    """Different readouts play different games yet meet the same pairing per game.

    Records the native pair of each stream's `k`th game root under two arms'
    policies; every (stream, k) seen by both must hold the same pair. Under
    complete-games closure each collect here finishes one game per stream, so
    the arms also stay aligned game for game.
    """
    torch.set_num_threads(1)
    space = ObservationSpace()
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    decisions: list[int] = []
    for torch_seed, aggregation in ((1, "masked_mean"), (2, "attention_readout")):
        torch.manual_seed(torch_seed)
        agent = ActorCritic(
            space,
            ModelConfig(
                hidden_dim=8,
                num_attention_heads=2,
                world="elemental-world-v2",
                value_aggregation=aggregation,
            ),
        )
        collector = SeatRoutedCollector(
            space,
            Match(
                MatchConfig.authored(
                    "elemental-world-v2", "elemental_ug", "elemental_bg"
                )
            ),
            Reward(RewardConfig()),
            num_envs=4,
            seed=25201 + 10000,
            opponent_mode="self",
            learning_seats="both",
            matchup_curriculum="elemental-uniform",
        )
        for _ in range(4):
            games = collector.stats.games_by_stream
            for stream, pair in enumerate(collector._env.current_matchup_indices()):
                key = (stream, games.get(stream, 0))
                assert seen.setdefault(key, pair) == pair
            collector.collect(agent, 64)
        decisions.append(collector.stats.micro_steps)
    # The policies played different games over the same pairings.
    assert decisions[0] != decisions[1]


def test_a_segment_continues_its_parent_with_only_the_watchdog_changed() -> None:
    export = StagedExport(key="k", sha256="0" * 64, bytes=1)
    continuation = LearnerContinuation(
        from_update=111,
        source_stage="self-play",
        source_run=export,
        raw=export,
        ema=export,
        optimizer=export,
    )
    run = value_readout_252.grid([25201], updates=150)[0]

    job = value_readout_252.segment(run, continuation, hours=1.5)

    assert job.continuation == continuation
    assert job.regime.wall_seconds == 1.5 * 3600 - 900 - 300
    left = run.regime.model_dump(mode="json")
    right = job.regime.model_dump(mode="json")
    for dump in (left, right):
        dump.pop("wall_seconds")
        dump["stages"][0]["execution"].pop("wall_seconds")
    assert left == right


def test_every_arm_and_seed_has_one_parent() -> None:
    runs = value_readout_252.grid([25201, 25202, 25203])
    parents = value_readout_252.PARENTS
    assert set(parents) == {(run.aggregation, run.seed) for run in runs}
    assert len(set(parents.values())) == len(parents)


def test_capacity_recovery_changes_only_that_segment() -> None:
    export = StagedExport(key="k", sha256="0" * 64, bytes=1)
    continuation = LearnerContinuation(
        from_update=111,
        source_stage="self-play",
        source_run=export,
        raw=export,
        ema=export,
        optimizer=export,
    )
    run = value_readout_252.grid([25202], updates=150)[1]

    job = value_readout_252.segment(run, continuation, 1.5, recover_capacity=True)

    assert job.regime.stages[0].game_errors.recover_capacity
    assert not run.regime.stages[0].game_errors.recover_capacity


def _stored(run: str, *updates: int) -> TrainingRun:
    return TrainingRun(
        name=run,
        experiment=value_readout_252.EXPERIMENT,
        state="finished",
        checkpoints=Checkpoints(
            Checkpoint(
                run=run,
                experiment=value_readout_252.EXPERIMENT,
                ordinal=ordinal,
                artifact=ArtifactRef(
                    uri=f"r2://bucket/{run}/updates-{update}.pt",
                    sha256="a" * 64,
                    size_bytes=1,
                ),
                clocks={"updates": float(update)},
                line=run,
            )
            for ordinal, update in enumerate(updates, start=1)
        ),
    )


def test_evaluation_scores_first_segments_and_finals_on_one_schedule() -> None:
    state = ExperimentState(
        experiment=value_readout_252.EXPERIMENT,
        runs=(
            *(
                _stored(parent, 20, 40, 75, 111)
                for parent in value_readout_252.PARENTS.values()
            ),
            *(_stored(final, 150) for final in value_readout_252.FINALS.values()),
        ),
    )
    matchups = value_readout_252.evaluation_matchups(state)
    heuristic = [m for m in matchups if "heuristic" in m.tags]
    head_to_head = [m for m in matchups if "head-to-head" in m.tags]
    # 9 runs x (20, 40, 75 from the first segment, 150 from its continuation);
    # the first segment's own end (111) is not scored.
    assert len(heuristic) == 36
    assert not any("u111" in m.name for m in heuristic)
    assert len(head_to_head) == 9
    schedule = matchups[0].conditions
    assert all(m.conditions == schedule for m in matchups)
    assert len(schedule) == len(ELEMENTAL_PAIRS.pairings)
    assert {m.games for m in matchups} == {ELEMENTAL_PAIRS.legs}


def test_an_unstored_final_is_left_for_a_later_pass() -> None:
    state = ExperimentState(
        experiment=value_readout_252.EXPERIMENT,
        runs=tuple(
            _stored(parent, 20, 40, 75) for parent in value_readout_252.PARENTS.values()
        ),
    )
    matchups = value_readout_252.evaluation_matchups(state)
    assert len(matchups) == 27
    assert not any("head-to-head" in m.tags for m in matchups)


def test_the_discovered_evaluation_callback_stays_importable() -> None:
    experiment = read_experiment(
        value_readout_252.ROOT / "experiments/value_readout_252.py"
    )
    state = ExperimentState(
        experiment=value_readout_252.EXPERIMENT,
        runs=tuple(
            _stored(parent, 20) for parent in value_readout_252.PARENTS.values()
        ),
    )
    (job,) = experiment.eval(state)  # type: ignore[arg-type]
    callback = evaluation_spec(job).play
    assert callback.module == "experiments.learning_signal"
    assert callback.load(job.repo) is learning_signal.play_deal
    assert "--extra evaluation" in job.command
