"""Direct storage preserves the frozen collector's trajectories and learning."""

from contextlib import ExitStack, closing
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from experiments.runners.collector_reference import (
    collector_implementation,
    frozen_collector,
)
from experiments.runners.inference_preparation import (
    checkpoint_witnesses,
    collection_witnesses,
)
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import RewardHypers
from manabot.model.agent import Agent
from manabot.sim.net_opponent import RolloutBatch, SeatRoutedCollector
from manabot.training import execution
from manabot.training.models import TrainSelfPlay
from manabot.verify.store import VerifyStore
from tests.training.test_recovery import (
    assert_learning_diagnostics_equal,
    assert_state_equal,
    recipe,
)


def _paired_collect(
    reference: SeatRoutedCollector,
    direct: SeatRoutedCollector,
    agent: Agent,
    steps: int,
) -> tuple[RolloutBatch, RolloutBatch]:
    # Frozen BatchedSampler uses Torch's global generator, unlike the collector's
    # owned self-play generator. Both executions must start at the same state.
    before = torch.get_rng_state()
    expected = reference.collect(agent, steps)
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    actual = direct.collect(agent, steps)
    assert_state_equal(after, torch.get_rng_state())
    return expected, actual


@pytest.mark.parametrize("mode", ["self", "random", "frozen"])
@pytest.mark.parametrize("categorical", [False, True])
def test_storage_matches_frozen_collector_and_owns_returned_batches(
    mode: str, categorical: bool
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(197)
    value = recipe()
    value.agent.value_kind = "categorical_wdl" if categorical else "scalar"
    space = ObservationSpace(value.observation)
    agent = Agent(space, value.agent).eval()
    opponent = Agent(space, value.agent).eval() if mode == "frozen" else None
    with ExitStack() as owners:
        collectors = [
            owners.enter_context(
                closing(
                    kind(
                        space,
                        Match(value.match),
                        Reward(RewardHypers()),
                        num_envs=4,
                        seed=97,
                        opponent_mode=mode,
                        opponent_agent=opponent,
                        workers=2,
                        recovery_max_game_microsteps=100_000
                        if mode == "self"
                        else None,
                    )
                )
            )
            for kind in (frozen_collector(), SeatRoutedCollector)
        ]
        reference, direct = collectors
        first = _paired_collect(reference, direct, agent, 128)
        assert_state_equal(vars(first[0]), vars(first[1]))
        saved = deepcopy(vars(first[1]))
        left_batch, right_batch = _paired_collect(reference, direct, agent, 17)
        assert_state_equal(vars(left_batch), vars(right_batch))
        assert_state_equal(saved, vars(first[1]))
        expected, actual = asdict(reference.stats), asdict(direct.stats)
        expected.pop("seconds")
        actual.pop("seconds")
        assert expected == actual
        assert direct.stats.games > 0
        if mode == "self":
            left, right = reference.snapshot(), direct.snapshot()
            left.stats.seconds = right.stats.seconds = 0
            assert_state_equal(asdict(left), asdict(right))


def test_failed_collection_cannot_be_snapshotted_or_silently_reused() -> None:
    torch.set_num_threads(1)
    value = recipe()
    space = ObservationSpace(value.observation)
    agent = Agent(space, value.agent).eval()
    with closing(
        SeatRoutedCollector(
            space,
            Match(value.match),
            Reward(RewardHypers()),
            num_envs=4,
            opponent_mode="self",
            recovery_max_game_microsteps=100_000,
        )
    ) as collector:
        calls = 0

        def interrupt() -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TimeoutError("injected deadline")

        with pytest.raises(TimeoutError, match="injected deadline"):
            collector.collect(agent, 8, check=interrupt)
        with pytest.raises(ValueError, match="update boundary"):
            collector.snapshot()
        with pytest.raises(ValueError, match="unfinished rollout"):
            collector.collect(agent, 8)


def test_storage_preserves_complete_training_against_frozen_reference(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    value = recipe()
    stage = value.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.streams = 4
    stage.execution.workers = 4
    runs = []
    witnesses = []
    collections = []
    for reference in (True, False):
        root = tmp_path / str(reference)
        root.mkdir()
        with collector_implementation(reference) as kind:
            with (
                collection_witnesses(root, kind),
                VerifyStore(root / "verify.sqlite") as store,
            ):
                run = execution.execute_regime(value, 197, root / "run", store)
        assert run.status == "completed"
        runs.append(run)
        witnesses.append(checkpoint_witnesses(root / "run"))
        collections.append(
            [
                json.loads(p.read_text())
                for p in sorted(root.glob("collection-witness-*.json"))
            ]
        )
    assert witnesses[0] == witnesses[1]
    assert len(collections[0]) == len(collections[1]) == 3
    assert collections[0] == collections[1]
    assert_learning_diagnostics_equal(runs[0].stages[0], runs[1].stages[0])
    assert execution.SeatRoutedCollector is SeatRoutedCollector
