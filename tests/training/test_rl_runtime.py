"""Bounded native self-play, persisted costs and ordinary raw/EMA reload."""

from pathlib import Path
from typing import Literal

import pytest
import torch

from experiments.direct_self_play import direct_self_play
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import execute_regime
from manabot.training.models import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore
import managym


def bounded_regime() -> TrainingRegime:
    recipe = direct_self_play()
    recipe.world = managym.WORLD_VERSION
    recipe.id = f"test-{recipe.world}-{recipe.id}"
    recipe.agent.hidden_dim = 8
    recipe.agent.num_attention_heads = 2
    recipe.stages = recipe.stages[:2]
    for stage in recipe.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.updates = 1
        stage.streams = 4
        stage.transitions = 128
        stage.learning.epochs = 1
        stage.learning.minibatches = 2
        stage.learning.ema = 0.5
        stage.execution.wall_seconds = 120
    recipe.wall_seconds = 300
    return recipe


@pytest.mark.parametrize("empty_second", [False, True])
@pytest.mark.parametrize("learning_seats", ["one", "both"])
def test_self_play_records_costs_and_reloads_distinct_raw_ema(
    tmp_path: Path, empty_second: bool, learning_seats: Literal["one", "both"]
) -> None:
    recipe = bounded_regime()
    for stage in recipe.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.learning_seats = learning_seats
    if empty_second:
        recipe.stages[1].learning.min_advantage = 100
    with VerifyStore(tmp_path / "runs.sqlite") as store:
        run = execute_regime(recipe, 197, tmp_path / "run", store)
        assert store.training_run(run.id).status == "completed"
    assert run.status == "completed"
    assert sum(stage.games for stage in run.stages) > 0
    for index, stage in enumerate(run.stages):
        outcomes = stage.diagnostics[0]["self_play_outcomes"]
        assert (
            sum(r["wins"] + r["losses"] + r["draws"] for r in outcomes)
            == 2 * stage.games
        )
        assert sum(r["wins"] for r in outcomes) == sum(r["losses"] for r in outcomes)
        assert {r["position"] for r in outcomes} == {"play", "draw"}
        if learning_seats == "one":
            assert stage.learner_transitions == 512
        else:
            assert stage.learner_transitions >= 512
            assert stage.learner_transitions == stage.environment_decisions
        assert stage.collection_seconds > 0
        assert stage.learning_seconds > 0
        assert stage.export_seconds > 0
        if empty_second and index == 1:
            assert stage.optimizer_exposures == 0
            assert stage.diagnostics[0]["skipped"] == "empty advantage filter"
        else:
            assert stage.optimizer_exposures > 0
        assert stage.artifacts["raw"]["sha256"] != stage.artifacts["ema"]["sha256"]
    raw0, ema0, raw1, ema1 = [
        load_checkpoint_agent(run.stages[index].artifacts[variant]["path"])[0]
        for index, variant in [(0, "raw"), (0, "ema"), (1, "raw"), (1, "ema")]
    ]
    for agent in (raw0, ema0, raw1, ema1):
        assert agent.hypers.semantic_pack == "ur-lessons-vs-gw-allies"
        assert {"semantic_cards", "known_hand"} <= agent.world_binding[
            "input_schema"
        ].keys()
        assert all(setup["sideboard"] for setup in agent.world_binding["setups"])
    assert any(
        not torch.equal(a, b) for a, b in zip(raw0.parameters(), ema0.parameters())
    )
    if empty_second:
        for previous, current in zip(raw0.parameters(), raw1.parameters()):
            torch.testing.assert_close(previous, current, rtol=0, atol=0)
    for averaged, previous, raw in zip(
        ema1.parameters(), ema0.parameters(), raw1.parameters()
    ):
        torch.testing.assert_close(averaged, 0.5 * previous + 0.5 * raw)
    for averaged, raw in zip(ema1.buffers(), raw1.buffers(), strict=True):
        torch.testing.assert_close(averaged, raw, rtol=0, atol=0)
    for index, stage in enumerate(run.stages, start=1):
        checkpoint = torch.load(stage.artifacts["ema"]["path"], weights_only=False)
        assert checkpoint["bc"]["averaging"] == {
            "clock": "collect-update-iteration",
            "iteration": index,
            "rate": 0.5,
        }
