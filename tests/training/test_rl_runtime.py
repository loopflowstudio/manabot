"""Bounded native self-play, persisted costs and ordinary raw/EMA reload."""

from pathlib import Path

import torch

from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import execute_regime
from manabot.training.models import TrainingRegime
from manabot.verify.store import VerifyStore


def bounded_regime():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/regimes/direct-self-play.json"
    )
    recipe = TrainingRegime.model_validate_json(path.read_text())
    recipe.agent.hidden_dim = 8
    recipe.agent.num_attention_heads = 2
    recipe.stages = recipe.stages[:2]
    for stage in recipe.stages:
        stage.updates = 1
        stage.streams = 4
        stage.transitions = 128
        stage.learning.epochs = 1
        stage.learning.minibatches = 2
        stage.learning.ema = 0.5
        stage.execution.wall_seconds = 120
    recipe.wall_seconds = 300
    return recipe


def test_self_play_records_costs_and_reloads_distinct_raw_ema(tmp_path, monkeypatch):
    from manabot.sim.net_opponent import NetOpponentTrainer

    initialize = NetOpponentTrainer.__init__

    def checked_initialize(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        assert self.env.match is self.collector.match
        assert self.env.match.hypers == bounded_regime().match

    monkeypatch.setattr(NetOpponentTrainer, "__init__", checked_initialize)
    with VerifyStore(tmp_path / "runs.sqlite") as store:
        run = execute_regime(bounded_regime(), 197, tmp_path / "run", store)
        assert store.training_run(run.id).status == "completed"
    assert run.status == "completed"
    assert sum(stage.games for stage in run.stages) > 0
    for stage in run.stages:
        assert stage.learner_transitions == 512
        assert stage.collection_seconds > 0
        assert stage.learning_seconds > 0
        assert stage.export_seconds > 0
        assert stage.optimizer_exposures > 0
        assert stage.artifacts["raw"]["sha256"] != stage.artifacts["ema"]["sha256"]
    raw0, ema0, raw1, ema1 = [
        load_checkpoint_agent(run.stages[index].artifacts[variant]["path"])[0]
        for index, variant in [(0, "raw"), (0, "ema"), (1, "raw"), (1, "ema")]
    ]
    assert any(
        not torch.equal(a, b) for a, b in zip(raw0.parameters(), ema0.parameters())
    )
    for averaged, previous, raw in zip(
        ema1.parameters(), ema0.parameters(), raw1.parameters()
    ):
        torch.testing.assert_close(averaged, 0.5 * previous + 0.5 * raw)
    for index, stage in enumerate(run.stages, start=1):
        checkpoint = torch.load(stage.artifacts["ema"]["path"], weights_only=False)
        assert checkpoint["bc"]["averaging"] == {
            "clock": "collect-update-iteration",
            "iteration": index,
            "rate": 0.5,
        }
