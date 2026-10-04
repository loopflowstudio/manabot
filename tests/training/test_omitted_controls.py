"""Selection semantics, independent recipes and real averaged-policy collection."""

from collections.abc import Callable
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import torch

from experiments.runners.omitted_controls import (
    ContrastName,
    resolve_contrast,
    smoke_plan,
)
from manabot.model.agent import Agent
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.net_opponent import RolloutBatch, SeatRoutedCollector
from manabot.training.execution import execute_regime
from manabot.training.models import AtaraxosMoveLearning, TrainingRegime, TrainSelfPlay
from manabot.training.selection import (
    selection_diagnostics,
    selection_mask,
    terminal_distances,
)
from manabot.verify.store import VerifyStore


def test_quantile_ties_and_censored_distances() -> None:
    advantages = torch.tensor([[0.2, 0.2], [0.2, 0.2]])
    assert selection_mask(advantages, "quantile", 0.25, 0.01).sum() == 4
    assert selection_mask(advantages, "top_count", 0.25, 0.01).sum() == 1
    assert selection_mask(advantages, "quantile", 0.25, 1).sum() == 0
    ends = torch.tensor([[False, True], [True, False], [False, False], [True, False]])
    distances = terminal_distances(ends)
    assert distances.tolist() == [[1, 0], [0, -1], [1, -1], [0, -1]]
    groups = selection_diagnostics(
        torch.ones(4, 2),
        torch.ones(4, 2) * 2,
        ends,
        ends,
        torch.zeros(4, 2, dtype=torch.long),
    )
    assert sum(g["rows"] for g in groups) == 8
    assert sum(g["retained"] for g in groups) == 3
    assert (
        next(g for g in groups if g["terminal_distance"] == "censored")[
            "retained_residual_abs_mean"
        ]
        is None
    )


@pytest.mark.parametrize("name", get_args(ContrastName))
def test_independent_resolved_contrasts(name: ContrastName) -> None:
    contrast = resolve_contrast(name)
    plan = smoke_plan(name)
    assert plan.protocol.study == "omitted-controls"
    assert plan.protocol.purpose == "workflow-smoke"
    assert contrast.baseline.match == contrast.treatment.match
    assert contrast.baseline.observation == contrast.treatment.observation
    assert plan.protocol.training_seeds == (693,)
    if name == "evaluation-ema":
        assert len(plan.recipes) == 1
        assert plan.protocol.evaluation_variants == ("raw", "ema")
    else:
        assert len(plan.recipes) == 2
    if name == "behavior-ema":
        for stage in contrast.treatment.stages:
            assert isinstance(stage, TrainSelfPlay)
            assert stage.behavior == "ema-self"
            assert stage.learning.ema == 0.999
    if name == "policy-trace":
        assert contrast.treatment.stages[0].learning.gamma == 1
        assert contrast.treatment.stages[0].learning.value_lambda == 0.95


@pytest.mark.parametrize("categorical", [False, True])
def test_real_ema_behavior_and_actor_only_empty_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    categorical: bool,
) -> None:
    recipe = resolve_contrast("behavior-ema").treatment
    recipe.agent.hidden_dim = 8
    if categorical:
        recipe.agent.value_kind = "categorical_wdl"
    for stage in recipe.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.updates = 1
        stage.transitions = 64
        if categorical:
            stage.learning = AtaraxosMoveLearning(gradient="ataraxos_move", ema=0.5)
        stage.learning.ema = 0.5
        stage.learning.filter_scope = "actor"
        stage.learning.min_advantage = 100
    original = SeatRoutedCollector.collect
    collected: list[Agent] = []

    def checked_collect(
        self: SeatRoutedCollector,
        agent: Agent,
        num_steps: int,
        *,
        deadline_monotonic: float | None = None,
        check: Callable[[], None] | None = None,
    ) -> RolloutBatch:
        batch = original(
            self, agent, num_steps, deadline_monotonic=deadline_monotonic, check=check
        )
        tensors = {k: torch.as_tensor(v).flatten(0, 1) for k, v in batch.obs.items()}
        with torch.no_grad():
            logits, values = agent(tensors)
        np.testing.assert_allclose(
            logits.softmax(-1).numpy(),
            batch.probabilities.reshape(logits.shape),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            values.numpy().reshape(batch.values.shape), batch.values, atol=1e-6
        )
        if collected:
            raw, _ = load_checkpoint_agent(str(tmp_path / "run/policy-0-raw.pt"))
            averaged, _ = load_checkpoint_agent(str(tmp_path / "run/policy-0-ema.pt"))
            with torch.no_grad():
                raw_values = raw.get_value(tensors)
                averaged_values = averaged.get_value(tensors)
            torch.testing.assert_close(values, averaged_values, rtol=0, atol=1e-6)
            assert not torch.allclose(values, raw_values, rtol=0, atol=1e-6)
        collected.append(agent)
        return batch

    monkeypatch.setattr(SeatRoutedCollector, "collect", checked_collect)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        run = execute_regime(recipe, 693, tmp_path / "run", store)
    assert run.status == "completed"
    assert sum(s.games for s in run.stages) > 0
    assert collected[0] is collected[1]
    for stage in run.stages:
        diagnostic = stage.diagnostics[0]
        assert diagnostic["behavior"] == "ema-self"
        assert diagnostic["actor_exposures"] == 0
        assert diagnostic["critic_exposures"] == 256
        assert len(diagnostic["selection_groups"]) > 0
        if categorical:
            assert diagnostic["schedule_clock"] == "collection-update-iteration"
        for variant in ("raw", "ema"):
            loaded, _ = load_checkpoint_agent(stage.artifacts[variant]["path"])
            assert loaded.hypers.value_kind == recipe.agent.value_kind
    assert (
        run.stages[-1].artifacts["raw"]["sha256"]
        != run.stages[-1].artifacts["ema"]["sha256"]
    )
    bad = recipe.model_dump()
    bad["stages"][0]["learning"]["ema"] = None
    with pytest.raises(ValueError, match="EMA"):
        TrainingRegime.model_validate(bad)


def test_variants_never_add_training_replicates() -> None:
    from manabot.training.analysis import paired_uncertainty

    cells = [
        dict(
            a="candidate",
            b="random",
            cutoff=0,
            phase="development",
            variant=variant,
            training_seed=seed,
            b_training_seed=None,
            scheduled_games=4,
            replay={"passed": True},
            rows=[
                dict(
                    deal_seed=964001,
                    score_a=float(variant == "ema"),
                    failure=None,
                    terminated=True,
                    truncated=False,
                    replay_passed=True,
                )
                for _ in range(4)
            ],
        )
        for variant in ("raw", "ema")
        for seed in (1, 2, 3)
    ]
    results = paired_uncertainty(cells)
    assert len(results) == 2
    assert {r["variant"]: r["score_a"] for r in results} == {"raw": 0, "ema": 1}
    assert all(r["training_seeds"] == 3 for r in results)
    assert all(
        r["status"] == "unavailable"
        for r in paired_uncertainty([c for c in cells if c["training_seed"] != 3])
    )


@pytest.mark.parametrize(
    "settings", [{"filter_kind": "quantile"}, {"filter_scope": "actor"}]
)
def test_compound_rejects_unsupported_filter_controls(settings: dict[str, str]) -> None:
    from manabot.training.models import TrainCompound

    with pytest.raises(ValueError, match="top-count actor-critic"):
        TrainCompound.model_validate(
            dict(operation="train_compound", id="compound", learning=settings)
        )
