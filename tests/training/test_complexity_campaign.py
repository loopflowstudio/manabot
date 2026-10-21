"""Scientific declaration preserves all cells, seeds and frozen controls."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import pytest
import torch

from experiments.runners.complexity_campaign import (
    ROOT,
    Anchor,
    ComplexityPlan,
    GPUComplexityPlan,
    SplitComplexityPlan,
    evaluate_fit,
    evaluate_retained,
    execute,
    parameter_digest,
    read_complexity_plan,
)
from experiments.runners.environment_complexity import Pool
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import execute_regime
from manabot.training.models import AtaraxosMoveLearning
from manabot.training.monitor_evaluation import MonitorProtocol, MonitorResult
from manabot.verify.store import VerifyStore
from tests.training.test_recovery import recipe as tiny_recipe


def plan() -> ComplexityPlan:
    return ComplexityPlan(
        updates=1000,
        training_seconds_per_run=7200,
        total_seconds=47 * 3600,
        evaluation_reserve_seconds=90000,
        calibration_hashes={str(n): "a" * 64 for n in range(9)},
    )


def test_declaration_and_complete_game_counts() -> None:
    frozen = plan()
    recipes = frozen.recipes()
    assert tuple(recipes) == ("l1", "l2", "l3")
    assert all(r.stages[0].updates == 1000 for r in recipes.values())
    families = [set(frozen.deals(n)) for n in (0, 500, 1000)]
    assert len(set.union(*families)) == 150
    # Per fit: one initial, two candidate and two control cohorts; both anchors,
    # fifty whole deals, eight seat/role legs. Nine independent fits overall.
    assert 9 * 5 * 2 * 50 * 8 == 36000
    assert frozen.experiment().resolve().cases[0].receipt()


def test_allocation_or_cohort_cannot_silently_shrink() -> None:
    for change in (
        {"updates": 100},
        {"updates": 1050},
        {"seed_order": (1, 2, 3)},
        {"total_seconds": 3600},
        {"calibration_hashes": {}},
    ):
        with pytest.raises(ValueError):
            ComplexityPlan.model_validate({**plan().model_dump(), **change})


def gpu_plan() -> GPUComplexityPlan:
    return GPUComplexityPlan.model_validate(
        {
            **plan().model_dump(),
            "schema_version": 2,
            "total_seconds": 43 * 3600,
            "evaluation_reserve_seconds": 85000,
            "max_usd_per_hour": 0.69,
            "calibration_usd": 2,
        }
    )


def test_gpu_plan_preserves_cohort_but_changes_bound_recipe() -> None:
    frozen = gpu_plan()
    for recipe in frozen.recipes().values():
        assert recipe.stages[0].streams == 64
        assert recipe.stages[0].execution.device == "cuda"
        assert recipe.stages[0].transitions == 64
    assert frozen.deals(1000) == plan().deals(1000)
    assert frozen.digest != plan().digest
    for recipe in plan().recipes().values():
        assert recipe.stages[0].streams == 4
        assert recipe.stages[0].execution.device == "cpu"


def test_gpu_plan_rejects_overspending_and_unapproved_settings() -> None:
    for change in (
        {"max_usd_per_hour": 0.80},
        {"calibration_usd": 2.01},
        {"inclusive_cap_usd": 36},
        {"settlement_reserve_usd": 0},
        {"streams": 512},
        {"device": "cpu"},
    ):
        with pytest.raises(ValueError):
            GPUComplexityPlan.model_validate({**gpu_plan().model_dump(), **change})


def test_gpu_campaign_rejects_cpu_before_starting_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ETUDE_RUN", "fixture-only")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        execute(gpu_plan())


def test_gpu_campaign_rejects_another_world_before_allocation_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ETUDE_RUN", "fixture-only")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("managym.WORLD_VERSION", "w7")
    with pytest.raises(ValueError, match="requires w6"):
        execute(gpu_plan())


def split_plan() -> SplitComplexityPlan:
    return SplitComplexityPlan(
        total_seconds=9 * 1740 + 1800, max_usd_per_hour=0.69, prior_usd=4.55
    )


def test_split_plan_trains_tested_recipe_and_keeps_the_scored_cohort() -> None:
    frozen = split_plan()
    for pool, recipe in frozen.recipes().items():
        stage = recipe.stages[0]
        assert (stage.updates, stage.streams, stage.transitions) == (128, 64, 64)
        assert stage.execution.device == "cuda"
        assert recipe.agent.hidden_dim == 64 and recipe.agent.attention_layers == 2
        learning = stage.learning
        assert isinstance(learning, AtaraxosMoveLearning)
        # MTG-135 protocol017's tested end-of-horizon rates.
        assert learning.rates(128) == (1e-4, 0.005076577477226473)
        assert recipe.match.content_pack == "mtg122-complexity-v1", pool
    assert frozen.evaluation_updates == (0, 64, 128)
    assert frozen.checkpoint_updates == (32, 64)
    # Same deal families as the earlier versions, by milestone ordinal.
    assert [frozen.deals(n) for n in (0, 64, 128)] == [
        plan().deals(n) for n in (0, 500, 1000)
    ]
    assert read_complexity_plan(frozen.model_dump_json()) == frozen


def test_split_plan_rejects_overspending_or_another_recipe() -> None:
    for change in (
        {"max_usd_per_hour": 0.70},
        {"prior_usd": 29},
        {"total_seconds": 9 * 1740},
        {"updates": 1000},
        {"training_seconds_per_run": 600},
        {"evaluation_placement": "gpu"},
        {"seed_order": (1, 2, 3)},
    ):
        with pytest.raises(ValueError):
            SplitComplexityPlan.model_validate({**split_plan().model_dump(), **change})


@pytest.mark.parametrize("name", ["plan", "split-plan"])
def test_frozen_plan_files_keep_their_meaning(name: str) -> None:
    path = ROOT / f"experiments/environment-complexity-{name}.json"
    retained = json.loads(path.read_text())
    assert read_complexity_plan(path.read_text()).model_dump(mode="json") == retained


@dataclass(frozen=True)
class _TinyPlan:
    """Three-update fit scored on one deal, standing in for the frozen cohort."""

    updates: int = 3

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return (0, 1, 3)

    def protocol(self, pool: Pool, anchor: Anchor, update: int) -> MonitorProtocol:
        return MonitorProtocol(deal_seeds=(1_225_900_000 + update,), opponent=anchor)


def test_fetched_fit_is_scored_once_and_never_overwrites_failures(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        run = execute_regime(
            tiny_recipe(), 13590, tmp_path / "gpu", store, checkpoint_updates=(1,)
        )
    assert run.status == "completed"
    fetched = tmp_path / "fetched/training"
    shutil.move(str(tmp_path / "gpu"), fetched)
    initial, _ = load_checkpoint_agent(
        str(fetched / Path(run.stages[0].artifacts["initial_raw"]["path"]).name)
    )
    digest = parameter_digest(initial)
    cells = tmp_path / "fetched"

    def score() -> int:
        return evaluate_fit(
            _TinyPlan(),  # type: ignore[arg-type]
            run,
            cells,
            "l3",
            digest,
            None,
            lambda: False,
            training=fetched,
        )

    # Initial candidate, then candidate/control at midpoint and endpoint.
    assert score() == 2 * 5
    assert score() == 0
    frozen = {path: path.read_bytes() for path in cells.glob("*/monitor.json")}
    # The opt-in anchor adds its own five cells and leaves frozen ones untouched.
    assert (
        evaluate_fit(
            _TinyPlan(),  # type: ignore[arg-type]
            run,
            cells,
            "l3",
            digest,
            None,
            lambda: False,
            training=fetched,
            anchors=("heuristic",),
        )
        == 5
    )
    assert {path: path.read_bytes() for path in frozen} == frozen
    added = MonitorResult.model_validate_json(
        (cells / "3-heuristic-control/monitor.json").read_text()
    )
    assert added.status == "completed" and added.protocol.opponent == "heuristic"
    assert score() == 0
    (cells / "3-random-control/monitor.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete attempt"):
        score()
    with pytest.raises(ValueError, match="paired preflight"):
        evaluate_fit(
            _TinyPlan(),  # type: ignore[arg-type]
            run,
            cells,
            "l3",
            "0" * 64,
            None,
            lambda: False,
            training=fetched,
        )


def test_mini_evaluation_rejects_local_hosts_and_inline_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("socket.gethostname", lambda: "laptop")
    with pytest.raises(ValueError, match="runs on the Mini"):
        evaluate_retained(tmp_path)
    monkeypatch.setattr("socket.gethostname", lambda: "mini-heart.local")
    (tmp_path / "plan.json").write_text(gpu_plan().model_dump_json())
    with pytest.raises(ValueError, match="only split plans"):
        evaluate_retained(tmp_path)


def test_split_campaign_rejects_cpu_before_starting_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ETUDE_RUN", "fixture-only")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        execute(split_plan())
