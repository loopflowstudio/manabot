"""Screen admission, scheduling and stop policy without scientific training."""

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.runners import run_training_regimes as runner
from experiments.runners.run_value_models import smoke_plan
from experiments.runners.run_value_screen import screen_plan
from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from experiments.runners.value_screen import screen_diagnostic
from manabot.arena.models import PlayerRegistration, canonical_sha256, file_sha256
from manabot.training.models import StageRecord, TrainingRegime, TrainingRun
from manabot.verify.store import VerifyStore


@pytest.fixture(scope="module")
def plan() -> ResolvedStudy:
    return screen_plan()


def test_screen_is_separate_and_fixed(plan: ResolvedStudy) -> None:
    assert ResolvedStudy.model_validate_json(plan.model_dump_json()) == plan
    assert len(plan.recipes) == len(plan.protocol.training_seeds) == 3
    assert 4 * len(plan.protocol.anchor_deals) == 100
    assert len(smoke_plan().recipes) == 8
    for fields in (
        {"purpose": "scientific"},
        {"study": "value-models"},
        {"anchors": ("random",)},
        {"paired_deals": (990001,)},
        {"training_seeds": (1,)},
        {"anchor_deals": (990001,)},
    ):
        with pytest.raises(ValueError):
            EvaluationProtocol.model_validate({**plan.protocol.model_dump(), **fields})
    # Existing scientific studies still require three anchors.
    with pytest.raises(ValueError, match="three fixed baselines"):
        EvaluationProtocol(
            study="learning-speed",
            purpose="scientific",
            regime_digests=("a" * 64, "b" * 64),
            training_seeds=(1, 2, 3),
            uncertainty="paired-seed-descriptive",
            cost_cutoffs_seconds=(100,),
            endpoint_paired_deals=(950001,),
            endpoint_anchor_deals=(960001,),
            anchors=("scripted-greedy",),
        )


def test_screen_rejects_unpaired_settings_and_missing_identity(
    plan: ResolvedStudy,
) -> None:
    changed = plan.model_dump()
    changed["recipes"][1]["stages"][0]["learning"]["learning_rate_scale"] = 0.123
    changed["protocol"]["regime_digests"] = tuple(
        canonical_sha256(r) for r in changed["recipes"]
    )
    with pytest.raises(ValueError, match="only value aggregation"):
        ResolvedStudy.model_validate(changed)
    with pytest.raises(ValueError, match="all runtime"):
        ResolvedStudy.model_validate({**plan.model_dump(), "runtime_identities": {}})


def _run(
    recipe: TrainingRegime, seed: int, out: Path, *, seconds: float = 2000
) -> TrainingRun:
    out.mkdir(parents=True)
    for stage in recipe.stages:
        (out / f"{stage.id}.pt").write_bytes(b"fixture")
    run = TrainingRun(
        id=f"{recipe.id}-{seed}",
        regime=recipe,
        regime_digest=canonical_sha256(recipe.model_dump(mode="json")),
        seed=seed,
        seed_streams={},
        identities={},
        status="completed",
        seconds=seconds,
        stages=[
            StageRecord(
                id=s.id,
                status="completed",
                diagnostics=[{}] * 400,
                cumulative_seconds=(i + 1) * seconds / 2,
                learner_transitions=102400,
                artifacts={
                    "raw": {
                        "path": str(out / f"{s.id}.pt"),
                        "sha256": file_sha256(out / f"{s.id}.pt"),
                        "bytes": 7,
                    }
                },
            )
            for i, s in enumerate(recipe.stages)
        ],
    )
    (out / "run.json").write_text(run.model_dump_json())
    return run


@pytest.mark.parametrize("seconds,stop", [(1000, False), (3000, True)])
def test_two_hour_diagnostic_retains_decision(
    plan: ResolvedStudy, tmp_path: Path, seconds: float, stop: bool
) -> None:
    recipe = TrainingRegime.model_validate(plan.recipes[0])
    for seed in range(3):
        _run(recipe, seed, tmp_path / str(seed), seconds=seconds)
    if stop:
        with pytest.raises(RuntimeError, match="projection exceeds"):
            screen_diagnostic(tmp_path, 7200, training=True)
    else:
        screen_diagnostic(tmp_path, 7200, training=True)
    receipt = json.loads((tmp_path / "diagnostic-2h.json").read_text())
    assert receipt["completed_updates"] == 2400
    assert receipt["decision"] == ("stop" if stop else "continue")
    assert not receipt["strength_inspected"]


@pytest.mark.parametrize("invalid_cell", [False, True])
def test_screen_executor_schedules_only_scripted_games(
    plan: ResolvedStudy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_cell: bool,
) -> None:
    training_calls: list[tuple[str, int]] = []
    game_counts: list[int] = []

    def train(
        recipe: TrainingRegime, seed: int, out: Path, store: VerifyStore
    ) -> TrainingRun:
        training_calls.append((recipe.id, seed))
        run = _run(recipe, seed, out)
        run.identities = plan.runtime_identities
        (out / "run.json").write_text(run.model_dump_json())
        return run

    def register(
        run: TrainingRun, stage: StageRecord, variant: str
    ) -> PlayerRegistration:
        return PlayerRegistration(
            player_id=f"{run.id}-{stage.id}",
            display_name=run.regime.id,
            role="challenger",
            runner_kind="code",
            player_spec={"kind": "random"},
            source_sha256="a" * 64,
            compute_class_id="fixture",
            information_boundary="acting-viewer",
            world=run.regime.world,
            content_suite=runner.SELECTED_SUITE,
            observation_abi_sha256=run.identities["observation_abi_sha256"],
            action_abi_sha256=run.identities["action_abi_sha256"],
            matchup_sha256=run.identities["matchup_sha256"],
            player_seed_derivation_id="arena-pair-deal-player-v1",
        )

    # Any is confined to this heterogeneous arena-call fixture boundary.
    def play(**kwargs: Any) -> tuple[list[dict[str, object]], None, dict[str, bool]]:
        assert kwargs["player_b"].player_spec == {"kind": "scripted_greedy"}
        assert kwargs["deal_seeds"] == plan.protocol.anchor_deals
        assert (
            kwargs["comparison_seed_aliases"][kwargs["player_b"].player_id]
            == "reference"
        )
        rows = [
            dict(
                deal_seed=d,
                leg=leg,
                player_a_seat=leg % 2,
                score_a=0.5,
                failure=None,
                terminated=True,
                truncated=False,
                replay_passed=True,
            )
            for d in kwargs["deal_seeds"]
            for leg in range(4)
        ]
        game_counts.append(len(rows))
        return rows, None, {"passed": not invalid_cell}

    monkeypatch.setattr(runner, "execute_regime", train)
    monkeypatch.setattr(runner, "registration", register)
    monkeypatch.setattr(runner, "play_cell", play)
    out = tmp_path / "screen"
    if invalid_cell:
        with pytest.raises(RuntimeError, match="invalid arena cell"):
            runner.run_study("value-token-screen", out, plan, render_report=False)
        result = json.loads((out / "study.json").read_text())
        assert result["status"] == "failed"
        assert game_counts == [100]
        return
    runner.run_study("value-token-screen", out, plan, render_report=False)
    assert len(set(training_calls)) == 9
    assert len(game_counts) == 18 and sum(game_counts) == 1800
    result = json.loads((out / "study.json").read_text())
    assert result["status"] == "completed"
    assert len(result["measurements"]) == 18
    runner.report(out)
    paths = ("metrics.json", "report.md", "cost-comparison.json")
    before = {p: (out / p).read_bytes() for p in paths}
    runner.report(out)
    assert before == {p: (out / p).read_bytes() for p in paths}
    assert json.loads(before["cost-comparison.json"])["status"] == "available"
    assert (out / "learning-transitions.png").exists()
    with pytest.raises(ValueError, match="retries require"):
        runner.run_study("value-token-screen", out, plan, resume=True)


def test_changed_source_fails_before_training(
    plan: ResolvedStudy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = ResolvedStudy.model_validate(
        {
            **plan.model_dump(),
            "runtime_identities": {
                **plan.runtime_identities,
                "study_source_sha256": "0" * 64,
            },
        }
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("training must not start")

    monkeypatch.setattr(runner, "execute_regime", forbidden)
    with pytest.raises(ValueError, match="runtime/source differs"):
        runner.run_study("value-token-screen", tmp_path / "screen", bad)


def test_two_hour_timer_stops_and_retains_failed_attempt(
    plan: ResolvedStudy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [0.0]
    timers: list[float] = []
    # The signal API accepts both handlers and integer sentinel values.
    handlers: list[Any] = []
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        runner.signal, "signal", lambda signum, handler: handlers.append(handler)
    )
    monkeypatch.setattr(
        runner.signal, "setitimer", lambda which, delay: timers.append(delay)
    )

    def train(
        recipe: TrainingRegime, seed: int, out: Path, store: VerifyStore
    ) -> TrainingRun:
        _run(recipe, seed, out, seconds=7000)
        clock[0] = 7200
        handlers[-1]()
        pytest.fail("infeasible screen must stop at the diagnostic")

    monkeypatch.setattr(runner, "execute_regime", train)
    out = tmp_path / "screen"
    with pytest.raises(RuntimeError, match="two-hour diagnostic stop"):
        runner.run_study("value-token-screen", out, plan, render_report=False)
    result = json.loads((out / "study.json").read_text())
    assert result["status"] == "failed" and result["seconds"] == 7200
    assert len(result["runs"]) == 1 and result["comparisons"] == []
    assert timers == [7200, 0]
