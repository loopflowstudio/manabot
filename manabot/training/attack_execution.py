"""Execute a frozen attack cohort and retain replayable arena evidence.

Training uses the ordinary regime executor. Every attacker checkpoint, including
its untrained baseline, faces the same target on all four seat/deck assignments.
Failures stop the cohort and remain in its manifest; they are never dropped from
an aggregate. These are bounded attacks, not exploitability certificates.
"""

from dataclasses import asdict
from pathlib import Path
import signal
import time
from typing import Any, Literal

import numpy as np
from pydantic import Field
import torch

from manabot.arena.match import SELECTED_SUITE, play_cell, selected_match
from manabot.arena.models import (
    ArenaKey,
    PlayerRegistration,
    canonical_sha256,
    file_sha256,
)
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.analysis import valid_game
from manabot.training.attacks import AttackPlan
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import Strict, TrainingRun
from manabot.verify.scenario_validation import validate_scenarios
from manabot.verify.store import VerifyStore


class AttackPoint(Strict):
    target: str
    seed: int
    updates: int
    training_seconds: float
    evaluation_seconds: float = 0
    score: float | None = None
    games: int = 0
    replay_passed: bool = False
    evidence_path: str


class AttackResult(Strict):
    plan_digest: str
    status: Literal["running", "completed", "failed"] = "running"
    runs: list[str] = Field(default_factory=list)
    run_paths: list[str] = Field(default_factory=list)
    points: list[AttackPoint] = Field(default_factory=list)
    seconds: float = 0
    error: str | None = None


def _player(path: Path, name: str, seed: int, run: TrainingRun) -> PlayerRegistration:
    agent, _ = load_checkpoint_agent(str(path))
    identity = run.identities
    return PlayerRegistration(
        player_id=name,
        display_name=name,
        role="challenger",
        runner_kind="checkpoint",
        player_spec={
            "kind": "checkpoint",
            "deterministic": False,
            "device": "cpu",
            "batch_size": 1,
        },
        compute_class_id="policy-cpu-one-thread-one-pass",
        information_boundary="acting-viewer",
        world=run.regime.world,
        content_suite=SELECTED_SUITE,
        observation_abi_sha256=identity["observation_abi_sha256"],
        action_abi_sha256=identity["action_abi_sha256"],
        matchup_sha256=identity["matchup_sha256"],
        checkpoint_sha256=file_sha256(path),
        checkpoint_bytes=path.stat().st_size,
        parameter_count=sum(p.numel() for p in agent.parameters()),
        training_seed=seed,
        artifact_id=f"attack/{file_sha256(path)}",
        player_seed_derivation_id="arena-pair-deal-player-v1",
    )


def _report(plan: AttackPlan, result: AttackResult, out: Path) -> None:
    lines = [
        "# Frozen-policy attack results",
        "",
        f"Status: {result.status}. Total elapsed: {result.seconds:.2f} seconds.",
        "",
        "A failed attack does not certify equilibrium or exact exploitability.",
        "",
        "| Target | Attacker seed | Updates | Training seconds | Score | Games | Replay |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for point in result.points:
        score = "unavailable" if point.score is None else f"{point.score:.4f}"
        lines.append(
            f"| {point.target} | {point.seed} | {point.updates} | {point.training_seconds:.2f} | {score} | {point.games} | {point.replay_passed} |"
        )
    if result.status == "completed":
        lines += [
            "",
            "## Endpoint improvement",
            "",
            "Exploratory 95% percentile bootstrap intervals resample attacker seeds, not individual games. Targets are reported separately; these intervals do not include uncertainty across target-policy training seeds.",
            "",
        ]
        for target in plan.targets:
            changes = []
            for seed in plan.attacker_seeds:
                points = [
                    p for p in result.points if p.target == target.id and p.seed == seed
                ]
                first, last = points[0], points[-1]
                assert first.score is not None and last.score is not None
                changes.append(last.score - first.score)
            rng = np.random.default_rng(937)
            values = np.asarray(changes)
            samples = rng.choice(values, size=(2000, len(values)), replace=True).mean(
                axis=1
            )
            low, high = np.quantile(samples, [0.025, 0.975])
            lines.append(
                f"- {target.id}: mean paired improvement {values.mean():.4f}, interval [{low:.4f}, {high:.4f}], {len(values)} independent attacker seeds."
            )
    if result.error:
        lines += ["", f"Failure: {result.error}"]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def execute_attack_plan(plan: AttackPlan, out: Path) -> AttackResult:
    """Fresh output only; alarm requires a POSIX main-thread caller.

    Legacy arena dictionaries are persisted intact at this boundary. Scores are
    admitted only after complete terminal games and policy-free Command replay.
    """
    plan = AttackPlan.model_validate(plan.model_dump())
    if plan.template.match != selected_match():
        raise ValueError("attack arena requires the selected matchup")
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] != 0:
        raise ValueError("attack runner requires ownership of its deadline timer")
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "plan.json", plan.model_dump(mode="json"))
    result = AttackResult(plan_digest=canonical_sha256(plan.model_dump(mode="json")))
    start = time.perf_counter()
    previous_threads = torch.get_num_threads()
    previous_handler = signal.getsignal(signal.SIGALRM)

    def timeout(_signum: int, _frame: Any) -> None:
        # Python signal frame is an interpreter boundary, not domain data.
        raise TimeoutError("attack cohort deadline exceeded")

    def save() -> None:
        result.seconds = time.perf_counter() - start
        atomic_json(out / "result.json", result.model_dump(mode="json"))
        _report(plan, result, out)

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, plan.total_seconds)
    torch.set_num_threads(1)
    evaluation_spent = 0.0
    try:
        save()
        atomic_json(
            out / "scenarios.json",
            [
                asdict(item)
                for item in validate_scenarios(selected_match=plan.template.match)
            ],
        )
        with VerifyStore(out / "training.sqlite") as store:
            for target in plan.targets:
                if file_sha256(target.policy.path) != target.policy.sha256:
                    raise ValueError("target bytes differ from frozen plan")
                for seed in plan.attacker_seeds:
                    run_dir = out / f"{target.id}-{seed}"
                    result.run_paths.append(str(run_dir))
                    save()
                    run = execute_regime(plan.regime_for(target), seed, run_dir, store)
                    result.runs.append(run.id)
                    target_player = _player(
                        Path(target.policy.path),
                        f"target-{target.id}",
                        target.producer_seeds[0],
                        run,
                    )
                    key = ArenaKey(
                        world=run.regime.world,
                        content_suite=SELECTED_SUITE,
                        viewer_boundary="acting-viewer",
                        arena_version="frozen-attack-v1",
                        rating_model_version="unrated",
                        rating_prior_sha256=canonical_sha256({}),
                        anchor_cohort_sha256=canonical_sha256(
                            [target_player.model_dump()]
                        ),
                        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
                    )
                    snapshots = [
                        (
                            0,
                            Path(run.stages[0].artifacts["initial_raw"]["path"]),
                            run.setup_seconds,
                        )
                    ]
                    snapshots.extend(
                        (
                            updates,
                            Path(stage.artifacts["raw"]["path"]),
                            float(stage.cumulative_seconds),
                        )
                        for updates, stage in zip(
                            plan.cumulative_updates, run.stages, strict=True
                        )
                    )
                    for updates, path, training_seconds in snapshots:
                        cell = out / f"arena-{target.id}-{seed}-{updates}"
                        cell.mkdir()
                        attacker = _player(
                            path, f"attacker-{seed}-{updates}", seed, run
                        )
                        atomic_json(
                            cell / "registrations.json",
                            [attacker.model_dump(), target_player.model_dump()],
                        )
                        point = AttackPoint(
                            target=target.id,
                            seed=seed,
                            updates=updates,
                            training_seconds=training_seconds,
                            evidence_path=str(cell),
                        )
                        result.points.append(point)
                        save()
                        available = min(
                            plan.total_seconds - (time.perf_counter() - start),
                            plan.evaluation_seconds - evaluation_spent,
                        )
                        if available <= 0:
                            raise TimeoutError("attack evaluation allocation exhausted")
                        signal.setitimer(signal.ITIMER_REAL, available)
                        tick = time.perf_counter()
                        try:
                            rows, trace, replay = play_cell(
                                key=key,
                                player_a=attacker,
                                player_b=target_player,
                                deal_seeds=plan.final_deal_seeds,
                                out_dir=cell,
                                checkpoint_paths={
                                    attacker.player_id: str(path),
                                    target_player.player_id: target.policy.path,
                                },
                                comparison_seed_aliases={
                                    attacker.player_id: "attacker",
                                    target_player.player_id: "target",
                                },
                                game_seconds=plan.game_seconds,
                                max_commands=plan.max_commands,
                            )
                            atomic_json(
                                cell / "evidence.json",
                                {"rows": rows, "trace": trace, "replay": replay},
                            )
                            point.games = len(rows)
                            point.replay_passed = bool(replay["passed"])
                            if (
                                not point.replay_passed
                                or len(rows) != 4 * len(plan.final_deal_seeds)
                                or not all(valid_game(row) for row in rows)
                            ):
                                raise ValueError(
                                    "incomplete, invalid or unreplayable attack cell"
                                )
                            point.score = sum(
                                float(row["score_a"]) for row in rows
                            ) / len(rows)
                        finally:
                            point.evaluation_seconds = time.perf_counter() - tick
                            evaluation_spent += point.evaluation_seconds
                            save()
                            remaining = plan.total_seconds - (
                                time.perf_counter() - start
                            )
                            if remaining > 0:
                                signal.setitimer(signal.ITIMER_REAL, remaining)
                    save()
        result.status = "completed"
        return result
    except BaseException as error:
        result.status = "failed"
        result.error = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        torch.set_num_threads(previous_threads)
        save()
