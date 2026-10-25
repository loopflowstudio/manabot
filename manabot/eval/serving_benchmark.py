"""Bounded, local, predeclared shared-serving calibration; never rents hardware.

The plan freezes four ablations, matched deal panels and numerical/adoption
thresholds. Every repetition is retained, including failures. Native retained
root calibration runs in a killable process. CPU success is not CUDA evidence.

uv run --extra evaluation python -m manabot.eval.serving_benchmark --plan PLAN --out DIR
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
import psutil
from pydantic import Field, model_validator
import torch

from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.execution import bind_execution
from manabot.infra.config import MatchConfig
from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel, file_sha256
from manabot.sim.execution import EvalExecution
from manabot.sim.performance import EvalPerformance
from manabot.sim.replay import read_trace, replay_prefix
from manabot.sim.serving import (
    DecisionRequest,
    PolicyAdmission,
    PolicyKey,
    PolicyStore,
    _infer,
    sample_rows,
)

Variant = Literal["cold_scalar", "resident_only", "resident_catalog", "batched"]
VARIANTS: tuple[Variant, ...] = (
    "cold_scalar",
    "resident_only",
    "resident_catalog",
    "batched",
)


@dataclass(frozen=True)
class _ProfileRoot:
    game: dict[str, Any]
    revision: int


def _select_profile_roots(
    games: list[dict[str, Any]], *, measured: int
) -> list[_ProfileRoot]:
    ordered_games = sorted(
        games,
        key=lambda game: (
            int(game["deal_seed"]),
            int(game["leg"]),
            str(game["match_id"]),
        ),
    )
    # Only the declared prefix is probed; do not materialize unused trace roots.
    candidates = (
        _ProfileRoot(game, int(decision["revision"]))
        for game in ordered_games
        for decision in game["decisions"]
    )
    roots = list(islice(candidates, measured))
    if len(roots) < measured:
        raise ValueError(
            f"matched-root corpus has {len(roots)} roots; {measured} are required"
        )
    return roots


class Panel(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    request: EvaluationRequest


class BenchmarkPlan(StrictModel):
    panels: tuple[Panel, ...] = Field(min_length=1)
    configurations: dict[Variant, EvalExecution]
    repetitions: Literal[3]
    measured_roots: int = Field(gt=0)
    absolute_tolerance: float = Field(ge=0, allow_inf_nan=False)
    relative_tolerance: float = Field(ge=0, allow_inf_nan=False)
    minimum_action_agreement: float = Field(ge=0, le=1)
    minimum_speedup: float = Field(gt=0, allow_inf_nan=False)
    maximum_instrumentation_overhead: float = Field(ge=0, allow_inf_nan=False)
    process_tree_bytes: int = Field(gt=0)
    deadline_unix: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def complete(self) -> BenchmarkPlan:
        if set(self.configurations) != set(VARIANTS):
            raise ValueError("all four explicit ablations are required")
        cold, resident, catalog, batch = (self.configurations[k] for k in VARIANTS)
        if (
            cold.device != "cpu"
            or cold.max_batch_rows != 1
            or cold.max_active_games != 1
            or cold.resident
            or cold.cache_catalog
        ):
            raise ValueError(
                "cold_scalar must declare scalar CPU without residency or catalog cache"
            )
        if (
            resident.max_batch_rows != 1
            or resident.cache_catalog
            or not resident.resident
        ):
            raise ValueError("resident_only must isolate residency")
        if (
            catalog.max_batch_rows != 1
            or not catalog.cache_catalog
            or not catalog.resident
        ):
            raise ValueError("resident_catalog must isolate catalog reuse")
        if (
            batch.max_batch_rows <= 1
            or batch.max_active_games <= 1
            or not batch.cache_catalog
            or not batch.resident
        ):
            raise ValueError("batched must admit multiple resident rows and games")
        if len({p.name for p in self.panels}) != len(self.panels):
            raise ValueError("panel names must be unique")
        for execution in self.configurations.values():
            if (
                execution.deadline_unix > self.deadline_unix
                or execution.diagnostic_batches
            ):
                raise ValueError(
                    "throughput ablations must fit plan deadline and exclude diagnostic overhead"
                )
        return self


def _numerical_worker(
    connection: Connection, plan: BenchmarkPlan, panel: Panel, trace: Path
) -> None:
    """Compare retained identical inputs, masks, seeds and outputs across modes."""
    torch.set_num_threads(1)
    stores: list[PolicyStore] = []
    try:
        request = panel.request
        games = read_trace(trace)
        roots = _select_profile_roots(games, measured=plan.measured_roots)
        results = []
        for player in (request.candidate, request.opponent):
            if player.runner_kind != "checkpoint":
                continue
            chosen = [
                r
                for r in roots
                if r.game["decisions"][r.revision]["player_id"] == player.player_id
            ]
            if not chosen:
                raise ValueError(
                    "retained root population has no decisions for checkpoint"
                )
            observations, seeds = [], []
            for root in chosen:
                env, observation = replay_prefix(root.game, root.revision)
                observations.append(observation)
                seeds.append(root.game["decisions"][root.revision]["policy_rng_seed"])
                env.close()
            tensor = {
                k: torch.from_numpy(np.stack([o[k] for o in observations])).float()
                for k in observations[0]
            }
            reference: torch.Tensor | None = None
            reference_actions: list[int] = []
            for name in VARIANTS:
                execution = plan.configurations[name]
                _, (bound,) = bind_execution(request.key, (player,), execution)
                perf = EvalPerformance(name)
                store = PolicyStore(execution, perf)
                stores.append(store)
                store.admit(
                    PolicyAdmission(
                        bound,
                        request.checkpoint_paths[player.player_id],
                        tuple(
                            MatchConfig.model_validate(g["match_hypers"]) for g in games
                        ),
                    )
                )
                resident = store.get(PolicyKey.from_registration(bound, execution))
                output = []
                with torch.no_grad():
                    for start in range(0, len(chosen), execution.max_batch_rows):
                        chunk = {
                            k: v[start : start + execution.max_batch_rows]
                            for k, v in tensor.items()
                        }
                        prepared = (
                            resident.scope.prepare(chunk)
                            if execution.cache_catalog
                            else {k: v.to(execution.device) for k, v in chunk.items()}
                        )
                        output.append(resident.agent(prepared)[0].cpu())
                logits = torch.cat(output)
                actions = sample_rows(
                    logits,
                    tensor["actions_valid"],
                    seeds,
                    bool(player.player_spec["deterministic"]),
                )
                if reference is None:
                    reference, reference_actions = logits, actions
                mask = tensor["actions_valid"] != 0
                close = torch.isclose(
                    logits[mask],
                    reference[mask],
                    atol=plan.absolute_tolerance,
                    rtol=plan.relative_tolerance,
                )
                agreement = sum(
                    a == b for a, b in zip(actions, reference_actions, strict=True)
                ) / len(actions)
                # Same retained schedule and row seeds in both modes. Interleave
                # three repetitions; this measures span overhead on warm serving,
                # separately from throughput and from diagnostic profiler waits.
                overhead_samples: dict[bool, list[float]] = {True: [], False: []}
                overhead_actions_match = True
                requests = [
                    DecisionRequest(
                        str(i),
                        "calibration",
                        0,
                        i,
                        PolicyKey.from_registration(bound, execution),
                        i,
                        seed,
                        obs,
                        plan.deadline_unix,
                        time.monotonic(),
                    )
                    for i, (obs, seed) in enumerate(
                        zip(observations, seeds, strict=True)
                    )
                ]
                for repetition in range(3):
                    for enabled in (
                        [False, True] if repetition % 2 == 0 else [True, False]
                    ):
                        perf.enabled = enabled
                        start = time.monotonic()
                        measured_actions = []
                        for offset in range(0, len(requests), execution.max_batch_rows):
                            measured_actions.extend(
                                reply.action
                                for reply in _infer(
                                    store,
                                    requests[
                                        offset : offset + execution.max_batch_rows
                                    ],
                                    "overhead",
                                )
                            )
                        overhead_samples[enabled].append(time.monotonic() - start)
                        overhead_actions_match &= measured_actions == actions
                overhead = float(
                    np.median(overhead_samples[True])
                    / np.median(overhead_samples[False])
                    - 1
                )
                results.append(
                    {
                        "player": player.player_id,
                        "variant": name,
                        "roots": len(chosen),
                        "span_instrumentation_overhead": overhead,
                        "overhead_samples_seconds": {
                            str(k): v for k, v in overhead_samples.items()
                        },
                        "instrumentation_actions_match": overhead_actions_match,
                        "legal_logits_close": bool(close.all()),
                        "maximum_absolute_error": float(
                            (logits[mask] - reference[mask]).abs().max()
                        ),
                        "action_agreement": agreement,
                        "passed": bool(close.all())
                        and agreement >= plan.minimum_action_agreement
                        and overhead_actions_match,
                    }
                )
                store.close()
        connection.send(
            {"results": results, "passed": all(r["passed"] for r in results)}
        )
    except Exception as exc:
        connection.send({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        for store in stores:
            store.close()
        connection.close()


def _bounded_probe(
    target: Callable[..., None], args: tuple[object, ...], plan: BenchmarkPlan
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe()
    process = context.Process(target=target, args=(sender, *args))
    process.start()
    sender.close()
    peak_rss = 0
    descendants: list[psutil.Process] = []
    try:
        while time.time() < plan.deadline_unix:
            if receiver.poll(0.1):
                try:
                    result = receiver.recv()
                except EOFError:
                    break
                result["peak_process_tree_rss_bytes"] = peak_rss
                return result
            if not process.is_alive():
                break
            try:
                parent = psutil.Process(process.pid)
                descendants = parent.children(recursive=True)
                rss = sum(
                    p.memory_info().rss
                    for p in [parent, *descendants]
                    if p.is_running()
                )
            except psutil.NoSuchProcess:
                continue
            peak_rss = max(peak_rss, rss)
            if rss > plan.process_tree_bytes:
                return {
                    "passed": False,
                    "error": "process tree memory limit",
                    "peak_process_tree_rss_bytes": peak_rss,
                }
        return {
            "passed": False,
            "error": "worker exited or deadline exhausted",
            "peak_process_tree_rss_bytes": peak_rss,
        }
    finally:
        if process.is_alive():
            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                pass
            for child in reversed(descendants):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill()
        process.join()
        receiver.close()


def numerical_check(plan: BenchmarkPlan, panel: Panel, trace: Path) -> dict[str, Any]:
    return _bounded_probe(_numerical_worker, (plan, panel, trace), plan)


def _panel_worker(
    connection: Connection,
    panel: Panel,
    execution: EvalExecution,
    name: Variant,
    directory: Path,
) -> None:
    started = time.monotonic()
    try:
        result = evaluate(
            panel.request,
            out_dir=directory,
            execution=None if name == "cold_scalar" else execution,
        )
        connection.send(
            {
                "passed": result.status == "completed",
                "games": len(result.rows),
                "trace": str(directory / result.rows[0].trace_path)
                if result.rows
                else None,
                "error": result.error,
                "seconds": time.monotonic() - started,
            }
        )
    except Exception as exc:
        connection.send(
            {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": time.monotonic() - started,
            }
        )
    finally:
        connection.close()


def run(plan_path: Path, out: Path) -> dict[str, Any]:
    plan = BenchmarkPlan.model_validate_json(plan_path.read_bytes())
    if plan.deadline_unix <= time.time() or any(
        e.remaining() <= 0 for e in plan.configurations.values()
    ):
        raise ValueError("benchmark deadline expired")
    # Byte admission happens before any game or output directory is created.
    for panel in plan.panels:
        request = panel.request
        for player in (request.candidate, request.opponent):
            if player.runner_kind == "checkpoint":
                path = Path(request.checkpoint_paths[player.player_id])
                if (
                    path.stat().st_size != player.checkpoint_bytes
                    or file_sha256(path) != player.checkpoint_sha256
                ):
                    raise ValueError("benchmark checkpoint identity mismatch")
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "plan.json", plan.model_dump(mode="json"))
    attempts: list[dict[str, Any]] = []
    numerical: dict[str, dict[str, Any]] = {}
    report_directories = []
    for repetition in range(plan.repetitions):
        for name in VARIANTS[repetition:] + VARIANTS[:repetition]:
            for panel in plan.panels:
                directory = out / f"{repetition}-{name}-{panel.name}"
                attempt: dict[str, Any] = {
                    "variant": name,
                    "repetition": repetition,
                    "panel": panel.name,
                }
                attempts.append(attempt)
                started = time.monotonic()
                try:
                    if time.time() >= plan.deadline_unix:
                        raise TimeoutError(
                            "benchmark deadline exhausted; unattempted panel"
                        )
                    execution = plan.configurations[name]
                    attempt.update(
                        _bounded_probe(
                            _panel_worker, (panel, execution, name, directory), plan
                        )
                    )
                    attempt.setdefault("seconds", time.monotonic() - started)
                    if not all(
                        (directory / name).exists()
                        for name in ("performance.json", "rows.json")
                    ):
                        continue
                    atomic_json(
                        directory / "report-run.json",
                        {"job": name + ":" + panel.name, "seed": repetition},
                    )
                    report_directories.append(directory)
                    if (
                        name == "cold_scalar"
                        and panel.name not in numerical
                        and attempt.get("trace")
                    ):
                        numerical[panel.name] = numerical_check(
                            plan, panel, Path(attempt["trace"])
                        )
                        atomic_json(out / "numerical.json", numerical)
                except Exception as exc:
                    attempt.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                finally:
                    attempt.setdefault("seconds", time.monotonic() - started)
                    atomic_json(out / "attempts.json", attempts)
    means: dict[str, list[float]] = defaultdict(list)
    for attempt in attempts:
        if attempt["passed"]:
            means[attempt["variant"]].append(attempt["seconds"])
    speedup = (
        float(np.mean(means["cold_scalar"]) / np.mean(means["batched"]))
        if means["cold_scalar"] and means["batched"]
        else None
    )
    overheads = [
        r["span_instrumentation_overhead"]
        for n in numerical.values()
        for r in n.get("results", [])
    ]
    overhead = max(overheads) if overheads else None
    passed = (
        all(a["passed"] for a in attempts)
        and len(numerical) == len(plan.panels)
        and all(n["passed"] for n in numerical.values())
        and overhead is not None
        and overhead <= plan.maximum_instrumentation_overhead
        and speedup is not None
        and speedup >= plan.minimum_speedup
    )
    result = {
        "attempts": attempts,
        "numerical": numerical,
        "speedup": speedup,
        "adoption_thresholds_met": passed,
        "performance_threshold_met": speedup is not None
        and speedup >= plan.minimum_speedup,
        "span_instrumentation_overhead": overhead,
        "instrumentation_population": "three interleaved warm retained-input schedules per mode; no game scheduling perturbation",
        "scope": "only the declared host, panels, device and numerical mode; no cross-device trajectory equivalence",
        "paid_execution": False,
    }
    atomic_json(out / "result.json", result)
    if report_directories:
        from experiments.report_prototype.build import write_evaluations

        write_evaluations(report_directories, out / "report")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.out)


if __name__ == "__main__":
    main()
