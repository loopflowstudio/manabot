#!/usr/bin/env python3
"""Run and verify the registered INT-4 production teacher/student iteration.

This is an additive production orchestrator.  The PR #133 smoke runner and its
contract remain immutable; this command binds that frozen iteration profile to
the exact Teacher-0 controls, matched-latency flat Monte Carlo, competencies,
resource receipts, and admission rules omitted by the smoke.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

from etude.study_protocol import StudyArtifact
from experiments.runners import run_visit_teacher_iteration as iteration
from manabot.sim.distill import ROOT_VALUE_KEY, load_shards
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.mcts import determinized_puct
from manabot.sim.teacher1_evidence import (
    ContractError,
    _fresh_env,
    _teacher_action,
    canonical_sha256,
    file_sha256,
    receipt_dict,
    replay_teacher_trajectories,
    source_bundle_sha256,
    validate_runtime_fingerprints,
)
from manabot.verify.competency import run_scenario_suite

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "int-4-visit-teacher-production-v1"
BASE_PROFILE = "iteration"
RESOURCE_LEDGER = "resource-ledger.jsonl"


class ResourceCapReached(RuntimeError):
    """Raised before launching work when a cumulative production cap is full."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ContractError(f"required JSON does not exist: {path}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"expected a JSON object in {path}")
    return payload


def _chip_name() -> str:
    completed = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        completed.stdout.strip() if completed.returncode == 0 else platform.processor()
    )


def _production_runtime(seed: int) -> dict[str, Any]:
    runtime = iteration._iteration_runtime(seed)
    runtime.update(
        production_source_sha256=source_bundle_sha256(
            [
                Path(__file__).resolve(),
                REPO_ROOT / "experiments/runners/run_visit_teacher_iteration.py",
                REPO_ROOT / "manabot/sim/teacher1_evidence.py",
                REPO_ROOT / "manabot/sim/mcts.py",
                REPO_ROOT / "manabot/sim/flat_mc.py",
                REPO_ROOT / "manabot/sim/distill.py",
                REPO_ROOT / "manabot/sim/search_supervised.py",
                REPO_ROOT / "manabot/sim/study_evidence.py",
                REPO_ROOT / "manabot/verify/competency.py",
                REPO_ROOT / "etude/study_protocol.py",
                REPO_ROOT / "protocol/study-v1.schema.json",
            ]
        ),
        chip=_chip_name(),
        operating_system=f"macOS {platform.mac_ver()[0]}",
    )
    return runtime


def _load_contracts(
    contract_path: Path, production_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any], str]:
    contract, contract_hash, profile, profile_hash = iteration._load_contract(
        contract_path, BASE_PROFILE
    )
    production = _load_json(production_path)
    if (
        production.get("schema_version") != 1
        or production.get("experiment") != EXPERIMENT
    ):
        raise ContractError("unexpected INT-4 production contract identity")
    base = production.get("base_iteration") or {}
    if (
        base.get("contract_sha256") != contract_hash
        or base.get("profile") != BASE_PROFILE
        or base.get("profile_sha256") != profile_hash
    ):
        raise ContractError("production contract does not bind the frozen iteration")
    controls = production.get("controls") or []
    if [item.get("arm") for item in controls] != ["policy_only", "policy_value"]:
        raise ContractError("production controls must be policy_only then policy_value")
    if any(not item.get("sha256") for item in controls):
        raise ContractError("every production control must have an exact SHA-256")
    caps = production.get("resource_accounting") or {}
    base_caps = contract.get("caps") or {}
    for field in ("workers", "wall_hours", "core_hours", "artifact_bytes"):
        if int(caps.get(field, -1)) != int(base_caps.get(field, -2)):
            raise ContractError(f"production {field} changed the frozen iteration cap")
    if (
        caps.get("ledger") != RESOURCE_LEDGER
        or caps.get("ledger_schema_version") != 1
        or caps.get("ledger_mode") != "append_only_hash_chained_attempt_receipts"
        or caps.get("failed_attempts_charged") is not True
        or caps.get("launch_requires_remaining_capacity") is not True
    ):
        raise ContractError("production resource ledger contract drifted")
    return (
        contract,
        contract_hash,
        profile,
        profile_hash,
        production,
        canonical_sha256(production),
    )


def _validate_production_runtime(
    production: dict[str, Any], runtime: dict[str, Any]
) -> None:
    expected = production.get("expected_runtime") or {}
    actual = {
        "production_source_sha256": runtime["production_source_sha256"],
        "chip": runtime["chip"],
        "machine": runtime["machine"],
        "operating_system": runtime["operating_system"],
        "python": runtime["python"],
        "torch": runtime["torch"],
        "mps_available": runtime["mps_available"],
    }
    if actual != expected:
        raise ContractError(
            f"production runtime identity {actual!r} != contract {expected!r}"
        )


def _control_receipt(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"frozen {expected['arm']} control does not exist: {path}")
    digest = file_sha256(path)
    if digest != expected["sha256"]:
        raise ContractError(
            f"frozen {expected['arm']} SHA-256 {digest} != {expected['sha256']}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    arm = (checkpoint.get("bc") or {}).get("arm")
    if arm != expected["arm"]:
        raise ContractError(
            f"frozen control metadata arm {arm!r} != {expected['arm']!r}"
        )
    agent, obs_space = load_checkpoint_agent(str(path))
    parameter_count = sum(parameter.numel() for parameter in agent.parameters())
    return {
        "arm": expected["arm"],
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "deterministic": bool(expected["deterministic"]),
        "parameter_count": parameter_count,
        "observation_shapes": {
            key: list(shape) for key, shape in obs_space.shapes.items()
        },
    }


def _validate_controls(
    production: dict[str, Any], policy_only: Path, policy_value: Path
) -> dict[str, Any]:
    paths = {"policy_only": policy_only, "policy_value": policy_value}
    return {
        expected["arm"]: _control_receipt(paths[expected["arm"]], expected)
        for expected in production["controls"]
    }


def _usage() -> dict[str, float | int]:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    own_rss = int(own.ru_maxrss)
    child_rss = int(children.ru_maxrss)
    scale = 1 if sys.platform == "darwin" else 1024
    return {
        "user_cpu_seconds": float(own.ru_utime + children.ru_utime),
        "system_cpu_seconds": float(own.ru_stime + children.ru_stime),
        "peak_rss_bytes": max(own_rss, child_rss) * scale,
    }


def _resource_receipt(
    before: dict[str, float | int], started: float
) -> dict[str, float | int]:
    after = _usage()
    return {
        "wall_seconds": time.perf_counter() - started,
        "user_cpu_seconds": float(after["user_cpu_seconds"])
        - float(before["user_cpu_seconds"]),
        "system_cpu_seconds": float(after["system_cpu_seconds"])
        - float(before["system_cpu_seconds"]),
        "peak_rss_bytes": int(after["peak_rss_bytes"]),
    }


def _ledger_path(out_dir: Path) -> Path:
    return out_dir / RESOURCE_LEDGER


def _read_resource_ledger(out_dir: Path) -> list[dict[str, Any]]:
    path = _ledger_path(out_dir)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(
                f"resource ledger line {line_number} is not valid JSON"
            ) from error
        if not isinstance(event, dict):
            raise ContractError(f"resource ledger line {line_number} is not an object")
        event_hash = event.get("event_sha256")
        unsigned = dict(event)
        unsigned.pop("event_sha256", None)
        if (
            event.get("sequence") != line_number
            or event.get("previous_event_sha256") != previous
            or event_hash != canonical_sha256(unsigned)
        ):
            raise ContractError(f"resource ledger chain drifted at line {line_number}")
        resources = event.get("resources") or {}
        numeric_resources = (
            "wall_seconds",
            "user_cpu_seconds",
            "system_cpu_seconds",
            "peak_rss_bytes",
        )
        if (
            event.get("scope") not in {"job", "stage"}
            or event.get("status") not in {"completed", "failed"}
            or any(
                not isinstance(resources.get(field), (int, float))
                or not math.isfinite(float(resources[field]))
                or float(resources[field]) < 0
                for field in numeric_resources
            )
        ):
            raise ContractError(
                f"resource ledger receipt invalid at line {line_number}"
            )
        events.append(event)
        previous = str(event_hash)
    return events


def _append_resource_event(
    out_dir: Path,
    *,
    scope: str,
    name: str,
    attempt: int,
    status: str,
    input_sha256: str,
    resources: dict[str, Any],
    result_path: str | None = None,
    result_sha256: str | None = None,
) -> dict[str, Any]:
    events = _read_resource_ledger(out_dir)
    unsigned = {
        "schema_version": 1,
        "sequence": len(events) + 1,
        "previous_event_sha256": (events[-1]["event_sha256"] if events else None),
        "recorded_at": datetime.now(UTC).isoformat(),
        "scope": scope,
        "name": name,
        "attempt": attempt,
        "status": status,
        "input_sha256": input_sha256,
        "resources": resources,
        "artifact_bytes_after": _directory_bytes(out_dir),
        "result_path": result_path,
        "result_sha256": result_sha256,
    }
    event = {**unsigned, "event_sha256": canonical_sha256(unsigned)}
    path = _ledger_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def _resource_ledger_receipt(out_dir: Path) -> dict[str, Any]:
    events = _read_resource_ledger(out_dir)
    path = _ledger_path(out_dir)
    return {
        "path": str(path.relative_to(out_dir)),
        "entries": len(events),
        "head_event_sha256": events[-1]["event_sha256"] if events else None,
        "sha256": file_sha256(path) if path.exists() else None,
    }


def _validate_ledger_prefix(manifest: dict[str, Any], out_dir: Path) -> None:
    receipt = manifest.get("resource_ledger")
    events = _read_resource_ledger(out_dir)
    if receipt is None:
        if events:
            raise ContractError("resource ledger exists without a manifest receipt")
        return
    entries = int(receipt.get("entries", -1))
    if entries < 0 or len(events) < entries:
        raise ContractError("resource ledger is shorter than its persisted receipt")
    if entries:
        prefix_lines = (
            _ledger_path(out_dir).read_bytes().splitlines(keepends=True)[:entries]
        )
        prefix_sha256 = hashlib.sha256(b"".join(prefix_lines)).hexdigest()
        head = events[entries - 1]["event_sha256"]
    else:
        prefix_sha256 = None
        head = None
    if (
        receipt.get("head_event_sha256") != head
        or receipt.get("sha256") != prefix_sha256
    ):
        raise ContractError("resource ledger no longer extends its manifest receipt")


def _attempt_number(events: list[dict[str, Any]], scope: str, name: str) -> int:
    return (
        max(
            (
                int(event["attempt"])
                for event in events
                if event.get("scope") == scope and event.get("name") == name
            ),
            default=0,
        )
        + 1
    )


def _ledger_event(events: list[dict[str, Any]], event_sha256: str) -> dict[str, Any]:
    matches = [event for event in events if event["event_sha256"] == event_sha256]
    if len(matches) != 1:
        raise ContractError(f"resource ledger event {event_sha256} is not unique")
    return matches[0]


def _bound_artifact_path(out_dir: Path, relative: Any, description: str) -> Path:
    relative_path = Path(str(relative))
    if relative_path.is_absolute():
        raise ContractError(f"{description} must be relative to the production run")
    path = (out_dir / relative_path).resolve()
    if not path.is_relative_to(out_dir.resolve()):
        raise ContractError(f"{description} escapes the production run")
    return path


def _relative_gap(left: float, right: float) -> float:
    if not math.isfinite(left) or not math.isfinite(right) or left <= 0 or right < 0:
        raise ContractError("latency values must be finite with a positive reference")
    return abs(right - left) / left


def _run_calibration(
    profile: dict[str, Any], production: dict[str, Any]
) -> dict[str, Any]:
    calibration = production["calibration"]
    seed = int(calibration["seed"])
    roots = int(calibration["roots"])
    warmup = int(calibration["warmup_roots"])
    budgets = [int(value) for value in profile["teacher_budgets"]]
    candidate_min = int(calibration["flat_sims_per_action_min"])
    candidate_max = int(calibration["flat_sims_per_action_max"])
    if roots < 1 or warmup < 0 or candidate_min < 1 or candidate_max < candidate_min:
        raise ContractError("invalid production calibration bounds")

    teacher_seconds = {budget: [] for budget in budgets}
    flat_seconds = {sims: [] for sims in range(candidate_min, candidate_max + 1)}
    env = _fresh_env(seed)
    env.reset(seed=seed)
    measured = 0
    root_index = 0
    game_index = 0
    while measured < roots:
        if env.last_raw_obs.game_over:
            game_index += 1
            env.reset(seed=seed + game_index)
        root_results: dict[int, Any] = {}
        for budget in budgets:
            started = time.perf_counter()
            root_results[budget] = determinized_puct(
                env._engine,
                simulations=budget,
                worlds=int(profile["worlds"]),
                c_puct=float(profile["c_puct"]),
                seed=seed * 1_000_003 + root_index * 257 + budget,
                max_steps=int(profile["max_steps"]),
            )
            elapsed = time.perf_counter() - started
            if root_index >= warmup:
                teacher_seconds[budget].append(elapsed)
        for sims in flat_seconds:
            rollouts = max(1, min(4, sims))
            worlds = max(1, sims // rollouts)
            started = time.perf_counter()
            env._engine.flat_mc_scores(
                worlds,
                rollouts,
                seed * 1_000_033 + root_index * 521 + sims,
                int(profile["max_steps"]),
            )
            elapsed = time.perf_counter() - started
            if root_index >= warmup:
                flat_seconds[sims].append(elapsed)
        action = _teacher_action(root_results[max(budgets)])
        _, _, terminated, truncated, _ = env.step(action)
        root_index += 1
        if root_index > warmup:
            measured += 1
        if terminated or truncated:
            game_index += 1
            env.reset(seed=seed + game_index)

    teacher_p50 = {
        budget: float(np.quantile(values, 0.50)) * 1000
        for budget, values in teacher_seconds.items()
    }
    flat_p50 = {
        sims: float(np.quantile(values, 0.50)) * 1000
        for sims, values in flat_seconds.items()
    }
    matches: dict[str, Any] = {}
    for budget in budgets:
        chosen = min(
            flat_p50,
            key=lambda sims: (
                _relative_gap(teacher_p50[budget], flat_p50[sims]),
                sims,
            ),
        )
        gap = _relative_gap(teacher_p50[budget], flat_p50[chosen])
        matches[str(budget)] = {
            "teacher_p50_decision_ms": teacher_p50[budget],
            "flat_sims_per_action": chosen,
            "flat_p50_decision_ms": flat_p50[chosen],
            "relative_gap": gap,
            "calibration_passed": gap <= float(calibration["max_realized_p50_gap"]),
        }
    return {
        "seed": seed,
        "roots": roots,
        "warmup_roots": warmup,
        "candidate_range": [candidate_min, candidate_max],
        "teacher_p50_decision_ms": {str(k): v for k, v in teacher_p50.items()},
        "flat_p50_decision_ms": {str(k): v for k, v in flat_p50.items()},
        "matches": matches,
    }


def _worker_job(job: dict[str, Any], output_path: Path) -> None:
    before = _usage()
    started = time.perf_counter()
    kind = job["kind"]
    if kind == "play_cell":
        result = iteration._play_cell(
            job["hero"],
            job["villain"],
            blocks=list(job["blocks"]),
            seed_offset=int(job.get("seed_offset", 0)),
        )
    elif kind == "competency":
        raw_path = output_path.with_suffix(".raw.json")
        result = run_scenario_suite(
            list(job["scenarios"]),
            list(job["agents"]),
            runs=int(job["runs"]),
            workers=int(job["workers"]),
            base_seed=int(job["seed"]),
            out_path=raw_path,
        )
    elif kind == "calibration":
        result = _run_calibration(job["profile"], job["production"])
    elif kind == "stability":
        from manabot.sim.teacher1_evidence import evaluate_root_stability

        result = evaluate_root_stability(
            budgets=[int(value) for value in job["budgets"]],
            worlds=int(job["worlds"]),
            c_puct=float(job["c_puct"]),
            roots=int(job["roots"]),
            repeat_seeds=[int(value) for value in job["repeat_seeds"]],
            seed=int(job["seed"]),
            max_steps=int(job["max_steps"]),
        )
    else:
        raise ContractError(f"unknown production worker kind {kind!r}")
    payload = {
        "schema_version": 1,
        "input_sha256": canonical_sha256(job),
        "kind": kind,
        "result": result,
        "resources": _resource_receipt(before, started),
    }
    _atomic_json(output_path, payload)


_JOB_REFERENCE_FIELDS = {
    "job_name",
    "job_path",
    "result_path",
    "output_sha256",
    "ledger_event_sha256",
}


def _job_reference(
    out_dir: Path,
    name: str,
    job_path: Path,
    output_path: Path,
    output: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    return {
        **output,
        "job_name": name,
        "job_path": str(job_path.relative_to(out_dir)),
        "result_path": str(output_path.relative_to(out_dir)),
        "output_sha256": file_sha256(output_path),
        "ledger_event_sha256": event["event_sha256"],
    }


def _verify_job_reference(
    out_dir: Path,
    reference: dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    events = _read_resource_ledger(out_dir) if events is None else events
    name = str(reference.get("job_name"))
    job_path = _bound_artifact_path(out_dir, reference.get("job_path"), "job input")
    result_path = _bound_artifact_path(
        out_dir, reference.get("result_path"), "job result"
    )
    if not job_path.is_file() or not result_path.is_file():
        raise ContractError(f"production job {name} is missing bound files")
    job = _load_json(job_path)
    output = _load_json(result_path)
    input_sha256 = canonical_sha256(job)
    output_sha256 = file_sha256(result_path)
    event = _ledger_event(events, str(reference.get("ledger_event_sha256")))
    if (
        event.get("scope") != "job"
        or event.get("name") != name
        or event.get("status") != "completed"
        or event.get("input_sha256") != input_sha256
        or event.get("result_path") != str(result_path.relative_to(out_dir))
        or event.get("result_sha256") != output_sha256
        or reference.get("input_sha256") != input_sha256
        or reference.get("output_sha256") != output_sha256
    ):
        raise ContractError(f"production job {name} digest binding drifted")
    copied_output = {
        key: value
        for key, value in reference.items()
        if key not in _JOB_REFERENCE_FIELDS
    }
    if copied_output != output:
        raise ContractError(f"production job {name} stage copy drifted")
    return output


def _execute_job(
    out_dir: Path,
    name: str,
    job: dict[str, Any],
    production: dict[str, Any],
) -> dict[str, Any]:
    if Path(name).name != name:
        raise ContractError(f"invalid production job name {name!r}")
    jobs = out_dir / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    job_path = jobs / f"{name}.job.json"
    digest = canonical_sha256(job)
    events = _read_resource_ledger(out_dir)
    completed_events = [
        event
        for event in events
        if event.get("scope") == "job"
        and event.get("name") == name
        and event.get("status") == "completed"
    ]
    if completed_events:
        event = completed_events[-1]
        if event.get("input_sha256") != digest:
            raise ContractError(f"completed production job {name} has different inputs")
        output_path = _bound_artifact_path(
            out_dir, event["result_path"], "cached job result"
        )
        output = _load_json(output_path)
        reference = _job_reference(out_dir, name, job_path, output_path, output, event)
        _verify_job_reference(out_dir, reference, events=events)
        return reference
    if job_path.exists():
        if canonical_sha256(_load_json(job_path)) != digest:
            raise ContractError(f"production job {name} has different saved inputs")
    else:
        _atomic_json(job_path, job)
    _require_launch_capacity(out_dir, production, f"job {name}")
    attempt = _attempt_number(events, "job", name)
    output_path = jobs / f"{name}.attempt-{attempt:04d}.result.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-job",
        str(job_path),
        "--worker-output",
        str(output_path),
    ]
    before = _usage()
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": "0", "WANDB_MODE": "disabled"},
            timeout=_remaining_wall_seconds(out_dir, production),
        )
        if completed.returncode:
            raise RuntimeError(
                f"production child {name} failed ({completed.returncode})"
            )
        output = _load_json(output_path)
        if output.get("input_sha256") != digest:
            raise ContractError(f"production child {name} returned the wrong identity")
        output_sha256 = file_sha256(output_path)
    except Exception:
        resources = _resource_receipt(before, started)
        _append_resource_event(
            out_dir,
            scope="job",
            name=name,
            attempt=attempt,
            status="failed",
            input_sha256=digest,
            resources=resources,
        )
        raise
    resources = _resource_receipt(before, started)
    event = _append_resource_event(
        out_dir,
        scope="job",
        name=name,
        attempt=attempt,
        status="completed",
        input_sha256=digest,
        resources=resources,
        result_path=str(output_path.relative_to(out_dir)),
        result_sha256=output_sha256,
    )
    return _job_reference(out_dir, name, job_path, output_path, output, event)


def _checkpoint_spec(receipt: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "kind": "checkpoint",
        "name": name,
        "path": receipt["path"],
        "deterministic": True,
    }


def _flat_spec(profile: dict[str, Any], sims: int, name: str) -> dict[str, Any]:
    return {
        "kind": "search",
        "name": name,
        "sims": sims,
        "max_steps": int(profile["max_steps"]),
    }


def _run_teacher(
    out_dir: Path,
    profile: dict[str, Any],
    production: dict[str, Any],
    controls: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    blocks = list(profile["teacher_seed_blocks"])
    cells: dict[str, Any] = {}
    opponents = {
        "random": {"kind": "random"},
        "policy_only": _checkpoint_spec(controls["policy_only"], "control-policy-only"),
        "policy_value": _checkpoint_spec(
            controls["policy_value"], "control-policy-value"
        ),
    }
    for budget_value in profile["teacher_budgets"]:
        budget = int(budget_value)
        teacher = iteration._teacher_spec(profile, budget)
        for opponent_name, opponent in opponents.items():
            name = f"t1-{budget}-vs-{opponent_name}"
            cells[name] = _execute_job(
                out_dir,
                name,
                {
                    "kind": "play_cell",
                    "hero": teacher,
                    "villain": opponent,
                    "blocks": blocks,
                },
                production,
            )
        matched = calibration["matches"][str(budget)]
        flat_sims = int(matched["flat_sims_per_action"])
        name = f"t1-{budget}-vs-flat-wall"
        cells[name] = _execute_job(
            out_dir,
            name,
            {
                "kind": "play_cell",
                "hero": teacher,
                "villain": _flat_spec(
                    profile, flat_sims, f"flat-{flat_sims}-matched-t1-{budget}"
                ),
                "blocks": blocks,
            },
            production,
        )
    high = int(profile["high_budget"])
    low = min(int(value) for value in profile["teacher_budgets"])
    name = f"t1-{high}-vs-t1-{low}"
    cells[name] = _execute_job(
        out_dir,
        name,
        {
            "kind": "play_cell",
            "hero": iteration._teacher_spec(profile, high),
            "villain": iteration._teacher_spec(profile, low),
            "blocks": blocks,
        },
        production,
    )

    stability = _execute_job(
        out_dir,
        "teacher-root-stability",
        {
            "kind": "stability",
            "budgets": [int(value) for value in profile["teacher_budgets"]],
            "worlds": int(profile["worlds"]),
            "c_puct": float(profile["c_puct"]),
            "roots": int(profile["stability_roots"]),
            "repeat_seeds": [int(value) for value in profile["stability_repeat_seeds"]],
            "seed": int(profile["stability_seed"]),
            "max_steps": int(profile["max_steps"]),
        },
        production,
    )

    competency_specs = [{"kind": "random", "name": "random"}]
    competency_specs.extend(
        iteration._teacher_spec(profile, int(budget))
        for budget in profile["teacher_budgets"]
    )
    competency_specs.extend(
        [
            _checkpoint_spec(controls["policy_only"], "control-policy-only"),
            _checkpoint_spec(controls["policy_value"], "control-policy-value"),
        ]
    )
    competencies = _execute_job(
        out_dir,
        "teacher-competencies",
        {
            "kind": "competency",
            "scenarios": list(production["competencies"]["scenarios"]),
            "agents": competency_specs,
            "runs": int(production["competencies"]["runs_per_agent_scenario"]),
            "workers": int(production["resource_accounting"]["workers"]),
            "seed": int(production["competencies"]["seed"]),
        },
        production,
    )
    result = {
        "cells": cells,
        "root_stability": stability,
        "competencies": competencies,
    }
    result["gate"] = _teacher_gate(result, profile, production)
    return result


def _result(job: dict[str, Any]) -> dict[str, Any]:
    return job["result"]


def _competency_rate(
    payload: dict[str, Any], scenario: str, agent: str
) -> float | None:
    item = payload.get(scenario, {}).get(agent, {})
    if "error" in item or "correct_rate" not in item:
        return None
    return float(item["correct_rate"])


def _teacher_gate(
    teacher: dict[str, Any], profile: dict[str, Any], production: dict[str, Any]
) -> dict[str, Any]:
    gates = production["teacher_gates"]
    high = int(profile["high_budget"])
    low = min(int(value) for value in profile["teacher_budgets"])
    cells = teacher["cells"]
    high_random = _result(cells[f"t1-{high}-vs-random"])
    high_control = _result(cells[f"t1-{high}-vs-policy_value"])
    high_low = _result(cells[f"t1-{high}-vs-t1-{low}"])
    high_search = high_random["hero_search"]
    stability = _result(teacher["root_stability"])
    stability_low = stability[str(low)]
    stability_high = stability[str(high)]
    competencies = _result(teacher["competencies"])
    delayed = []
    competency_nonregression = True
    competency_details: dict[str, Any] = {}
    high_name = f"t1-{high}-w{profile['worlds']}"
    for scenario in production["competencies"]["scenarios"]:
        high_rate = _competency_rate(competencies, scenario, high_name)
        random_rate = _competency_rate(competencies, scenario, "random")
        competency_details[scenario] = {"teacher": high_rate, "random": random_rate}
        if high_rate is None or random_rate is None:
            competency_nonregression = False
            high_rate = 0.0
            random_rate = 0.0
        if scenario in {
            "s1_counter_the_bomb",
            "s2_hold_the_wipe",
            "s5_hold_up_quench",
        }:
            delayed.append(high_rate)
        competency_nonregression &= high_rate >= random_rate - float(
            gates["competency_max_regression_vs_random"]
        )
    matched: dict[str, Any] = {}
    for budget in profile["teacher_budgets"]:
        budget = int(budget)
        cell = _result(cells[f"t1-{budget}-vs-flat-wall"])
        teacher_p50 = float(cell["hero_search"]["p50_decision_ms"])
        flat_p50 = float(cell["villain_search"]["p50_decision_ms"])
        gap = _relative_gap(teacher_p50, flat_p50)
        matched[str(budget)] = {
            "teacher_p50_decision_ms": teacher_p50,
            "flat_p50_decision_ms": flat_p50,
            "relative_gap": gap,
            "passed": gap <= float(production["calibration"]["max_realized_p50_gap"]),
        }
    checks = {
        "high_vs_low": float(high_low["win_rate"])
        >= float(gates["high_vs_low_win_rate"]),
        "high_vs_policy_value": float(high_control["win_rate"])
        >= float(gates["high_vs_policy_value_win_rate"]),
        "high_vs_random": float(high_random["win_rate"])
        >= float(gates["high_vs_random_win_rate"]),
        "top_action_stability": float(stability_high["top_action_agreement"])
        >= float(gates["high_top_action_agreement"]),
        "js_stability": float(stability_high["median_js_divergence"])
        <= float(gates["high_median_js_divergence"]),
        "tree_growth": float(stability_high["mean_tree_nodes"])
        > float(stability_low["mean_tree_nodes"]),
        "depth_growth": float(stability_high["mean_max_depth"])
        > float(stability_low["mean_max_depth"]),
        "competency_signal": max(delayed, default=0.0)
        >= float(gates["delayed_scenario_correct_rate"]),
        "competency_nonregression": competency_nonregression,
        "latency": float(high_search["p95_decision_ms"])
        <= float(gates["high_p95_decision_ms"]),
        "throughput": float(high_search["labels_per_second"])
        >= float(gates["high_labels_per_second_per_worker"]),
        "cap_rate": float(high_search["cap_rate"])
        < float(gates["max_playout_cap_rate"]),
        "matched_compute": all(item["passed"] for item in matched.values()),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "matched_compute": matched,
        "competencies": competency_details,
    }


def _calibration_diagnostics(dataset_stage: dict[str, Any]) -> dict[str, Any]:
    dataset = load_shards([Path(item["path"]) for item in dataset_stage["shards"]])
    predictions = np.asarray(dataset[ROOT_VALUE_KEY], dtype=np.float64)
    outcomes = (
        np.asarray(dataset["winner"], dtype=np.int64)
        == np.asarray(dataset["seat"], dtype=np.int64)
    ).astype(np.float64)
    brier = float(np.mean(np.square(predictions - outcomes)))
    bins = []
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (predictions >= lower) & (
            predictions <= upper if upper >= 1.0 else predictions < upper
        )
        count = int(mask.sum())
        if count:
            confidence = float(predictions[mask].mean())
            outcome = float(outcomes[mask].mean())
            ece += count / len(predictions) * abs(confidence - outcome)
        else:
            confidence = None
            outcome = None
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_prediction": confidence,
                "outcome_rate": outcome,
            }
        )
    return {"root_value_brier": brier, "ece_10_bin": ece, "reliability": bins}


def _villain_win_rate(cell: dict[str, Any]) -> float:
    games = float(cell["num_games"])
    if games <= 0:
        return 0.0
    return (games - float(cell["wins"]) - float(cell.get("draws_or_caps", 0.0))) / games


def _run_arena(
    out_dir: Path,
    profile: dict[str, Any],
    production: dict[str, Any],
    controls: dict[str, Any],
    training: dict[str, Any],
    teacher: dict[str, Any],
) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    competency_specs: list[dict[str, Any]] = [
        _checkpoint_spec(controls["policy_value"], "control-policy-value")
    ]
    for seed_value in profile["training_seeds"]:
        seed = int(seed_value)
        policy = iteration._checkpoint(training, seed, "visit_policy_only")
        student = iteration._checkpoint(training, seed, "visit_policy_value")
        players = {
            "teacher": iteration._teacher_spec(profile, int(profile["high_budget"])),
            "policy-only": {
                "kind": "checkpoint",
                "name": f"visit-policy-only-{seed}",
                "path": policy["path"],
                "deterministic": True,
            },
            "student": {
                "kind": "checkpoint",
                "name": f"visit-policy-value-{seed}",
                "path": student["path"],
                "deterministic": True,
            },
            "student+search": {
                "kind": "agent_puct",
                "name": f"visit-policy-value-search-{seed}",
                "checkpoint": student["path"],
                "sims": int(profile["student_search_simulations"]),
                "worlds": int(profile["worlds"]),
                "c_puct": float(profile["c_puct"]),
                "max_steps": int(profile["max_steps"]),
                "device": "cpu",
            },
        }
        round_robin: dict[str, Any] = {}
        for left, right in itertools.combinations(players, 2):
            cell_name = f"{left}-vs-{right}"
            round_robin[cell_name] = _execute_job(
                out_dir,
                f"arena-{seed}-{cell_name}",
                {
                    "kind": "play_cell",
                    "hero": players[left],
                    "villain": players[right],
                    "blocks": list(profile["arena_seed_blocks"]),
                    "seed_offset": seed * 10_000,
                },
                production,
            )
        chosen_policy = iteration._checkpoint(training, seed, "chosen_policy_only")
        chosen_value = iteration._checkpoint(training, seed, "chosen_policy_value")
        ablation_specs = {
            "visit_policy_only-vs-chosen_policy_only": (
                players["policy-only"],
                {
                    "kind": "checkpoint",
                    "name": f"chosen-policy-only-{seed}",
                    "path": chosen_policy["path"],
                    "deterministic": True,
                },
            ),
            "visit_policy_value-vs-chosen_policy_value": (
                players["student"],
                {
                    "kind": "checkpoint",
                    "name": f"chosen-policy-value-{seed}",
                    "path": chosen_value["path"],
                    "deterministic": True,
                },
            ),
        }
        ablations = {
            name: _execute_job(
                out_dir,
                f"arena-{seed}-{name}",
                {
                    "kind": "play_cell",
                    "hero": specs[0],
                    "villain": specs[1],
                    "blocks": list(profile["ablation_seed_blocks"]),
                    "seed_offset": seed * 10_000,
                },
                production,
            )
            for name, specs in ablation_specs.items()
        }
        control_cells = {}
        for control_name in ("policy_only", "policy_value"):
            name = f"student-vs-control-{control_name}"
            control_cells[name] = _execute_job(
                out_dir,
                f"arena-{seed}-{name}",
                {
                    "kind": "play_cell",
                    "hero": players["student"],
                    "villain": _checkpoint_spec(
                        controls[control_name],
                        f"control-{control_name.replace('_', '-')}",
                    ),
                    "blocks": list(profile["arena_seed_blocks"]),
                    "seed_offset": seed * 10_000,
                },
                production,
            )
        per_seed[str(seed)] = {
            "round_robin": round_robin,
            "ablations": ablations,
            "controls": control_cells,
        }
        competency_specs.extend([players["student"], players["student+search"]])

    competencies = _execute_job(
        out_dir,
        "student-competencies",
        {
            "kind": "competency",
            "scenarios": list(production["competencies"]["scenarios"]),
            "agents": competency_specs,
            "runs": int(production["competencies"]["runs_per_agent_scenario"]),
            "workers": int(production["resource_accounting"]["workers"]),
            "seed": int(production["competencies"]["seed"]) + 1_000_000,
        },
        production,
    )
    high_random = _result(
        teacher["cells"][f"t1-{int(profile['high_budget'])}-vs-random"]
    )
    teacher_p50 = float(high_random["hero_search"]["p50_decision_ms"])
    latency_ratios = {}
    for seed in profile["training_seeds"]:
        cell = _result(per_seed[str(seed)]["round_robin"]["student-vs-student+search"])
        search_p50 = float(cell["villain_search"]["p50_decision_ms"])
        latency_ratios[str(seed)] = {
            "teacher_p50_decision_ms": teacher_p50,
            "student_search_p50_decision_ms": search_p50,
            "student_search_over_teacher": search_p50 / teacher_p50,
            "compute_class": "equal_128_traversals_not_equal_wall_clock",
        }
    result = {
        "per_seed": per_seed,
        "competencies": competencies,
        "student_search_latency": latency_ratios,
    }
    result["admission"] = _admission(result, profile, production)
    return result


def _all_competencies_within(
    payload: dict[str, Any],
    scenarios: list[str],
    candidate: str,
    baseline: str,
    tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    passed = True
    details = {}
    for scenario in scenarios:
        candidate_rate = _competency_rate(payload, scenario, candidate)
        baseline_rate = _competency_rate(payload, scenario, baseline)
        cell_passed = (
            candidate_rate is not None
            and baseline_rate is not None
            and candidate_rate >= baseline_rate - tolerance
        )
        passed &= cell_passed
        details[scenario] = {
            "candidate": candidate_rate,
            "baseline": baseline_rate,
            "passed": cell_passed,
        }
    return passed, details


def _cross_seed_strength(rates: list[float], gates: dict[str, Any]) -> bool:
    return bool(
        np.median(rates) >= float(gates["median_win_rate"])
        and sum(rate > float(gates["per_seed_win_rate"]) for rate in rates)
        >= int(gates["winning_training_seeds"])
    )


def _admission(
    arena: dict[str, Any], profile: dict[str, Any], production: dict[str, Any]
) -> dict[str, Any]:
    gates = production["admission_gates"]
    competencies = _result(arena["competencies"])
    scenarios = list(production["competencies"]["scenarios"])
    search_rates = []
    policy_only_rates = []
    policy_value_rates = []
    search_competencies: dict[str, Any] = {}
    student_competencies: dict[str, Any] = {}
    search_competency_passed = True
    student_competency_passed = True
    for seed_value in profile["training_seeds"]:
        seed = int(seed_value)
        block = arena["per_seed"][str(seed)]
        search_rates.append(
            _villain_win_rate(
                _result(block["round_robin"]["student-vs-student+search"])
            )
        )
        policy_only_rates.append(
            float(
                _result(block["controls"]["student-vs-control-policy_only"])["win_rate"]
            )
        )
        policy_value_rates.append(
            float(
                _result(block["controls"]["student-vs-control-policy_value"])[
                    "win_rate"
                ]
            )
        )
        search_ok, search_detail = _all_competencies_within(
            competencies,
            scenarios,
            f"visit-policy-value-search-{seed}",
            f"visit-policy-value-{seed}",
            float(gates["competency_max_regression"]),
        )
        student_ok, student_detail = _all_competencies_within(
            competencies,
            scenarios,
            f"visit-policy-value-{seed}",
            "control-policy-value",
            float(gates["competency_max_regression"]),
        )
        search_competency_passed &= search_ok
        student_competency_passed &= student_ok
        search_competencies[str(seed)] = search_detail
        student_competencies[str(seed)] = student_detail

    search_strength = _cross_seed_strength(search_rates, gates)
    student_strength = _cross_seed_strength(
        policy_only_rates, gates
    ) and _cross_seed_strength(policy_value_rates, gates)
    if search_strength and search_competency_passed:
        decision = "admit_student_search"
        disposition = "continue"
        diagnosis = "student_search_cleared_strength_and_competency_gates"
    elif student_strength and student_competency_passed:
        decision = "admit_student"
        disposition = "continue"
        diagnosis = "visit_value_student_cleared_both_frozen_controls"
    else:
        decision = "prototype_failure"
        disposition = "revise"
        diagnosis = "no_candidate_cleared_complete_strength_and_competency_gates"
    return {
        "decision": decision,
        "disposition": disposition,
        "diagnosis": diagnosis,
        "student_search_win_rates": search_rates,
        "student_vs_policy_only_win_rates": policy_only_rates,
        "student_vs_policy_value_win_rates": policy_value_rates,
        "checks": {
            "student_search_strength": search_strength,
            "student_search_competency": search_competency_passed,
            "student_strength": student_strength,
            "student_competency": student_competency_passed,
        },
        "student_search_competencies": search_competencies,
        "student_competencies": student_competencies,
    }


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _resource_totals(out_dir: Path) -> dict[str, Any]:
    events = _read_resource_ledger(out_dir)
    return {
        "wall_seconds": sum(
            float(event["resources"]["wall_seconds"]) for event in events
        ),
        "user_cpu_seconds": sum(
            float(event["resources"]["user_cpu_seconds"]) for event in events
        ),
        "system_cpu_seconds": sum(
            float(event["resources"]["system_cpu_seconds"]) for event in events
        ),
        "peak_rss_bytes": max(
            (int(event["resources"]["peak_rss_bytes"]) for event in events),
            default=0,
        ),
        "artifact_bytes": max(
            _directory_bytes(out_dir),
            max(
                (int(event.get("artifact_bytes_after", 0)) for event in events),
                default=0,
            ),
        ),
        "attempts": len(events),
        "failed_attempts": sum(event["status"] == "failed" for event in events),
    }


def _caps(out_dir: Path, production: dict[str, Any]) -> dict[str, Any]:
    totals = _resource_totals(out_dir)
    limits = production["resource_accounting"]
    wall_limit = float(limits["wall_hours"]) * 3600
    core_limit = float(limits["core_hours"]) * 3600
    artifact_limit = int(limits["artifact_bytes"])
    core_seconds = totals["user_cpu_seconds"] + totals["system_cpu_seconds"]
    checks = {
        "wall": totals["wall_seconds"] <= wall_limit,
        "core": core_seconds <= core_limit,
        "artifacts": totals["artifact_bytes"] <= artifact_limit,
    }
    launchable = {
        "wall": totals["wall_seconds"] < wall_limit,
        "core": core_seconds < core_limit,
        "artifacts": totals["artifact_bytes"] < artifact_limit,
    }
    return {
        "totals": totals,
        "limits": {
            "wall_seconds": wall_limit,
            "core_seconds": core_limit,
            "artifact_bytes": artifact_limit,
        },
        "checks": checks,
        "launchable_checks": launchable,
        "passed": all(checks.values()),
        "launchable": all(launchable.values()),
    }


def _require_launch_capacity(
    out_dir: Path, production: dict[str, Any], description: str
) -> None:
    caps = _caps(out_dir, production)
    if not caps["launchable"]:
        failed = [
            name for name, passed in caps["launchable_checks"].items() if not passed
        ]
        raise ResourceCapReached(
            f"cannot launch {description}; cumulative cap reached: {', '.join(failed)}"
        )


def _remaining_wall_seconds(out_dir: Path, production: dict[str, Any]) -> float:
    caps = _caps(out_dir, production)
    remaining = caps["limits"]["wall_seconds"] - caps["totals"]["wall_seconds"]
    if remaining <= 0:
        raise ResourceCapReached("cumulative wall cap is exhausted")
    return remaining


def _stage_residual_resources(
    aggregate: dict[str, Any],
    ledger_before: dict[str, Any],
    ledger_after: dict[str, Any],
) -> dict[str, Any]:
    return {
        "wall_seconds": max(
            0.0,
            float(aggregate["wall_seconds"])
            - (
                float(ledger_after["wall_seconds"])
                - float(ledger_before["wall_seconds"])
            ),
        ),
        "user_cpu_seconds": max(
            0.0,
            float(aggregate["user_cpu_seconds"])
            - (
                float(ledger_after["user_cpu_seconds"])
                - float(ledger_before["user_cpu_seconds"])
            ),
        ),
        "system_cpu_seconds": max(
            0.0,
            float(aggregate["system_cpu_seconds"])
            - (
                float(ledger_after["system_cpu_seconds"])
                - float(ledger_before["system_cpu_seconds"])
            ),
        ),
        "peak_rss_bytes": int(aggregate["peak_rss_bytes"]),
    }


def _load_bound_stage_result(
    manifest: dict[str, Any],
    out_dir: Path,
    name: str,
    *,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stage = manifest.get("stages", {}).get(name) or {}
    if stage.get("status") != "completed":
        raise ContractError(f"production stage {name} is incomplete")
    result_path = _bound_artifact_path(
        out_dir, stage.get("result_path"), "stage result"
    )
    if not result_path.is_file():
        raise ContractError(f"production stage {name} result is missing")
    result_sha256 = file_sha256(result_path)
    result = _load_json(result_path)
    events = _read_resource_ledger(out_dir) if events is None else events
    event = _ledger_event(events, str(stage.get("ledger_event_sha256")))
    if (
        result_sha256 != stage.get("result_sha256")
        or result != stage.get("result")
        or event.get("scope") != "stage"
        or event.get("name") != name
        or event.get("status") != "completed"
        or event.get("input_sha256") != stage.get("input_sha256")
        or event.get("result_path") != str(result_path.relative_to(out_dir))
        or event.get("result_sha256") != result_sha256
    ):
        raise ContractError(f"production stage {name} result binding drifted")
    return result


def _run_stage(
    manifest: dict[str, Any],
    manifest_path: Path,
    out_dir: Path,
    production: dict[str, Any],
    name: str,
    inputs: dict[str, Any],
    function: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    digest = canonical_sha256(inputs)
    _validate_ledger_prefix(manifest, out_dir)
    existing = manifest["stages"].get(name)
    if existing and existing.get("status") == "completed":
        if existing.get("input_sha256") != digest:
            raise ContractError(
                f"completed production stage {name} has different inputs"
            )
        return _load_bound_stage_result(manifest, out_dir, name)
    _require_launch_capacity(out_dir, production, f"stage {name}")
    events = _read_resource_ledger(out_dir)
    attempt = _attempt_number(events, "stage", name)
    manifest["stages"][name] = {
        "status": "running",
        "attempt": attempt,
        "input_sha256": digest,
        "started_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(manifest_path, manifest)
    ledger_before = _resource_totals(out_dir)
    usage_before = _usage()
    started = time.perf_counter()
    try:
        result = function()
    except Exception as error:
        aggregate = _resource_receipt(usage_before, started)
        accounted = _stage_residual_resources(
            aggregate, ledger_before, _resource_totals(out_dir)
        )
        event = _append_resource_event(
            out_dir,
            scope="stage",
            name=name,
            attempt=attempt,
            status="failed",
            input_sha256=digest,
            resources=accounted,
        )
        manifest["stages"][name].update(
            status="failed",
            failed_at=datetime.now(UTC).isoformat(),
            error=f"{type(error).__name__}: {error}",
            aggregate_resources=aggregate,
            accounted_resources=accounted,
            ledger_event_sha256=event["event_sha256"],
        )
        if isinstance(error, ResourceCapReached):
            manifest["status"] = "inconclusive_resource_cap"
        manifest["resource_ledger"] = _resource_ledger_receipt(out_dir)
        manifest["resources"] = _caps(out_dir, production)
        _atomic_json(manifest_path, manifest)
        raise
    aggregate = _resource_receipt(usage_before, started)
    accounted = _stage_residual_resources(
        aggregate, ledger_before, _resource_totals(out_dir)
    )
    stage_results = out_dir / "stage-results"
    stage_results.mkdir(parents=True, exist_ok=True)
    result_path = stage_results / f"{name}.result.json"
    _atomic_json(result_path, result)
    result_sha256 = file_sha256(result_path)
    event = _append_resource_event(
        out_dir,
        scope="stage",
        name=name,
        attempt=attempt,
        status="completed",
        input_sha256=digest,
        resources=accounted,
        result_path=str(result_path.relative_to(out_dir)),
        result_sha256=result_sha256,
    )
    manifest["stages"][name] = {
        "status": "completed",
        "attempt": attempt,
        "input_sha256": digest,
        "finished_at": datetime.now(UTC).isoformat(),
        "aggregate_resources": aggregate,
        "accounted_resources": accounted,
        "result_path": str(result_path.relative_to(out_dir)),
        "result_sha256": result_sha256,
        "ledger_event_sha256": event["event_sha256"],
        "result": result,
    }
    manifest["resource_ledger"] = _resource_ledger_receipt(out_dir)
    manifest["resources"] = _caps(out_dir, production)
    _atomic_json(manifest_path, manifest)
    return result


def _integrity(dataset: dict[str, Any]) -> dict[str, Any]:
    replay = dataset["replay"]
    checks = {
        "dataset_targets": bool(dataset["diagnostics"]["passed"]),
        "exact_replay": bool(replay["passed"]),
        "all_sampled_roots": int(replay["missing_sampled_search_roots"]) == 0,
        "learner_audit_alignment": int(dataset["learner_audit_mismatches"]) == 0,
        "viewer_boundary": int(replay["opponent_private_cards_exposed"]) == 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _write_run_report(path: Path, manifest: dict[str, Any]) -> None:
    teacher = manifest["stages"]["teacher"]["result"]
    arena = manifest["stages"]["arena"]["result"]
    dataset = manifest["stages"]["dataset"]["result"]
    admission = arena["admission"]
    lines = [
        "# INT-4 Production Visit Teacher Iteration",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Admission: `{admission['decision']}`",
        f"- Research disposition: `{admission['disposition']}`",
        f"- Teacher gate: `{teacher['gate']['passed']}`",
        f"- Integrity: `{manifest['integrity']['passed']}`",
        f"- Base contract: `{manifest['contract_sha256']}`",
        f"- Production contract: `{manifest['production_contract_sha256']}`",
        f"- Production source: `{manifest['runtime']['production_source_sha256']}`",
        "",
        "## Teacher controls",
        "",
        "| Cell | Win rate | p50 ms | p95 ms |",
        "|---|---:|---:|---:|",
    ]
    for name, job in teacher["cells"].items():
        cell = _result(job)
        search = cell.get("hero_search") or {}
        lines.append(
            f"| {name} | {float(cell['win_rate']):.3f} | "
            f"{float(search.get('p50_decision_ms') or 0):.2f} | "
            f"{float(search.get('p95_decision_ms') or 0):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Dataset",
            "",
            f"- Games: {dataset['diagnostics']['games']}",
            f"- Decisions: {dataset['diagnostics']['decisions']}",
            f"- Exact replay: `{dataset['replay']['passed']}`",
            f"- Root-value Brier: {dataset['calibration']['root_value_brier']:.4f}",
            f"- Ten-bin ECE: {dataset['calibration']['ece_10_bin']:.4f}",
            "",
            "## Admission",
            "",
            f"- Decision: `{admission['decision']}`",
            f"- Diagnosis: `{admission['diagnosis']}`",
            f"- Student+search paired rates: {admission['student_search_win_rates']}",
            f"- Student vs policy-only: {admission['student_vs_policy_only_win_rates']}",
            f"- Student vs policy/value: {admission['student_vs_policy_value_win_rates']}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _iter_job_references(value: Any) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if _JOB_REFERENCE_FIELDS <= value.keys():
            references.append(value)
        else:
            for nested in value.values():
                references.extend(_iter_job_references(nested))
    elif isinstance(value, list):
        for nested in value:
            references.extend(_iter_job_references(nested))
    return references


def _verify(
    out_dir: Path,
    contract_hash: str,
    profile: dict[str, Any],
    profile_hash: str,
    production_hash: str,
    runtime: dict[str, Any],
    controls: dict[str, Any],
    production: dict[str, Any],
) -> dict[str, Any]:
    manifest = _load_json(out_dir / "manifest.json")
    if (
        manifest.get("status") != "completed"
        or manifest.get("experiment") != EXPERIMENT
        or manifest.get("contract_sha256") != contract_hash
        or manifest.get("profile_sha256") != profile_hash
        or manifest.get("production_contract_sha256") != production_hash
        or manifest.get("runtime") != runtime
        or manifest.get("controls") != controls
    ):
        raise ContractError("production run identity drifted")
    required = ("calibration", "teacher", "dataset", "training", "arena", "study")
    events = _read_resource_ledger(out_dir)
    if manifest.get("resource_ledger") != _resource_ledger_receipt(out_dir):
        raise ContractError("production resource ledger identity drifted")
    stage_results = {
        name: _load_bound_stage_result(manifest, out_dir, name, events=events)
        for name in required
    }
    references = [
        reference
        for stage in stage_results.values()
        for reference in _iter_job_references(stage)
    ]
    for reference in references:
        _verify_job_reference(out_dir, reference, events=events)
    referenced_events = {reference["ledger_event_sha256"] for reference in references}
    completed_job_events = {
        event["event_sha256"]
        for event in events
        if event.get("scope") == "job" and event.get("status") == "completed"
    }
    if referenced_events != completed_job_events:
        raise ContractError("completed production jobs do not match stage results")
    dataset = stage_results["dataset"]
    for shard in dataset["shards"]:
        if file_sha256(shard["path"]) != shard["sha256"]:
            raise RuntimeError(f"dataset shard changed: {shard['path']}")
    if file_sha256(dataset["audit_path"]) != dataset["audit_sha256"]:
        raise RuntimeError("trajectory audit changed")
    audit = _load_json(Path(dataset["audit_path"]))
    replay = replay_teacher_trajectories(
        audit,
        content_hash=runtime["experience_content_hash"],
        asset_manifest_hash=runtime["asset_manifest_hash"],
        sampled_search_roots=list(profile["sampled_search_roots"]),
    )
    if receipt_dict(replay) != dataset["replay"]:
        raise RuntimeError("production exact replay receipt changed")
    training = stage_results["training"]
    for checkpoint in training["checkpoints"].values():
        if file_sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError(f"student checkpoint changed: {checkpoint['path']}")
        load_checkpoint_agent(checkpoint["path"])
    study = stage_results["study"]
    if file_sha256(study["path"]) != study["sha256"]:
        raise RuntimeError("Study artifact changed")
    StudyArtifact.model_validate_json(Path(study["path"]).read_text())
    iteration._validate_study_in_rust(Path(study["path"]))
    teacher = stage_results["teacher"]
    if _teacher_gate(teacher, profile, production) != teacher["gate"]:
        raise RuntimeError("teacher gate no longer recomputes exactly")
    arena = stage_results["arena"]
    if _admission(arena, profile, production) != arena["admission"]:
        raise RuntimeError("admission decision no longer recomputes exactly")
    if manifest.get("admission") != arena["admission"]:
        raise RuntimeError("manifest admission copy no longer matches bound arena")
    caps = _caps(out_dir, production)
    if not caps["passed"]:
        raise RuntimeError("verified production run exceeds its resource caps")
    verification = {
        "verified": True,
        "verified_at": datetime.now(UTC).isoformat(),
        "contract_sha256": contract_hash,
        "profile_sha256": profile_hash,
        "production_contract_sha256": production_hash,
        "production_source_sha256": runtime["production_source_sha256"],
        "controls": controls,
        "trajectory_replay": receipt_dict(replay),
        "study_sha256": study["sha256"],
        "teacher_gate": teacher["gate"],
        "admission": arena["admission"],
        "resources": caps,
    }
    _atomic_json(out_dir / "verification.json", verification)
    return verification


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--production-contract", type=Path)
    parser.add_argument("--policy-only-control", type=Path)
    parser.add_argument("--policy-value-control", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--print-runtime", action="store_true")
    parser.add_argument("--worker-job", type=Path)
    parser.add_argument("--worker-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker_job is not None:
        if args.worker_output is None:
            raise SystemExit("--worker-output is required with --worker-job")
        _worker_job(_load_json(args.worker_job), args.worker_output)
        return
    required = {
        "--contract": args.contract,
        "--production-contract": args.production_contract,
        "--policy-only-control": args.policy_only_control,
        "--policy-value-control": args.policy_value_control,
        "--out-dir": args.out_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    (
        contract,
        contract_hash,
        profile,
        profile_hash,
        production,
        production_hash,
    ) = _load_contracts(args.contract, args.production_contract)
    runtime = _production_runtime(seed=int(contract["runtime_seed"]))
    if args.print_runtime:
        print(json.dumps(runtime, indent=2, sort_keys=True))
        return
    validate_runtime_fingerprints(contract["expected_fingerprints"], runtime)
    _validate_production_runtime(production, runtime)
    controls = _validate_controls(
        production, args.policy_only_control, args.policy_value_control
    )
    out_dir = args.out_dir.resolve()
    if args.verify:
        print(
            json.dumps(
                _verify(
                    out_dir,
                    contract_hash,
                    profile,
                    profile_hash,
                    production_hash,
                    runtime,
                    controls,
                    production,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if (
            manifest.get("contract_sha256") != contract_hash
            or manifest.get("profile_sha256") != profile_hash
            or manifest.get("production_contract_sha256") != production_hash
            or manifest.get("runtime") != runtime
            or manifest.get("controls") != controls
        ):
            raise ContractError("existing production run has different frozen inputs")
    else:
        manifest = {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "contract_path": str(args.contract.resolve()),
            "contract_sha256": contract_hash,
            "profile": BASE_PROFILE,
            "profile_sha256": profile_hash,
            "production_contract_path": str(args.production_contract.resolve()),
            "production_contract_sha256": production_hash,
            "runtime": runtime,
            "controls": controls,
            "resource_ledger": _resource_ledger_receipt(out_dir),
            "stages": {},
        }
        _atomic_json(manifest_path, manifest)

    calibration_job = _run_stage(
        manifest,
        manifest_path,
        out_dir,
        production,
        "calibration",
        {
            "profile_sha256": profile_hash,
            "production_contract_sha256": production_hash,
            "controls": controls,
        },
        lambda: _execute_job(
            out_dir,
            "calibration",
            {"kind": "calibration", "profile": profile, "production": production},
            production,
        ),
    )
    calibration = _result(calibration_job)
    if not _caps(out_dir, production)["passed"]:
        manifest["status"] = "inconclusive_resource_cap"
        manifest["resources"] = _caps(out_dir, production)
        _atomic_json(manifest_path, manifest)
        return
    teacher = _run_stage(
        manifest,
        manifest_path,
        out_dir,
        production,
        "teacher",
        {
            "profile_sha256": profile_hash,
            "production_contract_sha256": production_hash,
            "controls": controls,
            "calibration_output_sha256": calibration_job["output_sha256"],
        },
        lambda: _run_teacher(out_dir, profile, production, controls, calibration),
    )
    if not _caps(out_dir, production)["passed"]:
        manifest["status"] = "inconclusive_resource_cap"
        manifest["resources"] = _caps(out_dir, production)
        _atomic_json(manifest_path, manifest)
        return
    dataset = _run_stage(
        manifest,
        manifest_path,
        out_dir,
        production,
        "dataset",
        {
            "profile_sha256": profile_hash,
            "runtime": runtime,
            "teacher_result_sha256": manifest["stages"]["teacher"]["result_sha256"],
        },
        lambda: _run_dataset_with_calibration(out_dir, profile, profile_hash, runtime),
    )
    manifest["integrity"] = _integrity(dataset)
    if not manifest["integrity"]["passed"]:
        manifest["status"] = "invalid_no_admission"
        _atomic_json(manifest_path, manifest)
        raise RuntimeError("production dataset failed the integrity boundary")
    training = _run_stage(
        manifest,
        manifest_path,
        out_dir,
        production,
        "training",
        {
            "profile_sha256": profile_hash,
            "dataset_manifest_sha256": dataset["manifest_sha256"],
        },
        lambda: iteration._run_training(
            out_dir, profile, dataset, profile_hash=profile_hash
        ),
    )
    if not _caps(out_dir, production)["passed"]:
        manifest["status"] = "inconclusive_resource_cap"
        manifest["resources"] = _caps(out_dir, production)
        _atomic_json(manifest_path, manifest)
        return
    arena = _run_stage(
        manifest,
        manifest_path,
        out_dir,
        production,
        "arena",
        {
            "profile_sha256": profile_hash,
            "training_manifest_sha256": training["manifest_sha256"],
            "teacher_result_sha256": manifest["stages"]["teacher"]["result_sha256"],
            "controls": controls,
        },
        lambda: _run_arena(out_dir, profile, production, controls, training, teacher),
    )
    study = _run_stage(
        manifest,
        manifest_path,
        out_dir,
        production,
        "study",
        {
            "profile_sha256": profile_hash,
            "dataset_manifest_sha256": dataset["manifest_sha256"],
            "training_manifest_sha256": training["manifest_sha256"],
            "runtime": runtime,
        },
        lambda: iteration._run_study(
            out_dir, contract, profile, dataset, training, runtime
        ),
    )
    del study
    cap_result = _caps(out_dir, production)
    manifest["resources"] = cap_result
    manifest["resource_ledger"] = _resource_ledger_receipt(out_dir)
    if not cap_result["passed"]:
        manifest["status"] = "inconclusive_resource_cap"
    else:
        manifest["status"] = "completed"
        manifest["admission"] = arena["admission"]
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    _atomic_json(manifest_path, manifest)
    _write_run_report(out_dir / "report.md", manifest)
    print(
        json.dumps(
            {"status": manifest["status"], "admission": arena["admission"]}, indent=2
        )
    )


def _run_dataset_with_calibration(
    out_dir: Path,
    profile: dict[str, Any],
    profile_hash: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    result = iteration._run_dataset(
        out_dir, profile, profile_hash=profile_hash, runtime=runtime
    )
    result["calibration"] = _calibration_diagnostics(result)
    return result


if __name__ == "__main__":
    main()
