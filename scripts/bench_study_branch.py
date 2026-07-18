#!/usr/bin/env python3
"""Measure and verify the production Etude Study branch lifecycle.

Invoke only through uv:

    uv run scripts/bench_study_branch.py measure
    uv run scripts/bench_study_branch.py verify
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Callable

import psutil

from etude.replay_index import (
    CanonicalReplayV1,
    DecisionNotFoundError,
    InvalidAddressError,
    ReplayDecisionAddress,
    restore_decision,
)
from etude.server import GameSession
from etude.study_branch import (
    STUDY_BRANCH_DRIVER,
    STUDY_COMMAND_PATH,
    StudyBranch,
    StudyBranchUnavailableError,
    StudyForkProvider,
    StudyReturnReceipt,
)
import managym

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/benchmarks/study-branch-contract-v1.md"
DEFAULT_RAW = ROOT / "experiments/data/rul-6-study-branch-v1.json"
DEFAULT_REPORT = ROOT / "experiments/rul-6-study-branch-v1.md"
SCHEMA = "etude.study-branch.performance.v1"
CONTRACT_ID = "etude.study-branch.contract.v1"
CANONICAL_ITERATIONS = 2_000
CANONICAL_RETAINED = 512
CANONICAL_WARMUP = 64
FIXED_TIME = "2026-07-17T00:00:00+00:00"
MAX_HERO_COMMANDS = 2_000
GATES = {
    "fork_p95_ns_max": 1_000_000,
    "apply_p95_ns_max": 1_500_000,
    "return_p95_ns_max": 1_000_000,
    "end_to_end_p95_ns_max": 3_000_000,
    "cycles_per_second_min": 500.0,
    "retained_rss_delta_bytes_max": 128 * 1024 * 1024,
}
EXPECTED_FAILURE_CASES = {
    "submit_before_publish",
    "unknown_offer_consumes_offer_set",
    "unsupported_native_surface",
    "invalid_address",
    "missing_address",
    "other_viewer",
    "retained_root_drift_return",
    "retained_root_drift_fork",
    "projected_incarnation_tamper",
}
EXPECTED_FAILURE_TYPES = {
    "submit_before_publish": "StudyBranchUnavailableError",
    "unknown_offer_consumes_offer_set": "StudyBranchUnavailableError",
    "unsupported_native_surface": "StudyBranchUnavailableError",
    "invalid_address": "InvalidAddressError",
    "missing_address": "DecisionNotFoundError",
    "other_viewer": "DecisionNotFoundError",
    "retained_root_drift_return": "StudyBranchUnavailableError",
    "retained_root_drift_fork": "StudyBranchUnavailableError",
    "projected_incarnation_tamper": None,
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def artifact_hash(payload: dict[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("artifact_sha256", None)
    return sha256_bytes(canonical_json(unhashed))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        raise RuntimeError("latency sample is empty")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return int(ordered[index])


def summarize_latencies(values: list[int]) -> dict[str, int]:
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise RuntimeError("latency samples must be non-negative integers")
    return {
        "count": len(values),
        "min_ns": min(values),
        "p50_ns": percentile(values, 0.50),
        "p95_ns": percentile(values, 0.95),
        "p99_ns": percentile(values, 0.99),
        "max_ns": max(values),
        "mean_ns": round(sum(values) / len(values)),
    }


def _extension_path() -> Path:
    package = Path(managym.__file__).resolve().parent
    candidates = sorted(package.glob("_managym*.so"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one compiled managym extension, found {len(candidates)}"
        )
    return candidates[0]


def source_paths() -> list[Path]:
    paths: set[Path] = set()
    for relative, suffixes in (
        ("etude", {".py"}),
        ("managym/src", {".rs"}),
        ("content/semantic/v1", {".json"}),
    ):
        paths.update(
            path
            for path in (ROOT / relative).rglob("*")
            if path.is_file() and path.suffix in suffixes
        )
    for relative in (
        "docs/benchmarks/study-branch-contract-v1.md",
        "managym/Cargo.lock",
        "managym/Cargo.toml",
        "managym/__init__.py",
        "managym/__init__.pyi",
        "pyproject.toml",
        "scripts/bench_study_branch.py",
        "tests/bench/test_study_branch_benchmark.py",
        "tests/etude/test_study_branch.py",
        "uv.lock",
    ):
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"source closure path is missing: {relative}")
        paths.add(path)
    paths.add(_extension_path())
    return sorted(paths, key=lambda path: str(path.relative_to(ROOT)))


def source_identity() -> tuple[str, list[str]]:
    digest = hashlib.sha256()
    relative_paths: list[str] = []
    for path in source_paths():
        relative = str(path.relative_to(ROOT))
        contents = path.read_bytes()
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(contents)).encode("ascii"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    return digest.hexdigest(), relative_paths


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _build_completed_session(trace_dir: Path) -> tuple[GameSession, CanonicalReplayV1]:
    session = GameSession(
        trace_dir,
        id_factory=lambda kind: f"rul-6-study-{kind}",
        clock=lambda: FIXED_TIME,
        villain_offer_policy=lambda context: int(context.offers[-1]["id"]),
    )
    session.new_game(
        {
            "villain_type": "passive",
            "seed": 7,
            "hero_deck": "ur_lessons",
            "villain_deck": "gw_allies",
            "auto_pass": False,
        }
    )
    for index in range(MAX_HERO_COMMANDS):
        _require(session.obs is not None, "Study source observation disappeared")
        if session.obs.game_over:
            break
        frame = session._experience_frame()
        outcome = session.hero_command(
            {
                "command_id": f"rul-6-source-{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": frame["offers"][0]["id"],
                "answers": [],
            }
        )
        _require(outcome["status"] == "accepted", "source command was rejected")
    else:
        raise RuntimeError("deterministic Study source match exceeded command limit")

    session.close("game_over")
    _require(session.trace is not None, "Study source trace is missing")
    _require(
        session.trace.canonical_replay is not None,
        "Study source canonical replay is missing",
    )
    _require(
        not any(session.authority_fallback_counters.values()),
        f"source match exercised fallback: {session.authority_fallback_counters}",
    )
    replay = CanonicalReplayV1.model_validate(session.trace.canonical_replay)
    return session, replay


def _submission(offer: dict[str, Any]) -> dict[str, Any]:
    return {
        "offer_id": offer["id"],
        "answers": [
            {
                "kind": "candidates",
                "role": choice["role"],
                "candidates": [
                    candidate["id"]
                    for candidate in choice["candidates"]["initial"][
                        : int(choice["min"])
                    ]
                ],
            }
            for choice in offer["choices"]
        ],
    }


def _first_cast_workload(
    session: GameSession, replay: CanonicalReplayV1
) -> tuple[Any, str, dict[str, Any], dict[str, Any], StudyReturnReceipt]:
    for row in replay.decisions:
        if row.viewer != 0:
            continue
        address = ReplayDecisionAddress.from_decision(replay, row).serialize()
        branch = session.fork_study(address)
        try:
            projection = branch.structured_offers()
        except StudyBranchUnavailableError:
            branch.return_to_recorded()
            continue
        cast = next(
            (offer for offer in projection["offers"] if offer["verb"] == "cast"),
            None,
        )
        returned = branch.return_to_recorded()
        if cast is not None and cast.get("source", {}).get("kind") == "object":
            return row, address, projection, cast, returned
    raise RuntimeError("fixed Study source has no native object-bound cast offer")


def _canonical_return_payload(receipt: StudyReturnReceipt) -> dict[str, Any]:
    payload = receipt.model_dump(mode="json")
    payload.pop("source_digest")
    payload.pop("execution")
    return payload


def _assert_canonical_return(
    returned: StudyReturnReceipt,
    expected: Any,
    source_digest: str,
) -> None:
    _require(returned.source_digest == source_digest, "Study source digest drifted")
    _require(returned.address == expected.address, "return address drifted")
    _require(returned.frame == expected.frame, "return frame drifted")
    _require(returned.offer == expected.offer, "return offer drifted")
    _require(returned.command == expected.command, "return command drifted")
    _require(
        returned.presentation_cursor == expected.presentation_cursor,
        "return presentation cursor drifted",
    )
    _require(
        returned.continuation == expected.continuation,
        "return continuation drifted",
    )
    _require(
        not returned.frame.projection.opponent.hand,
        "canonical return exposed opponent hand",
    )


def _assert_execution(
    returned: StudyReturnReceipt,
    *,
    published: int,
    accepted: int,
    rejected: int,
    actions: int,
) -> None:
    execution = returned.execution
    _require(execution.driver == STUDY_BRANCH_DRIVER, "Study driver drifted")
    _require(execution.command_path == STUDY_COMMAND_PATH, "command path drifted")
    _require(execution.published_offer_sets == published, "publish count drifted")
    _require(execution.accepted_commands == accepted, "accepted count drifted")
    _require(execution.rejected_commands == rejected, "rejected count drifted")
    _require(
        execution.committed_engine_actions == actions,
        "committed engine action count drifted",
    )
    _require(execution.fallback_commands == 0, "Study fallback was exercised")


def _object_refs(value: Any) -> set[tuple[int, int]]:
    refs: set[tuple[int, int]] = set()
    if isinstance(value, dict):
        if isinstance(value.get("entity"), int) and isinstance(
            value.get("incarnation"), int
        ):
            refs.add((int(value["entity"]), int(value["incarnation"])))
        for child in value.values():
            refs.update(_object_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_object_refs(child))
    return refs


def _assert_applied_observation(observation: Any, object_ref: dict[str, int]) -> None:
    _require(
        all(
            int(card.zone) != int(managym.ZoneEnum.HAND)
            for card in observation.opponent_cards
        ),
        "Study child observation exposed opponent hand",
    )
    moved = next(
        (card for card in observation.agent_cards if card.id == object_ref["entity"]),
        None,
    )
    _require(moved is not None, "bound object disappeared from child observation")
    _require(
        int(moved.zone) == int(managym.ZoneEnum.STACK),
        "bound object did not perform the expected child-only zone change",
    )


def _assert_closed(branch: StudyBranch) -> None:
    try:
        branch.structured_offers()
    except StudyBranchUnavailableError as exc:
        _require("returned to replay" in str(exc), "closed branch raised wrong error")
        return
    raise RuntimeError("returned Study branch remained usable")


def _add_execution(total: dict[str, int], receipt: StudyReturnReceipt) -> None:
    execution = receipt.execution
    _require(execution.fallback_commands == 0, "Study fallback was exercised")
    for field in (
        "published_offer_sets",
        "accepted_commands",
        "rejected_commands",
        "committed_engine_actions",
        "fallback_commands",
    ):
        total[field] += int(getattr(execution, field))


def _timed_cycle(
    session: GameSession,
    address: str,
    submission: dict[str, Any],
) -> tuple[dict[str, int], StudyBranch, dict[str, Any], Any, StudyReturnReceipt]:
    started = time.perf_counter_ns()
    branch = session.fork_study(address)
    forked = time.perf_counter_ns()
    projection = branch.structured_offers()
    published = time.perf_counter_ns()
    observation, _, _, _, _, engine_actions = branch.submit(submission)
    applied = time.perf_counter_ns()
    returned = branch.return_to_recorded()
    completed = time.perf_counter_ns()
    _require(engine_actions == 1, "structured command committed wrong action count")
    return (
        {
            "fork_ns": forked - started,
            "publish_ns": published - forked,
            "apply_ns": applied - published,
            "return_ns": completed - applied,
            "end_to_end_ns": completed - started,
        },
        branch,
        projection,
        observation,
        returned,
    )


def _record_failure(
    identifier: str,
    call: Callable[[], Any],
    expected: tuple[type[Exception], ...],
) -> dict[str, Any]:
    try:
        call()
    except expected as exc:
        return {
            "id": identifier,
            "passed": True,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    except Exception as exc:
        raise RuntimeError(
            f"{identifier} raised untyped {type(exc).__name__}: {exc}"
        ) from exc
    raise RuntimeError(f"{identifier} unexpectedly succeeded")


def _branch_digest(branch: StudyBranch) -> str:
    env = branch._env
    _require(env is not None, "measurement branch unexpectedly closed")
    return env.state_digest()


def _failure_evidence(
    session: GameSession,
    replay: CanonicalReplayV1,
    row: Any,
    address: str,
    cast: dict[str, Any],
    expected: Any,
    source_digest: str,
    execution_total: dict[str, int],
) -> tuple[list[dict[str, Any]], int]:
    cases: list[dict[str, Any]] = []

    branch = session.fork_study(address)
    before = _branch_digest(branch)
    case = _record_failure(
        "submit_before_publish",
        lambda: branch.submit(_submission(cast)),
        (StudyBranchUnavailableError,),
    )
    case["branch_unchanged"] = _branch_digest(branch) == before
    returned = branch.return_to_recorded()
    _assert_canonical_return(returned, expected, source_digest)
    _assert_execution(returned, published=0, accepted=0, rejected=1, actions=0)
    _add_execution(execution_total, returned)
    cases.append(case)

    branch = session.fork_study(address)
    branch.structured_offers()
    before = _branch_digest(branch)
    case = _record_failure(
        "unknown_offer_consumes_offer_set",
        lambda: branch.submit({"offer_id": 2**31 - 1, "answers": []}),
        (StudyBranchUnavailableError,),
    )
    case["branch_unchanged"] = _branch_digest(branch) == before
    consumed = _record_failure(
        "offer_set_reuse_probe",
        lambda: branch.submit(_submission(cast)),
        (StudyBranchUnavailableError,),
    )
    case["offer_set_consumed"] = consumed["passed"] and "Publish" in consumed["message"]
    returned = branch.return_to_recorded()
    _assert_canonical_return(returned, expected, source_digest)
    _assert_execution(returned, published=1, accepted=0, rejected=2, actions=0)
    _add_execution(execution_total, returned)
    cases.append(case)

    unsupported = next(
        candidate
        for candidate in replay.decisions
        if candidate.viewer == 0 and candidate.frame.action_space == "DISCARD_THEN_DRAW"
    )
    unsupported_address = ReplayDecisionAddress.from_decision(
        replay, unsupported
    ).serialize()
    branch = session.fork_study(unsupported_address)
    case = _record_failure(
        "unsupported_native_surface",
        branch.structured_offers,
        (StudyBranchUnavailableError,),
    )
    returned = branch.return_to_recorded()
    _assert_execution(returned, published=0, accepted=0, rejected=0, actions=0)
    _add_execution(execution_total, returned)
    cases.append(case)

    cases.append(
        _record_failure(
            "invalid_address",
            lambda: session.fork_study("erd1.invalid!"),
            (InvalidAddressError,),
        )
    )
    missing = ReplayDecisionAddress.from_decision(replay, row).model_copy(
        update={"ordinal": len(replay.decisions) + 1}
    )
    cases.append(
        _record_failure(
            "missing_address",
            lambda: session.fork_study(missing.serialize()),
            (DecisionNotFoundError,),
        )
    )
    other = next(candidate for candidate in replay.decisions if candidate.viewer == 1)
    other_address = ReplayDecisionAddress.from_decision(replay, other).serialize()
    cases.append(
        _record_failure(
            "other_viewer",
            lambda: session.fork_study(other_address),
            (DecisionNotFoundError,),
        )
    )

    _require(session._study_provider is not None, "Study provider disappeared")
    retained, _ = session._study_provider._roots[int(row.ordinal)]
    isolated_root = retained.clone_env()
    provider = StudyForkProvider(replay, {int(row.ordinal): isolated_root})
    branch = provider.fork(address, authorized_viewer=0)
    isolated_root.step(0)
    case = _record_failure(
        "retained_root_drift_return",
        branch.return_to_recorded,
        (StudyBranchUnavailableError,),
    )
    consumed = _record_failure(
        "drifted_return_reuse_probe",
        branch.structured_offers,
        (StudyBranchUnavailableError,),
    )
    case["branch_consumed"] = (
        consumed["passed"] and "returned to replay" in consumed["message"]
    )
    cases.append(case)
    cases.append(
        _record_failure(
            "retained_root_drift_fork",
            lambda: provider.fork(address, authorized_viewer=0),
            (StudyBranchUnavailableError,),
        )
    )

    branch = session.fork_study(address)
    projection = branch.structured_offers()
    projected_cast = next(
        offer for offer in projection["offers"] if offer["verb"] == "cast"
    )
    projected_cast["source"]["id"]["incarnation"] += 1_000
    observation, _, _, _, _, _ = branch.submit(_submission(projected_cast))
    _assert_applied_observation(observation, cast["source"]["id"])
    returned = branch.return_to_recorded()
    _assert_canonical_return(returned, expected, source_digest)
    _assert_execution(returned, published=1, accepted=1, rejected=0, actions=1)
    _add_execution(execution_total, returned)
    cases.append(
        {
            "id": "projected_incarnation_tamper",
            "passed": True,
            "exception_type": None,
            "message": "authority-held native offer remained binding",
            "bound_entity": cast["source"]["id"]["entity"],
            "projected_incarnation_delta": 1_000,
        }
    )

    for case in cases:
        _require(case["passed"], f"failure case did not pass: {case['id']}")
        if "branch_unchanged" in case:
            _require(case["branch_unchanged"], f"rejection mutated {case['id']}")
        if "offer_set_consumed" in case:
            _require(case["offer_set_consumed"], "rejected offer set was reusable")
        if "branch_consumed" in case:
            _require(case["branch_consumed"], "drifted return branch was reusable")
    return cases, 1


def run_measurement(
    *,
    iterations: int = CANONICAL_ITERATIONS,
    retained_count: int = CANONICAL_RETAINED,
    warmup: int = CANONICAL_WARMUP,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    if iterations < 1 or retained_count < 2 or warmup < 0:
        raise ValueError("iterations >= 1, retained >= 2, and warmup >= 0 required")
    started_at = utc_now()
    initial_source, initial_paths = source_identity()
    process = psutil.Process()
    canonical = (
        iterations == CANONICAL_ITERATIONS
        and retained_count == CANONICAL_RETAINED
        and warmup == CANONICAL_WARMUP
    )

    with tempfile.TemporaryDirectory(prefix="etude-study-branch-") as temporary:
        session, replay = _build_completed_session(Path(temporary))
        row, address, expected_projection, cast, baseline_return = _first_cast_workload(
            session, replay
        )
        expected = restore_decision(replay, address, authorized_viewer=0)
        source_digest = baseline_return.source_digest
        object_ref = dict(cast["source"]["id"])
        _require(object_ref["incarnation"] >= 0, "negative object incarnation")
        _require(
            (object_ref["entity"], object_ref["incarnation"])
            in _object_refs(expected_projection),
            "cast object reference is absent from source projection",
        )
        submission = _submission(cast)

        replay_before = canonical_json(session.trace.canonical_replay)
        events_before = canonical_json(
            [asdict(event) for event in session.trace.events]
        )
        source_fallback_before = dict(session.authority_fallback_counters)
        execution_total = {
            "published_offer_sets": 0,
            "accepted_commands": 0,
            "rejected_commands": 0,
            "committed_engine_actions": 0,
            "fallback_commands": 0,
        }

        for _ in range(warmup):
            _, branch, projection, observation, returned = _timed_cycle(
                session, address, submission
            )
            _require(
                projection == expected_projection, "warmup offer projection drifted"
            )
            _assert_applied_observation(observation, object_ref)
            _assert_canonical_return(returned, expected, source_digest)
            _assert_execution(returned, published=1, accepted=1, rejected=0, actions=1)
            _assert_closed(branch)

        samples = {
            "fork_ns": [],
            "publish_ns": [],
            "apply_ns": [],
            "return_ns": [],
            "end_to_end_ns": [],
        }
        logical_checksums: set[str] = set()
        wall_started = time.perf_counter_ns()
        for _ in range(iterations):
            timings, branch, projection, observation, returned = _timed_cycle(
                session, address, submission
            )
            for field, value in timings.items():
                samples[field].append(value)
            _require(
                projection == expected_projection, "sequential offer projection drifted"
            )
            _require(
                (object_ref["entity"], object_ref["incarnation"])
                in _object_refs(projection),
                "sequential object incarnation drifted",
            )
            _assert_applied_observation(observation, object_ref)
            _assert_canonical_return(returned, expected, source_digest)
            _assert_execution(returned, published=1, accepted=1, rejected=0, actions=1)
            _add_execution(execution_total, returned)
            _assert_closed(branch)
            logical_checksums.add(
                sha256_bytes(
                    canonical_json(
                        {
                            "source_digest": returned.source_digest,
                            "return": _canonical_return_payload(returned),
                            "object": object_ref,
                            "object_zone": int(
                                next(
                                    card
                                    for card in observation.agent_cards
                                    if card.id == object_ref["entity"]
                                ).zone
                            ),
                        }
                    )
                )
            )
        wall_elapsed_ns = time.perf_counter_ns() - wall_started
        _require(len(logical_checksums) == 1, "sequential logical checksum drifted")

        gc.collect()
        rss_samples = [{"phase": "baseline", "rss_bytes": process.memory_info().rss}]
        branches = [session.fork_study(address) for _ in range(retained_count)]
        rss_samples.append({"phase": "forked", "rss_bytes": process.memory_info().rss})
        retained_projections = [branch.structured_offers() for branch in branches]
        _require(
            all(
                projection == expected_projection for projection in retained_projections
            ),
            "retained sibling offer projection drifted",
        )
        rss_samples.append(
            {"phase": "offers_published", "rss_bytes": process.memory_info().rss}
        )
        applied_observations: dict[int, Any] = {}
        for index in range(0, retained_count, 2):
            observation, _, _, _, _, actions = branches[index].submit(submission)
            _require(actions == 1, "retained sibling committed wrong action count")
            _assert_applied_observation(observation, object_ref)
            applied_observations[index] = observation
        rss_samples.append(
            {"phase": "alternating_applied", "rss_bytes": process.memory_info().rss}
        )
        for index in range(1, retained_count, 2):
            _require(
                branches[index].structured_offers() == expected_projection,
                "untouched sibling changed after adjacent apply",
            )
        retained_returns: list[StudyReturnReceipt] = []
        for index, branch in enumerate(branches):
            returned = branch.return_to_recorded()
            _assert_canonical_return(returned, expected, source_digest)
            if index % 2 == 0:
                _assert_execution(
                    returned, published=1, accepted=1, rejected=0, actions=1
                )
            else:
                _assert_execution(
                    returned, published=2, accepted=0, rejected=0, actions=0
                )
            _add_execution(execution_total, returned)
            _assert_closed(branch)
            retained_returns.append(returned)
        rss_samples.append(
            {"phase": "returned", "rss_bytes": process.memory_info().rss}
        )
        branches.clear()
        retained_projections.clear()
        applied_observations.clear()
        retained_returns.clear()
        gc.collect()
        rss_samples.append(
            {"phase": "collected", "rss_bytes": process.memory_info().rss}
        )
        rss_baseline = rss_samples[0]["rss_bytes"]
        rss_peak = max(sample["rss_bytes"] for sample in rss_samples)
        retained_rss_delta = max(0, rss_peak - rss_baseline)

        failures, projection_tamper_checks = _failure_evidence(
            session,
            replay,
            row,
            address,
            cast,
            expected,
            source_digest,
            execution_total,
        )

        fresh_return = session.fork_study(address).return_to_recorded()
        _assert_canonical_return(fresh_return, expected, source_digest)
        _require(
            fresh_return.execution.fallback_commands == 0,
            "fresh return exercised fallback",
        )
        replay_after = canonical_json(session.trace.canonical_replay)
        events_after = canonical_json([asdict(event) for event in session.trace.events])
        source_fallback_after = dict(session.authority_fallback_counters)
        _require(
            replay_after == replay_before, "Study workload mutated canonical replay"
        )
        _require(events_after == events_before, "Study workload mutated trace events")
        _require(
            source_fallback_after == source_fallback_before
            and not any(source_fallback_after.values()),
            "Study workload changed source fallback counters",
        )

    completed_source, completed_paths = source_identity()
    _require(
        (completed_source, completed_paths) == (initial_source, initial_paths),
        "source closure changed during Study measurement",
    )
    extension = _extension_path()
    summary = {field: summarize_latencies(values) for field, values in samples.items()}
    cycles_per_second = iterations * 1_000_000_000 / wall_elapsed_ns
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "contract": {
            "id": CONTRACT_ID,
            "sha256": sha256_file(CONTRACT_PATH),
        },
        "run": {
            "started_at": started_at,
            "completed_at": utc_now(),
            "argv": list(sys.argv if argv is None else argv),
            "cwd": str(ROOT),
            "status": "complete",
            "canonical": canonical,
            "source_sha256": initial_source,
            "source_paths": initial_paths,
            "dimensions": {
                "iterations": iterations,
                "retained_siblings": retained_count,
                "warmup_cycles": warmup,
            },
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_logical": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "memory_bytes": psutil.virtual_memory().total,
            "managym_extension": {
                "path": str(extension.relative_to(ROOT)),
                "bytes": extension.stat().st_size,
                "sha256": sha256_file(extension),
            },
        },
        "workload": {
            "matchup": "ur_lessons-vs-gw_allies",
            "seed": 7,
            "hero_policy": "first-etude-offer-v1",
            "villain_policy": "last-etude-offer-v1",
            "replay_id": replay.replay_id,
            "match_id": replay.match_id,
            "decision_ordinal": int(row.ordinal),
            "address": address,
            "source_digest": source_digest,
            "source_object_ref": object_ref,
            "offer_projection_sha256": sha256_bytes(
                canonical_json(expected_projection)
            ),
            "canonical_return_sha256": sha256_bytes(
                canonical_json(_canonical_return_payload(baseline_return))
            ),
            "canonical_replay_sha256": sha256_bytes(replay_before),
            "trace_events_sha256": sha256_bytes(events_before),
            "logical_cycle_sha256": next(iter(logical_checksums)),
        },
        "performance": {
            "gates": dict(GATES),
            "sequential": {
                "iterations": iterations,
                "wall_elapsed_ns": wall_elapsed_ns,
                "cycles_per_second": cycles_per_second,
                "samples": samples,
                "summary": summary,
            },
            "retained_siblings": {
                "count": retained_count,
                "applied_count": len(range(0, retained_count, 2)),
                "untouched_count": len(range(1, retained_count, 2)),
                "rss_samples": rss_samples,
                "rss_baseline_bytes": rss_baseline,
                "rss_peak_bytes": rss_peak,
                "rss_peak_delta_bytes": retained_rss_delta,
            },
        },
        "exactness": {
            "sequential_checks": iterations,
            "retained_return_checks": retained_count,
            "fresh_return_checks": 1,
            "source_digest_mismatches": 0,
            "canonical_return_mismatches": 0,
            "offer_projection_mismatches": 0,
            "sibling_isolation_mismatches": 0,
            "canonical_replay_mutations": 0,
            "trace_event_mutations": 0,
            "closed_branch_failures": 0,
        },
        "privacy": {
            "applied_observation_checks": iterations
            + len(range(0, retained_count, 2))
            + 1,
            "canonical_return_checks": iterations + retained_count + 1,
            "opponent_hand_exposures": 0,
            "viewer_actor_mismatches": 0,
        },
        "incarnation": {
            "source_object_ref": object_ref,
            "projection_reference_checks": iterations + retained_count,
            "child_zone_change_checks": iterations
            + len(range(0, retained_count, 2))
            + projection_tamper_checks,
            "projection_tamper_checks": projection_tamper_checks,
            "object_ref_mismatches": 0,
            "child_zone_change_failures": 0,
            "projection_tamper_binding_failures": 0,
        },
        "failures": {
            "cases": failures,
            "typed_failure_mismatches": 0,
            "rejected_branch_mutations": 0,
            "fallback_recoveries": 0,
        },
        "execution": {
            "driver": STUDY_BRANCH_DRIVER,
            "command_path": STUDY_COMMAND_PATH,
            "source_fallback_counters_before": source_fallback_before,
            "source_fallback_counters_after": source_fallback_after,
            "study_totals": execution_total,
        },
    }
    payload["artifact_sha256"] = artifact_hash(payload)
    return payload


def _zero_fields(value: dict[str, Any], fields: tuple[str, ...], context: str) -> None:
    for field in fields:
        if value.get(field) != 0:
            raise RuntimeError(f"nonzero {context}.{field}: {value.get(field)!r}")


def verify(
    payload: Any,
    *,
    check_source: bool,
    require_canonical: bool,
    enforce_gates: bool,
) -> None:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise RuntimeError("Study artifact schema mismatch")
    if payload.get("artifact_sha256") != artifact_hash(payload):
        raise RuntimeError("Study artifact SHA-256 mismatch")
    contract = payload.get("contract", {})
    if contract != {"id": CONTRACT_ID, "sha256": sha256_file(CONTRACT_PATH)}:
        raise RuntimeError("Study contract identity mismatch")
    run = payload.get("run", {})
    if run.get("status") != "complete":
        raise RuntimeError("Study measurement is incomplete")
    dimensions = run.get("dimensions", {})
    canonical_dimensions = {
        "iterations": CANONICAL_ITERATIONS,
        "retained_siblings": CANONICAL_RETAINED,
        "warmup_cycles": CANONICAL_WARMUP,
    }
    if require_canonical and (
        run.get("canonical") is not True or dimensions != canonical_dimensions
    ):
        raise RuntimeError("Study artifact is not the canonical measurement cell")
    if check_source:
        current_sha, current_paths = source_identity()
        if run.get("source_sha256") != current_sha:
            raise RuntimeError("Study artifact source closure is stale")
        if run.get("source_paths") != current_paths:
            raise RuntimeError("Study artifact source path closure is stale")

    environment = payload.get("environment", {})
    extension = environment.get("managym_extension", {})
    extension_path = ROOT / str(extension.get("path", ""))
    if (
        not extension_path.is_file()
        or extension.get("bytes") != extension_path.stat().st_size
        or extension.get("sha256") != sha256_file(extension_path)
    ):
        raise RuntimeError("compiled managym extension identity mismatch")

    performance = payload.get("performance", {})
    if performance.get("gates") != GATES:
        raise RuntimeError("Study performance gates drifted")
    sequential = performance.get("sequential", {})
    iterations = dimensions.get("iterations")
    if sequential.get("iterations") != iterations:
        raise RuntimeError("Study sequential iteration count mismatch")
    samples = sequential.get("samples", {})
    summaries = sequential.get("summary", {})
    for field in (
        "fork_ns",
        "publish_ns",
        "apply_ns",
        "return_ns",
        "end_to_end_ns",
    ):
        values = samples.get(field)
        if not isinstance(values, list) or len(values) != iterations:
            raise RuntimeError(f"missing raw Study latency samples for {field}")
        if summaries.get(field) != summarize_latencies(values):
            raise RuntimeError(f"Study latency summary mismatch for {field}")
    wall_elapsed_ns = sequential.get("wall_elapsed_ns")
    if not isinstance(wall_elapsed_ns, int) or wall_elapsed_ns <= 0:
        raise RuntimeError("invalid Study wall elapsed time")
    expected_rate = iterations * 1_000_000_000 / wall_elapsed_ns
    if not math.isclose(
        float(sequential.get("cycles_per_second", -1.0)),
        expected_rate,
        rel_tol=1e-12,
    ):
        raise RuntimeError("Study throughput summary mismatch")

    retained = performance.get("retained_siblings", {})
    if retained.get("count") != dimensions.get("retained_siblings"):
        raise RuntimeError("retained Study sibling count mismatch")
    rss_samples = retained.get("rss_samples")
    if not isinstance(rss_samples, list) or [
        sample.get("phase") for sample in rss_samples
    ] != [
        "baseline",
        "forked",
        "offers_published",
        "alternating_applied",
        "returned",
        "collected",
    ]:
        raise RuntimeError("retained Study RSS phases are incomplete")
    rss_values = [sample.get("rss_bytes") for sample in rss_samples]
    if any(not isinstance(value, int) or value <= 0 for value in rss_values):
        raise RuntimeError("invalid retained Study RSS sample")
    baseline = rss_values[0]
    peak = max(rss_values)
    if (
        retained.get("rss_baseline_bytes") != baseline
        or retained.get("rss_peak_bytes") != peak
        or retained.get("rss_peak_delta_bytes") != max(0, peak - baseline)
    ):
        raise RuntimeError("retained Study RSS summary mismatch")

    if enforce_gates:
        if summaries["fork_ns"]["p95_ns"] > GATES["fork_p95_ns_max"]:
            raise RuntimeError("Study fork p95 gate failed")
        if summaries["apply_ns"]["p95_ns"] > GATES["apply_p95_ns_max"]:
            raise RuntimeError("Study apply p95 gate failed")
        if summaries["return_ns"]["p95_ns"] > GATES["return_p95_ns_max"]:
            raise RuntimeError("Study return p95 gate failed")
        if summaries["end_to_end_ns"]["p95_ns"] > GATES["end_to_end_p95_ns_max"]:
            raise RuntimeError("Study end-to-end p95 gate failed")
        if expected_rate < GATES["cycles_per_second_min"]:
            raise RuntimeError("Study throughput gate failed")
        if retained["rss_peak_delta_bytes"] > GATES["retained_rss_delta_bytes_max"]:
            raise RuntimeError("retained Study RSS gate failed")

    exactness = payload.get("exactness", {})
    _zero_fields(
        exactness,
        (
            "source_digest_mismatches",
            "canonical_return_mismatches",
            "offer_projection_mismatches",
            "sibling_isolation_mismatches",
            "canonical_replay_mutations",
            "trace_event_mutations",
            "closed_branch_failures",
        ),
        "exactness",
    )
    privacy = payload.get("privacy", {})
    _zero_fields(
        privacy,
        ("opponent_hand_exposures", "viewer_actor_mismatches"),
        "privacy",
    )
    if privacy.get("applied_observation_checks", 0) < iterations:
        raise RuntimeError("insufficient Study privacy checks")
    incarnation = payload.get("incarnation", {})
    source_ref = incarnation.get("source_object_ref", {})
    if not isinstance(source_ref.get("entity"), int) or not isinstance(
        source_ref.get("incarnation"), int
    ):
        raise RuntimeError("missing Study object-incarnation witness")
    if source_ref["incarnation"] < 0:
        raise RuntimeError("negative Study object incarnation")
    if incarnation.get("projection_tamper_checks", 0) < 1:
        raise RuntimeError("missing projected-incarnation tamper check")
    if incarnation.get("child_zone_change_checks", 0) < iterations:
        raise RuntimeError("insufficient child-only zone-change checks")
    _zero_fields(
        incarnation,
        (
            "object_ref_mismatches",
            "child_zone_change_failures",
            "projection_tamper_binding_failures",
        ),
        "incarnation",
    )

    failures = payload.get("failures", {})
    cases = failures.get("cases")
    if (
        not isinstance(cases, list)
        or {case.get("id") for case in cases} != EXPECTED_FAILURE_CASES
    ):
        raise RuntimeError("Study failure case set is incomplete")
    cases_by_id = {case["id"]: case for case in cases}
    for case in cases:
        if case.get("passed") is not True:
            raise RuntimeError(f"Study failure case did not pass: {case.get('id')}")
        if case.get("exception_type") != EXPECTED_FAILURE_TYPES[case["id"]]:
            raise RuntimeError(f"Study failure was not typed: {case['id']}")
    if cases_by_id["submit_before_publish"].get("branch_unchanged") is not True:
        raise RuntimeError("submit-before-publish mutated its Study branch")
    unknown = cases_by_id["unknown_offer_consumes_offer_set"]
    if unknown.get("branch_unchanged") is not True:
        raise RuntimeError("unknown Study offer mutated its branch")
    if unknown.get("offer_set_consumed") is not True:
        raise RuntimeError("Study rejected offer set remained reusable")
    drifted_return = cases_by_id["retained_root_drift_return"]
    if drifted_return.get("branch_consumed") is not True:
        raise RuntimeError("Study drifted return branch remained reusable")
    tamper = cases_by_id["projected_incarnation_tamper"]
    if (
        tamper.get("bound_entity") != source_ref["entity"]
        or tamper.get("projected_incarnation_delta") != 1_000
    ):
        raise RuntimeError("projected Study incarnation tamper evidence drifted")
    _zero_fields(
        failures,
        (
            "typed_failure_mismatches",
            "rejected_branch_mutations",
            "fallback_recoveries",
        ),
        "failures",
    )

    execution = payload.get("execution", {})
    if execution.get("driver") != STUDY_BRANCH_DRIVER:
        raise RuntimeError("Study execution driver mismatch")
    if execution.get("command_path") != STUDY_COMMAND_PATH:
        raise RuntimeError("Study execution command path mismatch")
    for field in (
        "source_fallback_counters_before",
        "source_fallback_counters_after",
    ):
        counters = execution.get(field)
        if not isinstance(counters, dict) or any(counters.values()):
            raise RuntimeError(f"nonzero Study {field}")
    totals = execution.get("study_totals", {})
    if totals.get("accepted_commands", 0) < iterations:
        raise RuntimeError("insufficient accepted Study command receipts")
    if totals.get("fallback_commands") != 0:
        raise RuntimeError("Study execution fallback was exercised")


def _milliseconds(value: int) -> str:
    return f"{value / 1_000_000:.3f}"


def render_report(payload: dict[str, Any]) -> str:
    sequential = payload["performance"]["sequential"]
    summary = sequential["summary"]
    retained = payload["performance"]["retained_siblings"]
    environment = payload["environment"]
    workload = payload["workload"]
    lines = [
        "# Production Study fork/apply/return evidence",
        "",
        f"Contract: `{payload['contract']['id']}` (`{payload['contract']['sha256']}`)",
        f"Driver: `{payload['execution']['driver']}`",
        f"Command path: `{payload['execution']['command_path']}`",
        f"Run: `{payload['run']['started_at']}`; canonical: `{str(payload['run']['canonical']).lower()}`",
        "",
        "## Interactive lifecycle",
        "",
        "| Phase | p50 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, field in (
        ("Fork", "fork_ns"),
        ("Publish offers", "publish_ns"),
        ("Structured apply", "apply_ns"),
        ("Return", "return_ns"),
        ("End to end", "end_to_end_ns"),
    ):
        row = summary[field]
        lines.append(
            f"| {label} | {_milliseconds(row['p50_ns'])} ms | "
            f"{_milliseconds(row['p95_ns'])} ms | {_milliseconds(row['p99_ns'])} ms | "
            f"{_milliseconds(row['max_ns'])} ms |"
        )
    lines.extend(
        [
            "",
            f"Sustained rate: **{sequential['cycles_per_second']:.1f} cycles/s** "
            f"across {sequential['iterations']} checked cycles.",
            "",
            "## Retained siblings",
            "",
            f"Retained {retained['count']} simultaneous production branches; applied "
            f"{retained['applied_count']} and re-checked {retained['untouched_count']} untouched siblings.",
            f"Peak RSS delta: **{retained['rss_peak_delta_bytes'] / (1024 * 1024):.1f} MiB** "
            f"(peak {retained['rss_peak_bytes'] / (1024 * 1024):.1f} MiB).",
            "",
            "## Exactness, privacy, incarnation, and failure",
            "",
            f"- Source digest: `{workload['source_digest']}`.",
            f"- Bound object: entity `{workload['source_object_ref']['entity']}`, "
            f"incarnation `{workload['source_object_ref']['incarnation']}`.",
            f"- Child-only zone-change checks: {payload['incarnation']['child_zone_change_checks']}; "
            "object-ref mismatches: 0.",
            f"- Viewer-private observation checks: {payload['privacy']['applied_observation_checks']}; "
            "opponent-hand exposures: 0.",
            f"- Typed failure cases: {len(payload['failures']['cases'])}; rejected-branch mutations: 0.",
            f"- Study fallbacks: {payload['execution']['study_totals']['fallback_commands']}; "
            "source authority fallback counters remained zero.",
            "",
            "## Reproduction and identity",
            "",
            "```bash",
            "uv run scripts/bench_study_branch.py measure",
            "uv run scripts/bench_study_branch.py verify",
            "```",
            "",
            f"Host: `{environment['platform']}`; Python `{environment['python']}`; "
            f"logical CPUs: `{environment['cpu_logical']}`.",
            f"Compiled extension SHA-256: `{environment['managym_extension']['sha256']}`.",
            f"Source closure SHA-256: `{payload['run']['source_sha256']}` "
            f"over {len(payload['run']['source_paths'])} recorded paths.",
            f"Artifact SHA-256: `{payload['artifact_sha256']}`.",
            "",
            "Raw evidence retains every latency sample, all RSS phase samples, exact source and binary identity, "
            "the workload address/object/return digests, failure receipts, and zero-mismatch counters.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure = subparsers.add_parser("measure")
    measure.add_argument("--iterations", type=int, default=CANONICAL_ITERATIONS)
    measure.add_argument("--retained", type=int, default=CANONICAL_RETAINED)
    measure.add_argument("--warmup", type=int, default=CANONICAL_WARMUP)
    measure.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    measure.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.command == "measure":
        payload = run_measurement(
            iterations=arguments.iterations,
            retained_count=arguments.retained,
            warmup=arguments.warmup,
        )
        verify(
            payload,
            check_source=True,
            require_canonical=payload["run"]["canonical"],
            enforce_gates=payload["run"]["canonical"],
        )
        atomic_write(
            arguments.raw,
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        atomic_write(arguments.report, render_report(payload).encode("utf-8"))
        print(f"wrote {arguments.raw}")
        print(f"wrote {arguments.report}")
        return 0

    payload = json.loads(arguments.raw.read_text(encoding="utf-8"))
    verify(
        payload,
        check_source=True,
        require_canonical=True,
        enforce_gates=True,
    )
    print(f"verified {arguments.raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
