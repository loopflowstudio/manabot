#!/usr/bin/env python3
"""Measure and verify the RUL-13 prepared possible-world provider."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping

from experiments.runners.run_belief_calibration import (
    _load_inputs,
    _semantic_command,
    int17_runtime_fingerprints,
    load_contract,
)
from manabot.sim.teacher1_evidence import REPO_ROOT, file_sha256, source_bundle_sha256
import managym
from managym.decision import DecisionFrame, apply_semantic_command
from managym.possible_worlds import PossibleWorldSpace

V1_CONTRACT = REPO_ROOT / "experiments/contracts/int-17-belief-calibration-v1.json"

EXPERIMENT = "rul-13-prepared-possible-world-materializer-v1"
DEFAULT_OUT = (
    REPO_ROOT / "experiments/data/rul-13-prepared-possible-world-materializer-v1.json"
)
SOURCE_ORDINAL = 3
MAX_BATCH_SIZE = 256
COUNTERFACTUAL_SEED = 907
EXPECTED_SUPPORT_SIZE = 121_485
IDENTITY_STREAM_SHA256 = (
    "8d9f46fa86742f323f915ba69bc0007e225415e8fc5504792eac4b49ebae66b6"
)
V1_FAILURE = (
    REPO_ROOT
    / "experiments/data/int-17-belief-calibration-v1/sha256"
    / "78bde491e16957b743a59cebe6f87fd519dc982793d5d6f7dbb649a98d57e027"
    / "failure.json"
)


class Rul13Error(RuntimeError):
    """The prepared provider or its retained receipt failed closed."""


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def provider_runtime() -> dict[str, Any]:
    runtime = int17_runtime_fingerprints(COUNTERFACTUAL_SEED)
    runtime["provider_source_sha256"] = source_bundle_sha256(
        [
            Path(__file__).resolve(),
            REPO_ROOT / "managym/Cargo.toml",
            REPO_ROOT / "managym/__init__.pyi",
            REPO_ROOT / "managym/tests/prepared_possible_worlds_tests.rs",
            REPO_ROOT / "tests/belief/test_prepared_likelihood.py",
            REPO_ROOT / "tests/experiments/test_rul13_prepared_materializer.py",
            REPO_ROOT / "scripts/verify-prepared-possible-world-materializer",
        ]
    )
    return runtime


def _root_at_ordinal() -> tuple[managym.Env, Mapping[str, Any], Mapping[str, Any]]:
    contract = load_contract(V1_CONTRACT)
    trace, public_provider = _load_inputs(contract)
    env = managym.Env(seed=int(contract["cohort"]["game_seed"]))
    from etude import server

    env.reset(
        [
            managym.PlayerConfig("Hero", dict(server.UR_LESSONS_DECK)),
            managym.PlayerConfig("Villain", dict(server.GW_ALLIES_DECK)),
        ]
    )
    for row in trace["decisions"][:SOURCE_ORDINAL]:
        frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
        apply_semantic_command(env, _semantic_command(row, frame))
    if env.state_digest() != trace["decisions"][SOURCE_ORDINAL]["state"]["before"]:
        raise Rul13Error("ordinal-3 source root does not match the frozen trace")
    return env, trace, public_provider


def _branch_witness(branch: managym.Env, viewer: int) -> dict[str, Any]:
    opponent = (viewer + 1) % 2
    hand = Counter(
        str(card.name)
        for card in branch.observation_for_player(opponent).agent_cards
        if int(card.zone) == 1
    )
    return {
        "state_digest": branch.state_digest(),
        "event_cursor": branch.semantic_event_cursor(),
        "acting_player": branch.current_agent_index(),
        "decision_frame": json.loads(branch.semantic_decision_frame_json()),
        "viewer_observation": json.loads(branch.semantic_observation_json(viewer)),
        "realized_hand": dict(hand),
    }


def _expect_rejection(call, fragment: str) -> str:
    try:
        call()
    except managym.AgentError as error:
        message = str(error)
        if fragment not in message:
            raise Rul13Error(
                f"expected rejection containing {fragment!r}, got {message!r}"
            ) from error
        return message
    raise Rul13Error(f"provider accepted a request expected to contain {fragment!r}")


def measure() -> dict[str, Any]:
    root, trace, public_provider = _root_at_ordinal()
    from etude.public_commitment_parity import build_receipt

    # Replay the actual live/headless/persisted boundaries with this extension.
    # Keep the previous immutable receipt as a reference, never rewrite it.
    current_public = build_receipt()
    if current_public["summary"] != public_provider["summary"]:
        raise Rul13Error("public commitment parity drifted")
    if _canonical_sha256(current_public["identity_stream"]) != IDENTITY_STREAM_SHA256:
        raise Rul13Error("current public commitment identity stream drifted")
    next_row = trace["decisions"][SOURCE_ORDINAL]
    actor = int(next_row["actor"])
    viewer = (actor + 1) % 2
    if viewer != 0:
        raise Rul13Error("the frozen ordinal-3 likelihood viewer drifted")

    reference_root = root.clone_env()
    reference_space = PossibleWorldSpace.from_engine(reference_root, viewer)
    if reference_space.support_size != EXPECTED_SUPPORT_SIZE:
        raise Rul13Error("the frozen ordinal-3 support size drifted")
    root_before = {
        "state_digest": root.state_digest(),
        "event_cursor": root.semantic_event_cursor(),
        "viewer_observation": json.loads(root.semantic_observation_json(viewer)),
    }
    baseline_rss = _peak_rss_bytes()
    prepare_started = time.perf_counter()
    prepared = root.prepare_possible_world_materializer(
        viewer, reference_space.identity, MAX_BATCH_SIZE
    )
    prepare_seconds = time.perf_counter() - prepare_started
    if prepared.construction_count != 1:
        raise Rul13Error("prepared provider did not enumerate exactly once")

    parity_indexes = [
        reference_space.support_size - 1,
        0,
        reference_space.support_size // 2,
        1,
    ]
    parity_seeds = [COUNTERFACTUAL_SEED + index for index in parity_indexes]
    scalar = [
        reference_space.materialize(
            index,
            seed=seed,
            refresh_opponent_commitment=True,
        )
        for index, seed in zip(parity_indexes, parity_seeds, strict=True)
    ]
    prepared_branches = prepared.materialize_indexes(parity_indexes, parity_seeds, True)
    parity_rows = []
    for index, expected, actual in zip(
        parity_indexes, scalar, prepared_branches, strict=True
    ):
        expected_witness = _branch_witness(expected, viewer)
        actual_witness = _branch_witness(actual, viewer)
        if actual_witness != expected_witness:
            raise Rul13Error(f"scalar/prepared parity failed at row {index}")
        if actual_witness["realized_hand"] != dict(reference_space.world(index).hand):
            raise Rul13Error(f"prepared branch realized the wrong hand at row {index}")
        parity_rows.append(
            {
                "index": index,
                "weight": reference_space.world(index).weight,
                "witness_sha256": _canonical_sha256(actual_witness),
            }
        )

    sibling_before = prepared_branches[1].state_digest()
    prepared_branches[0].step(0)
    if root.state_digest() != root_before["state_digest"]:
        raise Rul13Error("prepared sibling mutated the live root")
    if prepared_branches[1].state_digest() != sibling_before:
        raise Rul13Error("prepared sibling mutated another sibling")
    del actual, expected, prepared_branches

    repeated = prepared.materialize_indexes(
        [parity_indexes[0]], [parity_seeds[0]], True
    )[0]
    if _branch_witness(repeated, viewer) != _branch_witness(scalar[0], viewer):
        raise Rul13Error("identical index/seed materialization was nondeterministic")
    del repeated, scalar

    rejections = {
        "identity": _expect_rejection(
            lambda: root.prepare_possible_world_materializer(
                viewer, "0" * 64, MAX_BATCH_SIZE
            ),
            "space identity mismatch",
        ),
        "empty": _expect_rejection(
            lambda: prepared.materialize_indexes([], [], True), "must not be empty"
        ),
        "length_mismatch": _expect_rejection(
            lambda: prepared.materialize_indexes([0], [], True), "equal length"
        ),
        "oversize": _expect_rejection(
            lambda: prepared.materialize_indexes(
                list(range(MAX_BATCH_SIZE + 1)),
                [COUNTERFACTUAL_SEED] * (MAX_BATCH_SIZE + 1),
                True,
            ),
            "exceeds maximum",
        ),
        "index": _expect_rejection(
            lambda: prepared.materialize_indexes(
                [reference_space.support_size], [COUNTERFACTUAL_SEED], True
            ),
            "outside support size",
        ),
    }

    thresholds = [MAX_BATCH_SIZE, 4_096, 32_768, reference_space.support_size]
    scaling = []
    rss_samples = []
    completed = 0
    full_started = time.perf_counter()
    while completed < reference_space.support_size:
        stop = min(completed + MAX_BATCH_SIZE, reference_space.support_size)
        indexes = list(range(completed, stop))
        branches = prepared.materialize_indexes(
            indexes, [COUNTERFACTUAL_SEED] * len(indexes), True
        )
        if any(branch.current_agent_index() != actor for branch in branches):
            raise Rul13Error("a prepared branch did not refresh the acting commitment")
        completed = stop
        elapsed = time.perf_counter() - full_started
        while thresholds and completed >= thresholds[0]:
            threshold = thresholds.pop(0)
            scaling.append(
                {
                    "completed_rows": completed,
                    "threshold_rows": threshold,
                    "seconds": elapsed,
                    "rows_per_second": completed / elapsed,
                    "peak_rss_bytes": _peak_rss_bytes(),
                }
            )
        del branches
        rss_samples.append(
            {"completed_rows": completed, "peak_rss_bytes": _peak_rss_bytes()}
        )
    full_seconds = time.perf_counter() - full_started
    diagnostics = json.loads(prepared.diagnostics_json())
    if diagnostics["space_constructions"] != 1:
        raise Rul13Error("batch materialization reconstructed the canonical space")
    if diagnostics["maximum_returned_batch"] > MAX_BATCH_SIZE:
        raise Rul13Error("provider returned an unbounded batch")
    if root.state_digest() != root_before["state_digest"]:
        raise Rul13Error("full prepared pass mutated the live root")
    if root.semantic_event_cursor() != root_before["event_cursor"]:
        raise Rul13Error("full prepared pass moved the live event cursor")
    if (
        json.loads(root.semantic_observation_json(viewer))
        != root_before["viewer_observation"]
    ):
        raise Rul13Error("full prepared pass changed the live viewer projection")

    root.step(0)
    rejections["stale_root"] = _expect_rejection(
        lambda: prepared.materialize_indexes([0], [COUNTERFACTUAL_SEED], True),
        "source identity is stale",
    )
    diagnostics = json.loads(prepared.diagnostics_json())
    peak_rss = _peak_rss_bytes()
    if peak_rss > 2_147_483_648:
        raise Rul13Error("prepared provider exceeded the existing 2 GiB cap")

    runtime = provider_runtime()
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "measured_and_verified",
        "summary": current_public["summary"],
        "identity": current_public["identity"],
        "surfaces": current_public["surfaces"],
        "identity_stream": current_public["identity_stream"],
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        },
        "public_commitment_provider": {
            "path": str(Path(load_contract(V1_CONTRACT)["provider_receipt"]["path"])),
            "sha256": file_sha256(
                REPO_ROOT / load_contract(V1_CONTRACT)["provider_receipt"]["path"]
            ),
            "rules_provider_gaps": 0,
            "identity_stream_sha256": IDENTITY_STREAM_SHA256,
        },
        "prepared_materializer": {
            "source_ordinal": SOURCE_ORDINAL,
            "viewer": viewer,
            "actor": actor,
            "source_revision": reference_space.source_revision,
            "source_viewer_state_hash": reference_space.source_viewer_state_hash,
            "space_identity": reference_space.identity,
            "support_size": reference_space.support_size,
            "total_weight": reference_space.total_weight,
            "max_batch_size": MAX_BATCH_SIZE,
            "counterfactual_seed": COUNTERFACTUAL_SEED,
            "mode": "RefreshOpponentCommitment",
            "prepare_seconds": prepare_seconds,
            "full_pass_seconds": full_seconds,
            "full_pass_rows_per_second": reference_space.support_size / full_seconds,
            "baseline_rss_bytes": baseline_rss,
            "peak_rss_bytes": peak_rss,
            "diagnostics": diagnostics,
            "parity": parity_rows,
            "rejections": rejections,
            "scaling": scaling,
            "rss_samples": rss_samples,
            "source_witness": root_before,
            "root_and_sibling_isolation": True,
            "full_support_materialized": True,
        },
        "runtime": runtime,
        "frozen_v1": {
            "contract_path": str(V1_CONTRACT.relative_to(REPO_ROOT)),
            "contract_sha256": file_sha256(V1_CONTRACT),
            "failure_path": str(V1_FAILURE.relative_to(REPO_ROOT)),
            "failure_sha256": file_sha256(V1_FAILURE),
            "rerun": False,
        },
    }


def verify_receipt(
    receipt: Mapping[str, Any], *, check_extension: bool = False
) -> None:
    if receipt.get("schema_version") != 1 or receipt.get("experiment") != EXPERIMENT:
        raise Rul13Error("RUL-13 receipt identity drifted")
    if receipt.get("status") != "measured_and_verified":
        raise Rul13Error("RUL-13 receipt is incomplete")
    public = receipt.get("public_commitment_provider", {})
    if public.get("rules_provider_gaps") != 0:
        raise Rul13Error("RUL-13 public provider has gaps")
    if public.get("identity_stream_sha256") != IDENTITY_STREAM_SHA256:
        raise Rul13Error("RUL-13 identity stream drifted")
    if _canonical_sha256(receipt.get("identity_stream")) != IDENTITY_STREAM_SHA256:
        raise Rul13Error("RUL-13 measured identity stream drifted")
    if public.get("sha256") != file_sha256(REPO_ROOT / public["path"]):
        raise Rul13Error("RUL-13 reference provider bytes drifted")
    runtime = provider_runtime()
    portable_runtime = set(runtime) - {
        "engine_extension_sha256",
        "engine_extension_name",
    }
    for key in runtime if check_extension else portable_runtime:
        if receipt.get("runtime", {}).get(key) != runtime[key]:
            raise Rul13Error(f"RUL-13 runtime drifted: {key}")
    prepared = receipt.get("prepared_materializer", {})
    if prepared.get("support_size") != EXPECTED_SUPPORT_SIZE:
        raise Rul13Error("RUL-13 support size drifted")
    trace, _ = _load_inputs(load_contract(V1_CONTRACT))
    witness = prepared.get("source_witness", {})
    if (
        witness.get("state_digest")
        != trace["decisions"][SOURCE_ORDINAL]["state"]["before"]
    ):
        raise Rul13Error("RUL-13 source root identity drifted")
    observation_identity = witness.get("viewer_observation", {}).get("identity", {})
    if (
        observation_identity.get("revision") != prepared.get("source_revision")
        or observation_identity.get("viewer_state_hash")
        != prepared.get("source_viewer_state_hash")
        or observation_identity.get("viewer") != prepared.get("viewer")
    ):
        raise Rul13Error("RUL-13 source observation identity drifted")
    diagnostics = prepared.get("diagnostics", {})
    if diagnostics.get("space_constructions") != 1:
        raise Rul13Error("RUL-13 construction count is not one")
    if diagnostics.get("materialized_rows") != EXPECTED_SUPPORT_SIZE + 5:
        raise Rul13Error("RUL-13 materialized row count drifted")
    if diagnostics.get("requested_rows") != diagnostics["materialized_rows"]:
        raise Rul13Error("RUL-13 dropped requested rows")
    if diagnostics.get("current_live_prepared_branches") != 0:
        raise Rul13Error("RUL-13 retained branches between batches")
    if (
        diagnostics.get("maximum_live_prepared_branches", MAX_BATCH_SIZE + 1)
        > MAX_BATCH_SIZE
    ):
        raise Rul13Error("RUL-13 exceeded the live branch bound")
    if diagnostics.get("maximum_returned_batch", MAX_BATCH_SIZE + 1) > MAX_BATCH_SIZE:
        raise Rul13Error("RUL-13 maximum batch drifted")
    if not prepared.get("full_support_materialized"):
        raise Rul13Error("RUL-13 did not materialize the full maximum support")
    samples = prepared.get("rss_samples", [])
    expected_stops = list(
        range(MAX_BATCH_SIZE, EXPECTED_SUPPORT_SIZE, MAX_BATCH_SIZE)
    ) + [EXPECTED_SUPPORT_SIZE]
    if [row["completed_rows"] for row in samples] != expected_stops:
        raise Rul13Error("RUL-13 resource ledger is incomplete")
    if not prepared.get("root_and_sibling_isolation"):
        raise Rul13Error("RUL-13 isolation proof failed")
    if set(prepared.get("rejections", {})) != {
        "identity",
        "empty",
        "length_mismatch",
        "oversize",
        "index",
        "stale_root",
    }:
        raise Rul13Error("RUL-13 rejection proof is incomplete")
    if max(row["peak_rss_bytes"] for row in samples) > prepared["peak_rss_bytes"]:
        raise Rul13Error("RUL-13 peak RSS contradicts samples")
    if prepared.get("peak_rss_bytes", 2_147_483_649) > 2_147_483_648:
        raise Rul13Error("RUL-13 exceeded the existing RSS cap")
    frozen = receipt.get("frozen_v1", {})
    if frozen.get("contract_sha256") != file_sha256(V1_CONTRACT):
        raise Rul13Error("frozen INT-17 v1 contract changed")
    if frozen.get("failure_sha256") != file_sha256(V1_FAILURE):
        raise Rul13Error("frozen INT-17 v1 failure changed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.verify:
        receipt = json.loads(args.out.read_text())
        verify_receipt(receipt, check_extension=True)
        print(json.dumps({"status": "verified", "path": str(args.out)}))
        return
    receipt = measure()
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    verify_receipt(receipt)
    print(
        json.dumps(
            {
                "status": "measured_and_verified",
                "path": str(args.out),
                "support_size": EXPECTED_SUPPORT_SIZE,
                "rows_per_second": receipt["prepared_materializer"][
                    "full_pass_rows_per_second"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
