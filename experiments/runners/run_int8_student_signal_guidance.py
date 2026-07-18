#!/usr/bin/env python3
"""Verify retained INT-4 bytes and run INT-8 through the INT-6 arena."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from experiments.runners.run_skill_arena import (
    ArenaError,
    diagnostic,
    freeze_anchors,
    load_json,
    preflight_int8_dependencies,
    verify_manifest,
    write_json,
)
from manabot.arena.int8_input import RetainedInputError, verify_retained_input

REPO_ROOT = Path(__file__).resolve().parents[2]
ARENA_CONTRACT = REPO_ROOT / "experiments/contracts/int-6-skill-arena-v1.json"
EXPERIMENT_CONTRACT = (
    REPO_ROOT / "experiments/contracts/int-8-student-signal-guidance-v1.json"
)
UNIFORM_CANDIDATE = (
    REPO_ROOT / "experiments/candidates/int-8-uniform-prior-puct-32-v1.json"
)
CHOSEN_CANDIDATE = (
    REPO_ROOT / "experiments/candidates/int-8-chosen-policy-prior-puct-32-v1.json"
)
VISIT_CANDIDATE = (
    REPO_ROOT / "experiments/candidates/int-8-visit-policy-prior-puct-32-v1.json"
)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    out_dir = args.out_dir.resolve()
    anchor_dir = out_dir.parent / f"{out_dir.name}-anchors"
    compatibility_path = out_dir.parent / f"{out_dir.name}-input-compatibility.json"
    failure_path = out_dir.parent / f"{out_dir.name}-input-failure.json"
    if out_dir.exists() or anchor_dir.exists() or compatibility_path.exists():
        raise ArenaError("INT-8 output, anchor, or compatibility path already exists")
    compatibility = verify_retained_input(
        args.input_manifest, failure_receipt=failure_path
    )
    compatibility_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(compatibility_path, compatibility)
    payload = Path(args.input_manifest).resolve().parent / "payload/training"
    chosen_checkpoint = payload / "chosen_policy_only-seed-197-9004b87e2be4a893.pt"
    visit_checkpoint = payload / "visit_policy_only-seed-197-c2c8dcec02dbcf19.pt"
    common_args = argparse.Namespace(
        contract=ARENA_CONTRACT,
        experiment_contract=EXPERIMENT_CONTRACT,
        input_compatibility=compatibility_path,
        uniform_candidate=UNIFORM_CANDIDATE,
        chosen_candidate=CHOSEN_CANDIDATE,
        chosen_checkpoint=chosen_checkpoint,
        visit_candidate=VISIT_CANDIDATE,
        visit_checkpoint=visit_checkpoint,
    )
    freeze_anchors(
        argparse.Namespace(
            **vars(common_args),
            out_dir=anchor_dir,
            profile="smoke",
        )
    )
    diagnostic_args = argparse.Namespace(
        **vars(common_args),
        anchor_artifact=anchor_dir / "manifest.json",
        out_dir=out_dir,
        profile="smoke",
        verify=False,
        task_started=started,
    )
    diagnostic(diagnostic_args)
    diagnostic_args.verify = True
    diagnostic_args.chosen_checkpoint = (
        out_dir / "checkpoints/chosen-policy-prior-puct-32-v1.pt"
    )
    diagnostic_args.visit_checkpoint = (
        out_dir / "checkpoints/visit-policy-prior-puct-32-v1.pt"
    )
    diagnostic(diagnostic_args)
    (
        arena_contract,
        arena_contract_sha,
        runtime,
        _,
        experiment_contract_sha,
        int8_authority,
        _,
        _,
    ) = preflight_int8_dependencies(diagnostic_args, validate_checkpoint_load=False)
    verification = verify_manifest(
        out_dir,
        arena_contract,
        arena_contract_sha,
        expected_runtime=runtime,
        experiment_contract_sha256=experiment_contract_sha,
        int8_authority=int8_authority,
    )
    decision = load_json(out_dir / "decision.json")
    return {
        "state": "complete",
        "artifact": str(out_dir / "manifest.json"),
        "manifest_sha256": verification["manifest_sha256"],
        "evidence_class": decision["evidence_class"],
        "decision": decision["decision"],
        "promotion_eligible": False,
        "admission_eligible": False,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        print(json.dumps(run(args), sort_keys=True))
    except (ArenaError, RetainedInputError, ValueError, FileNotFoundError) as error:
        print(
            json.dumps(
                {
                    "state": "failed",
                    "error": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
