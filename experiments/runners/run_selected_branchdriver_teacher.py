#!/usr/bin/env python3
"""Run and verify the RUL-2 real-search BranchDriver contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

from manabot.sim.selected_branchdriver_teacher import (
    EXPERIMENT_ID,
    SelectedTeacherError,
    execute_contract,
    replay_failure,
    verify_receipt,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _summary(receipt: dict[str, Any]) -> dict[str, Any]:
    cells = {}
    for cell_name, measurements in receipt["measurements"].items():
        cells[cell_name] = {
            driver: {
                key: measurement["derived"][key]
                for key in (
                    "decisions_per_second",
                    "traversals_per_second",
                    "p50_decision_ms",
                    "p95_decision_ms",
                    "peak_rss_bytes",
                    "cpu_ms_per_label",
                    "cap_rate",
                    "counters",
                )
            }
            for driver, measurement in measurements.items()
        }
    return {
        "experiment": receipt["experiment"],
        "driver": receipt["runtime"]["drivers"]["selected"],
        "pack_key": receipt["runtime"]["pack_manifest"].get("pack_key")
        or receipt["runtime"]["pack_manifest"]["compiled_semantics"]["pack_key"],
        "exactness_mismatches": len(receipt["exactness_mismatches"]),
        "cells": cells,
        "verdict": receipt["verdict"],
        "artifact_sha256": receipt["artifact_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--replay-failure", type=Path)
    args = parser.parse_args()

    try:
        if args.replay_failure is not None:
            replay = replay_failure(_load(args.replay_failure))
            print(json.dumps(asdict(replay), indent=2, sort_keys=True))
            return 0 if replay.reproduced else 1
        if args.contract is None or args.out is None:
            parser.error("--contract and --out are required")
        contract = _load(args.contract)
        if contract.get("experiment") != EXPERIMENT_ID:
            raise SelectedTeacherError("unexpected RUL-2 contract identity")
        if args.verify:
            verified = verify_receipt(contract, _load(args.out))
            print(json.dumps(verified, indent=2, sort_keys=True))
            return 0

        receipt = execute_contract(contract)
        _write(args.out, receipt)
        if receipt["exactness_mismatches"]:
            exactness = contract["exactness"]
            capsule = {
                "schema_version": 1,
                "experiment": EXPERIMENT_ID,
                "mismatch": receipt["exactness_mismatches"][0],
                "workload": {
                    "deal_seed": int(exactness["deal_seeds"][0]),
                    "ur_seat": 0,
                    "simulations": int(exactness["simulations"]),
                    "worlds": int(exactness["worlds"]),
                    "max_steps": int(exactness["max_steps"]),
                    "max_decisions": int(exactness["max_decisions"]),
                },
                "replay_command": (
                    "uv run experiments/runners/"
                    "run_selected_branchdriver_teacher.py --replay-failure "
                    f"{args.out.with_suffix('.failure.json')}"
                ),
            }
            _write(args.out.with_suffix(".failure.json"), capsule)
        print(json.dumps(_summary(receipt), indent=2, sort_keys=True))
        return 0 if receipt["verdict"]["decision"] == "remain" else 1
    except SelectedTeacherError as error:
        print(f"RUL-2 evidence failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
