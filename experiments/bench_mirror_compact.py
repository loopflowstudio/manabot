"""Paired fixed-root compact-evidence admission; no games or campaign changes.

JSON dictionaries are retained runner/native evidence at this boundary. Untimed
instrumentation captures actual Commands and final viewer observations separately
from production receipts. No instrumentation wraps latency or profile samples.
"""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
from datetime import UTC, datetime
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import time
import traceback
from typing import Any
from unittest.mock import patch

from managym.decision import Command
import numpy as np
from pydantic import TypeAdapter
import torch

from experiments.bench_mirror_batching import compare
from experiments.search_mirror import OracleHandSampler
from experiments.search_scaling import CHECKPOINT_SHA256, PACK
from manabot.env import Env, Match, Reward
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.infra.identity import file_sha256
from manabot.sim.search.branch import _SelectedSession
from manabot.sim.search.local_update import (
    LocalSearchConfig,
    LocalUpdatePlayer,
    LocalUpdateReceipt,
    LocalUpdateTeacher,
    regularized_update,
)
from manabot.sim.search.sampling import PhysicalHandSampler
import managym

POSITIONS = (0, 100, 286)
BASELINE = Path("scratch/mirror-latency-baseline-20261110")
ADAPTER = TypeAdapter(LocalUpdateReceipt)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--inference-threads", type=positive_int, default=1)
    parser.add_argument("--modes", choices=("paired", "compact"), default="paired")
    parser.add_argument(
        "--control", type=Path, help="completed one-thread compact harness directory"
    )
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--native-encoder", action="store_true")
    parser.add_argument(
        "--source-commit", help="Verified revision of an unpacked source archive"
    )
    parser.add_argument(
        "--cross-host",
        action="store_true",
        help="Record differing host/runtime metadata; preserve every behavioral gate",
    )
    parser.add_argument(
        "--encoding-comparison",
        action="store_true",
        help="retain and explicitly compare changed encoder/runtime identities",
    )
    args = parser.parse_args(argv)
    if args.modes == "compact" and args.control is None:
        parser.error("compact-only measurement requires --control")
    if args.cross_host and not args.encoding_comparison:
        parser.error("cross-host admission requires --encoding-comparison")
    return args


def admit_control(path: Path, current: dict[str, Any]) -> dict[str, Any]:
    """Bind the thread comparison to a completed, otherwise identical control."""
    control = json.loads((path / "result.json").read_text())
    assert control["all_passed"] and not control["errors"]
    assert control["torch_threads"] == 1 or (
        current.get("encoding_comparison")
        and control["torch_threads"] == current["torch_threads"]
    )
    for key in (
        "checkpoint_sha256",
        "runtime_sha256",
        "search_source_sha256",
        "attempt_sha256",
        "world_binding",
        "config",
        "interop_threads",
        "device",
        "packages",
        "python",
    ):
        if (
            (
                current.get("cross_host")
                and key in ("interop_threads", "packages", "python")
            )
            or current.get("encoding_comparison")
            and key
            in (
                "runtime_sha256",
                "search_source_sha256",
            )
        ):
            current.setdefault("encoding_identity_comparison", {})[key] = {
                "control": control[key],
                "candidate": current[key],
            }
        else:
            assert control[key] == current[key], f"control {key} differs"
    assert {(c["step"], c["sampler"]) for c in control["cells"]} == {
        (step, sampler) for step in POSITIONS for sampler in ("oracle", "uniform")
    }
    assert all("compact" in c["modes"] for c in control["cells"])
    return control


def compare_control(
    path: Path,
    step: int,
    sampler: str,
    receipt: LocalUpdateReceipt,
    capture: dict[str, Any],
    *,
    encoding_comparison: bool = False,
) -> dict[str, Any]:
    receipt_path = path / f"{step}-{sampler}-compact-warm.json.gz"
    commands_path = path / f"{step}-{sampler}-compact-commands.json.gz"
    with gzip.open(receipt_path, "rt") as stream:
        control = ADAPTER.validate_json(stream.read())
    with gzip.open(commands_path, "rt") as stream:
        control_capture = json.load(stream)
    assert control_capture == capture, (
        "one-thread control Commands or final observations differ"
    )
    assert [np.sign(r.signed_value) for r in control.rollouts] == [
        np.sign(r.signed_value) for r in receipt.rollouts
    ], "one-thread control leaf-value signs differ"
    return {
        "control_max_errors": compare(
            replace(control, runtime_sha256=receipt.runtime_sha256)
            if encoding_comparison
            else control,
            receipt,
        ),
        "control_runtime_sha256": control.runtime_sha256,
        "candidate_runtime_sha256": receipt.runtime_sha256,
        "control_evidence_sha256": {
            p.name: file_sha256(p) for p in (receipt_path, commands_path)
        },
    }


def save_gzip(path: Path, value: object) -> None:
    with gzip.open(path, "wt") as stream:
        json.dump(value, stream, sort_keys=True)


def work(receipt: LocalUpdateReceipt) -> dict[str, int | str]:
    config = LocalSearchConfig.model_validate_json(receipt.config_json)
    counts = np.bincount(
        [r.action_index for r in receipt.rollouts], minlength=len(receipt.base)
    )
    assert tuple(counts) == receipt.allocation_counts
    assert all(n == 16 for n, p in zip(counts, receipt.base) if p > 0)
    np.testing.assert_allclose(
        receipt.target,
        regularized_update(
            np.asarray(receipt.base),
            np.asarray(receipt.values, dtype=float),
            np.asarray(receipt.reference),
            alpha=0.02,
            beta=0.01,
        ),
    )
    for i, n in enumerate(counts):
        if n:
            assert np.isclose(
                receipt.values[i],
                np.mean(
                    [r.signed_value for r in receipt.rollouts if r.action_index == i]
                ),
            )
    for row in receipt.rollouts:
        audit = json.loads(row.branch_audit_json)
        assert row.canonical_commands == audit["counters"]["applies"]["child"]
        assert 0 < row.canonical_commands <= 20
        assert row.terminal or row.canonical_commands == 20
        assert audit["audit"] == (config.rollout_evidence == "full")
        assert len(row.actor_observation_hashes) == (
            row.canonical_commands if audit["audit"] else 0
        )
        assert audit["tape_lengths"]["child"] == (
            row.canonical_commands if audit["audit"] else 0
        )
        if not audit["audit"]:
            assert audit["tapes"] == dict(world=[], child=[], leaf=[])
    return {
        "rollouts": len(receipt.rollouts),
        "commands": sum(r.canonical_commands for r in receipt.rollouts),
        "terminal_rollouts": sum(r.terminal for r in receipt.rollouts),
        "legal_actions": len(receipt.base),
        "receipt_digest": receipt.replay_identity(),
    }


def behavior(full: LocalUpdateReceipt, compact: LocalUpdateReceipt) -> dict[str, float]:
    """Omissions only: all other fields pass the retained admission comparison."""
    rows = []
    for old, new in zip(full.rollouts, compact.rollouts, strict=True):
        left, right = (json.loads(r.branch_audit_json) for r in (old, new))
        for field in ("audit", "tapes", "tape_lengths"):
            left.pop(field)
            right.pop(field)
        assert left == right, "native metadata/counters differ"
        rows.append(
            replace(
                new,
                actor_observation_hashes=old.actor_observation_hashes,
                branch_audit_json=old.branch_audit_json,
            )
        )
    assert json.loads(compact.config_json) == {
        **json.loads(full.config_json),
        "rollout_evidence": "compact",
    }
    projected = replace(
        compact, schema=full.schema, config_json=full.config_json, rollouts=tuple(rows)
    )
    return compare(full, projected)


def main() -> None:
    args = arguments()
    args.out.mkdir(exist_ok=False)
    # Process-owned setting, before model construction or inference. Search never
    # changes torch globals; inter-op remains untouched for every sweep process.
    torch.set_num_threads(args.inference_threads)
    modes = ("full", "compact") if args.modes == "paired" else ("compact",)
    checkpoint = Path.home() / ".cache/manabot/sha256" / CHECKPOINT_SHA256
    config = LocalSearchConfig(
        depth=20, worlds=16, alpha=0.02, beta=0.01, decision_seconds=600
    )
    teacher = LocalUpdateTeacher(
        checkpoint, CHECKPOINT_SHA256, config, native_encoder=args.native_encoder
    )
    baseline = json.loads((BASELINE / "result.json").read_text())
    if not args.encoding_comparison:
        assert teacher.runtime_sha256 == baseline["runtime_sha256"]
    assert file_sha256(args.attempt) == baseline["attempt_sha256"]
    if not args.cross_host:
        assert torch.get_num_interop_threads() == baseline["interop_threads"]
    events: list[dict[str, Any]] = [
        json.loads(line) for line in gzip.open(args.attempt, "rt")
    ]
    assert events[0]["event"] == "start" and events[-1]["event"] == "outcome"
    game = events[0]["game"]
    decks = [game["heuristic_deck"], game["heuristic_deck"]]
    decks[game["model_seat"]] = game["model_deck"]
    setup = MatchConfig.authored(PACK, *decks, hero="seat-0", villain="seat-1")
    env = Env(
        Match(setup.model_copy(update={"content_pack": PACK})),
        teacher.space,
        Reward(RewardConfig()),
        seed=game["deal"],
        auto_reset=False,
        enable_profiler=False,
        enable_behavior_tracking=False,
    )
    env.reset(seed=game["deal"])
    player = LocalUpdatePlayer(teacher, seed=game["deal"])
    player.start_game(env, game["model_seat"])
    for source, dest in (
        (Path(__file__), "harness.py"),
        (Path("manabot/sim/search/local_update.py"), "local_update.py"),
    ):
        (args.out / dest).write_bytes(source.read_bytes())
    result: dict[str, Any] = {
        "claim": "Test fixed-root latency while preserving retained behavior, actual branch Commands and final viewer observations.",
        "started_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "head": args.source_commit
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "harness_sha256": file_sha256(Path(__file__)),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime_sha256": teacher.runtime_sha256,
        "native_encoder": args.native_encoder,
        "encoding_comparison": args.encoding_comparison,
        "cross_host": args.cross_host,
        "python_encoder_sha256": file_sha256(Path("manabot/env/observation.py")),
        "search_source_sha256": teacher.source_sha256,
        "attempt": str(args.attempt),
        "attempt_sha256": file_sha256(args.attempt),
        "baseline_sha256": file_sha256(BASELINE / "result.json"),
        "world_binding": teacher.agent.world_binding,
        "config": config.model_dump(),
        "torch_threads": torch.get_num_threads(),
        "requested_inference_threads": args.inference_threads,
        "modes": list(modes),
        "repeats": args.repeats,
        "profiles": not args.skip_profiles,
        "control": str(args.control) if args.control else None,
        "interop_threads": torch.get_num_interop_threads(),
        "device": str(next(teacher.agent.parameters()).device),
        "packages": {name: version(name) for name in ("torch", "numpy", "pydantic")},
        "python": platform.python_version(),
        "load_start": os.getloadavg(),
        "new_games": 0,
        "replayed_commands": 0,
        "errors": [],
        "cells": [],
        "timing_scope": "warm teacher.search only; alternating modes when paired; thread counts run in separate processes; separate optional profiles and untimed externally instrumented warmup admission",
    }

    def save() -> None:
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        if args.control:
            admit_control(args.control, result)
            result["control_sha256"] = file_sha256(args.control / "result.json")
            save()
        for event in events[1:]:
            step = event.get("step", -1)
            if step > max(POSITIONS):
                break
            if event["event"] == "pending":
                actor = int(env._engine.current_agent_index())
                assert (
                    json.loads(env._engine.semantic_decision_frame_json())
                    == event["frame"]
                )
                assert (
                    json.loads(env._engine.semantic_observation_json(actor))
                    == event["observation"]
                )
            if event["event"] == "search" and step in POSITIONS:
                assert player.history is not None
                seed = event["receipt"]["seed"]
                root = (
                    env._engine.search_witness_json(),
                    env._engine.semantic_decision_frame_json(),
                )
                for sampler in ("oracle", "uniform"):
                    teacher.hand_sampler = teacher._sampler_authority = (
                        OracleHandSampler()
                        if sampler == "oracle"
                        else PhysicalHandSampler()
                    )
                    cell: dict[str, Any] = {
                        "step": step,
                        "sampler": sampler,
                        "seed": seed,
                        "modes": {},
                    }
                    result["cells"].append(cell)
                    warmups: dict[str, LocalUpdateReceipt] = {}
                    captures: dict[str, dict[str, Any]] = {}
                    for mode in modes:
                        teacher.config = config.model_copy(
                            update={"rollout_evidence": mode}
                        )
                        capture: dict[str, Any] = {}

                        def apply(
                            session: _SelectedSession,
                            branch: managym.Env,
                            *,
                            site: str,
                            policy_index: int,
                        ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
                            # Native returns the actual submitted Command even with audit off.
                            obs, reward, terminal, truncated, info, encoded = (
                                session.runtime.apply_policy_choice(
                                    branch, site, policy_index
                                )
                            )
                            record = json.loads(encoded)
                            entry = capture.setdefault(
                                session.match_id, {"commands": []}
                            )
                            entry["commands"].append(
                                {
                                    k: record[k]
                                    for k in (
                                        "site",
                                        "policy_index",
                                        "offer_id",
                                        "command",
                                        "native_receipt",
                                    )
                                }
                            )
                            if (
                                branch.is_game_over()
                                or len(entry["commands"]) == config.depth
                            ):
                                entry["final_observations"] = [
                                    json.loads(branch.semantic_observation_json(viewer))
                                    for viewer in (0, 1)
                                ]
                            return obs, reward, terminal, truncated, info

                        with patch.object(
                            _SelectedSession, "apply_policy_choice", apply
                        ):
                            warm = teacher.search(
                                env._engine, player.history, seed=seed
                            )
                        warmups[mode] = warm
                        captures[mode] = capture
                        if sampler == "uniform":
                            LocalUpdateReceipt.from_json(warm.to_json())
                        counts = work(warm)
                        for index, row in enumerate(warm.rollouts):
                            entry = capture[f"local-{seed}-{index}"]
                            assert len(entry["commands"]) == row.canonical_commands
                            assert len(entry["final_observations"]) == 2
                            if mode == "full":
                                tape = json.loads(row.branch_audit_json)["tapes"][
                                    "child"
                                ]
                                assert entry["commands"] == [
                                    {
                                        k: r[k]
                                        for k in (
                                            "site",
                                            "policy_index",
                                            "offer_id",
                                            "command",
                                            "native_receipt",
                                        )
                                    }
                                    for r in tape
                                ]
                        cell["modes"][mode] = {"warmup": counts, "runs": []}
                        save_gzip(
                            args.out / f"{step}-{sampler}-{mode}-warm.json.gz",
                            json.loads(warm.to_json()),
                        )
                        save_gzip(
                            args.out / f"{step}-{sampler}-{mode}-commands.json.gz",
                            capture,
                        )
                        assert root == (
                            env._engine.search_witness_json(),
                            env._engine.semantic_decision_frame_json(),
                        )
                    if "full" in modes:
                        assert captures["full"] == captures["compact"], (
                            "actual Commands or final observations differ"
                        )
                        cell["max_errors"] = behavior(
                            warmups["full"], warmups["compact"]
                        )
                        with gzip.open(
                            BASELINE / f"{step}-{sampler}-warm.json.gz", "rt"
                        ) as stream:
                            old = ADAPTER.validate_json(stream.read())
                        cell["baseline_max_errors"] = compare(old, warmups["full"])
                    if args.control:
                        cell.update(
                            compare_control(
                                args.control,
                                step,
                                sampler,
                                warmups["compact"],
                                captures["compact"],
                                encoding_comparison=args.encoding_comparison,
                            )
                        )
                    cell["external_commands_and_final_observations_equal"] = True
                    save()
                    print(
                        json.dumps(
                            {
                                "phase": "admission",
                                "step": step,
                                "sampler": sampler,
                                "errors": cell.get(
                                    "control_max_errors", cell.get("max_errors")
                                ),
                            }
                        ),
                        flush=True,
                    )
                    for repeat in range(args.repeats):
                        order = modes if repeat % 2 == 0 else modes[::-1]
                        for mode in order:
                            teacher.config = config.model_copy(
                                update={"rollout_evidence": mode}
                            )
                            began = time.perf_counter()
                            receipt = teacher.search(
                                env._engine, player.history, seed=seed
                            )
                            seconds = time.perf_counter() - began
                            counts = work(receipt)
                            assert counts == cell["modes"][mode]["warmup"]
                            assert root == (
                                env._engine.search_witness_json(),
                                env._engine.semantic_decision_frame_json(),
                            )
                            cell["modes"][mode]["runs"].append(
                                {
                                    "repeat": repeat + 1,
                                    "seconds": seconds,
                                    "load": os.getloadavg(),
                                    **counts,
                                }
                            )
                            save()
                            print(
                                json.dumps(
                                    {
                                        "phase": "timed",
                                        "step": step,
                                        "sampler": sampler,
                                        "mode": mode,
                                        "repeat": repeat + 1,
                                        "seconds": seconds,
                                    }
                                ),
                                flush=True,
                            )
                    for mode, detail in cell["modes"].items():
                        seconds = [r["seconds"] for r in detail["runs"]]
                        detail.update(
                            median_seconds=statistics.median(seconds),
                            min_seconds=min(seconds),
                            max_seconds=max(seconds),
                        )
                        if args.skip_profiles:
                            save()
                            continue
                        teacher.config = config.model_copy(
                            update={"rollout_evidence": mode}
                        )
                        profiler = cProfile.Profile()
                        profiler.enable()
                        receipt = teacher.search(env._engine, player.history, seed=seed)
                        profiler.disable()
                        assert work(receipt) == detail["warmup"]
                        assert root == (
                            env._engine.search_witness_json(),
                            env._engine.semantic_decision_frame_json(),
                        )
                        profiler.dump_stats(
                            str(args.out / f"{step}-{sampler}-{mode}.pstats")
                        )
                        stats = pstats.Stats(profiler)
                        detail["profile"] = {
                            "total_seconds": stats.total_tt,
                            "functions": [
                                {
                                    "file": key[0],
                                    "line": key[1],
                                    "name": key[2],
                                    "primitive_calls": value[0],
                                    "calls": value[1],
                                    "self_seconds": value[2],
                                    "cumulative_seconds": value[3],
                                }
                                for key, value in stats.stats.items()
                            ],
                        }
                        save()
                    print(
                        json.dumps(
                            {
                                "phase": "cell_done",
                                "step": step,
                                "sampler": sampler,
                                "medians": {
                                    m: d["median_seconds"]
                                    for m, d in cell["modes"].items()
                                },
                            }
                        ),
                        flush=True,
                    )
            if event["event"] == "command" and step < max(POSITIONS):
                actor = int(env._engine.current_agent_index())
                _, _, ended, truncated, _, transition = env.step_semantic(
                    Command(**event["command"])
                )
                assert not ended and not truncated
                player.observe_step(env, actor, transition)
                result["replayed_commands"] += 1
        assert len(result["cells"]) == 6
        assert all(
            len(d["runs"]) == args.repeats
            for c in result["cells"]
            for d in c["modes"].values()
        )
        result["all_passed"] = True
    except BaseException:
        result["errors"].append(traceback.format_exc())
        raise
    finally:
        result["finished_utc"] = datetime.now(UTC).isoformat()
        result["load_end"] = os.getloadavg()
        save()
        env.close()


if __name__ == "__main__":
    main()
