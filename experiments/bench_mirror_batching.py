"""Bounded w13 admission of the MTG-268 Python frontier batching port.

Replay a sealed smoke trajectory to three fixed roots; never play new games.
Compare all discrete receipt witnesses and float predictions at MTG-268's
existing atol=1e-6/rtol=1e-5. Retain both complete receipts before comparison,
including rejected cells. Timings are single-host paired diagnostics only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import types
from typing import Any, Sequence, cast

from managym.decision import Command
import numpy as np
import torch

from experiments.search_mirror import OracleHandSampler
from experiments.search_scaling import CHECKPOINT_SHA256, PACK
from manabot.env import Env, Match, Reward
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.infra.identity import file_sha256
from manabot.sim.search.local_update import (
    LocalSearchConfig,
    LocalUpdatePlayer,
    LocalUpdateReceipt,
    LocalUpdateTeacher,
)
from manabot.sim.search.sampling import PhysicalHandSampler

REFERENCE = "0063d781710928df996402957a0b30123d332875"
POSITIONS = (0, 100, 286)


def reference_teacher() -> tuple[type[LocalUpdateTeacher], str]:
    path = "manabot/sim/search/local_update.py"
    source = subprocess.check_output(["git", "show", f"{REFERENCE}:{path}"], text=True)
    module = types.ModuleType("mirror_sequential_reference")
    module.__file__ = str(Path(path).resolve())
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return cast(type[LocalUpdateTeacher], module.LocalUpdateTeacher), source


def error(left: Sequence[float | None], right: Sequence[float | None]) -> float:
    assert [v is None for v in left] == [v is None for v in right]
    a, b = (np.asarray([v for v in row if v is not None]) for row in (left, right))
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
    return float(np.max(np.abs(a - b), initial=0))


def compare(before: LocalUpdateReceipt, after: LocalUpdateReceipt) -> dict[str, float]:
    for left, right in zip(before.rollouts, after.rollouts, strict=True):
        assert asdict(replace(left, signed_value=right.signed_value)) == asdict(
            right
        ), "discrete rollout mismatch"
    assert asdict(
        replace(
            before,
            teacher_source_sha256=after.teacher_source_sha256,
            seconds=after.seconds,
            belief_seconds=after.belief_seconds,
            base=after.base,
            values=after.values,
            target=after.target,
            rollouts=after.rollouts,
        )
    ) == asdict(after), "receipt provenance/support mismatch"
    return {
        "base": error(before.base, after.base),
        "values": error(before.values, after.values),
        "target": error(before.target, after.target),
        "rollout_values": error(
            [r.signed_value for r in before.rollouts],
            [r.signed_value for r in after.rollouts],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--positions", type=int, nargs="+", default=POSITIONS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert file_sha256(args.checkpoint) == CHECKPOINT_SHA256
    reference, source = reference_teacher()
    (args.out / "sequential.py").write_text(source)
    # These JSON events are the existing runner's private, sealed match evidence.
    events: list[dict[str, Any]] = [
        json.loads(line) for line in gzip.open(args.attempt, "rt")
    ]
    start = events[0]
    assert start["event"] == "start" and events[-1]["event"] == "outcome"
    config = LocalSearchConfig(
        depth=20, worlds=4, alpha=0.02, beta=0.01, decision_seconds=600
    )
    teachers = {
        "sequential": reference(args.checkpoint, CHECKPOINT_SHA256, config),
        "batched": LocalUpdateTeacher(args.checkpoint, CHECKPOINT_SHA256, config),
    }
    teacher = teachers["batched"]
    game = start["game"]
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
    result: dict[str, Any] = {
        "observed_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "reference": REFERENCE,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime_sha256": teacher.runtime_sha256,
        "search_source_sha256": teacher.source_sha256,
        "world_binding": teacher.agent.world_binding,
        "config": config.model_dump(),
        "attempt": str(args.attempt),
        "attempt_sha256": file_sha256(args.attempt),
        "packages": {name: version(name) for name in ("torch", "numpy", "pydantic")},
        "positions": args.positions,
        "cells": [],
        "replayed_commands": 0,
        "all_passed": False,
        "new_games": 0,
        "training_updates": 0,
        "timing_scope": "one pair per fixed root/sampler/world count; laptop, no mini contention or campaign ETA claim",
    }

    def save() -> None:
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        for event in events[1:]:
            step = event.get("step", -1)
            if step > max(args.positions):
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
            if event["event"] == "search" and step in args.positions:
                assert player.history is not None
                for sampler in ("oracle", "uniform"):
                    for worlds in (4, 8, 16):
                        cell: dict[str, Any] = {
                            "step": step,
                            "sampler": sampler,
                            "worlds": worlds,
                            "passed": False,
                        }
                        result["cells"].append(cell)
                        receipts: dict[str, LocalUpdateReceipt] = {}
                        order = (
                            ("sequential", "batched")
                            if len(result["cells"]) % 2
                            else ("batched", "sequential")
                        )
                        for name in order:
                            current = teachers[name]
                            current.config = config.model_copy(
                                update={"worlds": worlds}
                            )
                            current.hand_sampler = current._sampler_authority = (
                                OracleHandSampler()
                                if sampler == "oracle"
                                else PhysicalHandSampler()
                            )
                            # Each call starts at the same root and receives the same retained seed.
                            began = time.perf_counter()
                            receipt = current.search(
                                env._engine,
                                player.history,
                                seed=event["receipt"]["seed"],
                            )
                            cell[name + "_seconds"] = time.perf_counter() - began
                            receipts[name] = receipt
                            with gzip.open(
                                args.out / f"{step}-{sampler}-{worlds}-{name}.json.gz",
                                "wt",
                            ) as output:
                                output.write(receipt.to_json())
                        try:
                            cell["max_error"] = compare(
                                receipts["sequential"], receipts["batched"]
                            )
                            if sampler == "uniform":
                                LocalUpdateReceipt.from_json(
                                    receipts["batched"].to_json()
                                )
                            cell["passed"] = True
                        except Exception:
                            cell["error"] = traceback.format_exc()
                        cell["rollouts"] = len(receipts["batched"].rollouts)
                        cell["commands"] = sum(
                            r.canonical_commands for r in receipts["batched"].rollouts
                        )
                        cell["terminal_rollouts"] = sum(
                            r.terminal for r in receipts["batched"].rollouts
                        )
                        cell["speedup"] = (
                            cell["sequential_seconds"] / cell["batched_seconds"]
                        )
                        save()
                        print(json.dumps(cell), flush=True)
                # Deadline failure is checked on the actual trained root for both paths.
                for current in teachers.values():
                    try:
                        current.search(env._engine, player.history, seed=1, deadline=1)
                    except TimeoutError:
                        pass
                    else:
                        raise AssertionError("expired deadline returned a target")
            if event["event"] == "command" and step < max(args.positions):
                acting = int(env._engine.current_agent_index())
                command = Command(**event["command"])
                _, _, ended, truncated, _, transition = env.step_semantic(command)
                assert not ended and not truncated
                player.observe_step(env, acting, transition)
                result["replayed_commands"] += 1
        assert len(result["cells"]) == 6 * len(args.positions)
        result["all_passed"] = all(c["passed"] for c in result["cells"])
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["finished_utc"] = datetime.now(UTC).isoformat()
        save()
        env.close()
    if not result["all_passed"]:
        raise SystemExit("Batching admission failed; retain the sequential campaign")


if __name__ == "__main__":
    main()
