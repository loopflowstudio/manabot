"""Bounded fixed-position local-search timing and pre-change receipt comparison.

Run `uv run python scripts/bench_local_update.py --reference 841b1c50c`.
No games are played: both teachers search the same two seeded opening positions
at depth 20, on one CPU thread. By default the policy is initialized (64 x 2),
not trained; --checkpoint selects an admitted observation-only checkpoint.
The reference is loaded from git, never installed as another production path.
Timings are unprofiled; a separate search profiles the largest allocation.
For native attribution, build the extension with --features python,benchmark;
ordinary wheels omit those timers. Instrumented builds also include existing
layer/allocation instrumentation, so use an ordinary wheel for final latency.
Before replacing the native library, --retain-reference DIR saves complete
reference receipts and times. After rebuilding, --native-reference DIR compares
against that original runtime as well as both reference Python implementations.
Native categories are inclusive: witness subcategories overlap construction;
returned observation build and Python conversion are separate from rules work.
"""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import types
from typing import Sequence, cast
from unittest.mock import patch

from managym.decision import Observation
import numpy as np
import torch

from manabot.belief.state import ViewerHistory
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.infra.identity import file_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint
from manabot.model.world import validate_agent_setup
from manabot.sim.search.local_update import (
    LocalSearchConfig,
    LocalUpdateReceipt,
    LocalUpdateTeacher,
)
import managym


def reference_module(revision: str, relative_path: str) -> types.ModuleType:
    """Load retained code for timing; its module path uses the current checkout.

    Compare rollout content, not this temporary teacher's source receipt identity.
    """
    path = Path(__file__).resolve().parents[1] / relative_path
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{relative_path}"], text=True
    )
    module = types.ModuleType(f"_local_update_benchmark_{path.stem}")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


@dataclass(frozen=True)
class Parity:
    max_base_error: float
    max_value_error: float
    max_target_error: float


@dataclass(frozen=True)
class Timing:
    position_seed: int
    worlds: int
    before_seconds: float
    after_seconds: float
    rollouts: int
    commands: int
    parity: Parity

    @property
    def speedup(self) -> float:
        return self.before_seconds / self.after_seconds


@dataclass
class Cost:
    calls: int = 0
    seconds: float = 0.0
    self_seconds: float = 0.0


def compare(before: LocalUpdateReceipt, after: LocalUpdateReceipt) -> Parity:
    """All discrete evidence must match; only numeric predictions may round."""
    for left, right in zip(before.rollouts, after.rollouts, strict=True):
        assert asdict(replace(left, signed_value=right.signed_value)) == asdict(right)
    # Compare all fields by default, including future receipt additions. These
    # exceptions are either measured below or intentionally differ across runs.
    # asdict also handles the reference loaded under a different module name.
    assert asdict(
        replace(
            before,
            runtime_sha256=after.runtime_sha256,
            teacher_source_sha256=after.teacher_source_sha256,
            seconds=after.seconds,
            belief_seconds=after.belief_seconds,
            base=after.base,
            values=after.values,
            target=after.target,
            rollouts=after.rollouts,
        )
    ) == asdict(after)
    return Parity(
        max_base_error=_prediction_error(before.base, after.base),
        max_value_error=max(
            _prediction_error(before.values, after.values),
            _prediction_error(
                [row.signed_value for row in before.rollouts],
                [row.signed_value for row in after.rollouts],
            ),
        ),
        max_target_error=_prediction_error(before.target, after.target),
    )


def _prediction_error(
    before: Sequence[float | None], after: Sequence[float | None]
) -> float:
    """Require identical support and the same tolerance for all predictions."""
    assert [value is None for value in before] == [value is None for value in after]
    left = np.asarray([value for value in before if value is not None])
    right = np.asarray([value for value in after if value is not None])
    np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)
    return float(np.max(np.abs(left - right), initial=0.0))


def profile_costs(profile: cProfile.Profile) -> dict[str, Cost]:
    """Inclusive and self costs, from a separate instrumented search.

    Model subcosts overlap ``network``; do not add them to it. Tensor assembly
    counts NumPy stacking and Torch's zero-copy wrappers, not model operations.
    Native stepping includes Command validation, audit and returned observation.
    Semantic serialization includes the native viewer hash.
    """
    groups: dict[str, Cost] = {}
    for entry in profile.getstats():
        code = entry.code
        if isinstance(code, str):
            name = None
            for method, group in (
                ("semantic_observation_json", "semantic_json_and_hash"),
                ("observation_for_player", "native_observation"),
                ("encode_observation_into", "observation_encoding"),
                ("apply_policy_choice", "native_step_and_audit"),
                ("as_tensor", "tensor_assembly"),
            ):
                if method in code:
                    name = group
                    break
        elif code.co_qualname == "ActorCritic.forward" and code.co_filename.endswith(
            "model/actor_critic.py"
        ):
            name = "network"
        elif code.co_name == "snapshot" and code.co_filename.endswith(
            "search/branch.py"
        ):
            name = "session_snapshot"
        elif code.co_qualname == "Observation.from_json" and code.co_filename.endswith(
            "managym/decision.py"
        ):
            name = "semantic_python_parsing"
        elif code.co_qualname == "ObservationSpace.encode":
            name = "observation_encoding"
        elif code.co_name == "stack" and "numpy" in code.co_filename:
            name = "tensor_assembly"
        elif code.co_filename.endswith("model/actor_critic.py") and code.co_name in (
            "_gather_object_embeddings",
            "_attend_objects",
            "_gather_informed_actions",
            "_value_from_objects",
        ):
            name = "model/" + code.co_name.removeprefix("_")
        else:
            name = None
        if name is not None:
            bucket = groups.setdefault(name, Cost())
            bucket.calls += entry.callcount
            bucket.seconds += entry.totaltime
            bucket.self_seconds += entry.inlinetime
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="841b1c50c")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--worlds", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--position-seeds", type=int, nargs="+", default=[42, 43])
    retention = parser.add_mutually_exclusive_group()
    retention.add_argument(
        "--retain-reference",
        type=Path,
        help="Save original-runtime timings and complete reference receipts",
    )
    retention.add_argument(
        "--native-reference",
        type=Path,
        help="Compare against previously retained original-runtime receipts",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(15)
    # Retain both changed execution owners. Loading only the old teacher would
    # silently give the baseline today's optimized attention implementation.
    reference = cast(
        type[LocalUpdateTeacher],
        reference_module(
            args.reference, "manabot/sim/search/local_update.py"
        ).LocalUpdateTeacher,
    )
    reference_model = reference_module(args.reference, "manabot/model/actor_critic.py")
    match = Match(
        MatchConfig(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        )
    )
    with TemporaryDirectory(prefix="local-update-bench-") as temporary:
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = Path(temporary) / "initialized.pt"
            space = ObservationSpace()
            agent = ActorCritic(
                space,
                ModelConfig(hidden_dim=64, num_attention_heads=4, attention_layers=2),
            )
            save_checkpoint(
                agent,
                space,
                checkpoint,
                player_configs=match.to_rust(),
                extra={"value_semantic": "signed_outcome"},
            )
        digest = file_sha256(checkpoint)
        config = LocalSearchConfig(depth=20, decision_seconds=120)
        with patch("manabot.model.checkpoint.ActorCritic", reference_model.ActorCritic):
            before = reference(checkpoint, digest, config)
        teachers = {
            "before": before,
            "after": LocalUpdateTeacher(checkpoint, digest, config),
        }
        for teacher in teachers.values():
            validate_agent_setup(teacher.agent, match.to_rust())
        results: list[Timing] = []
        profiles: dict[str, dict[str, Cost]] = {}
        native_profiles: dict[str, object] = {}
        original_timings: list[Timing] = []
        original_runtime_sha256: str | None = None
        for position_seed in args.position_seeds:
            engine = managym.Env(seed=position_seed, skip_trivial=True)
            engine.reset(match.to_rust())
            viewer = engine.current_agent_index()
            history = ViewerHistory.from_observation(
                Observation.from_json(engine.semantic_observation_json(viewer))
            )
            for teacher in teachers.values():
                teacher.predict(engine)  # Warm the policy, without advancing a game.
            for worlds in args.worlds:
                receipts: dict[str, LocalUpdateReceipt] = {}
                seconds: dict[str, float] = {}
                for name, teacher in teachers.items():
                    teacher.config = config.model_copy(update={"worlds": worlds})
                    started = time.perf_counter()
                    receipts[name] = teacher.search(engine, history, seed=17)
                    seconds[name] = time.perf_counter() - started
                results.append(
                    Timing(
                        position_seed,
                        worlds,
                        seconds["before"],
                        seconds["after"],
                        len(receipts["after"].rollouts),
                        sum(r.canonical_commands for r in receipts["after"].rollouts),
                        compare(receipts["before"], receipts["after"]),
                    )
                )
                evidence_name = f"{position_seed}-{worlds}.json"
                if args.retain_reference:
                    args.retain_reference.mkdir(parents=True, exist_ok=True)
                    (args.retain_reference / evidence_name).write_text(
                        json.dumps(
                            {
                                "reference": args.reference,
                                "seconds": seconds["before"],
                                "receipt": asdict(receipts["before"]),
                            }
                        )
                    )
                if args.native_reference:
                    retained = json.loads(
                        (args.native_reference / evidence_name).read_text()
                    )
                    original = LocalUpdateReceipt.from_json(
                        json.dumps(retained["receipt"])
                    )
                    if original_runtime_sha256 is not None:
                        assert original.runtime_sha256 == original_runtime_sha256
                    original_runtime_sha256 = original.runtime_sha256
                    assert retained["reference"] == args.reference
                    original_timings.append(
                        replace(
                            results[-1],
                            before_seconds=retained["seconds"],
                            parity=compare(original, receipts["after"]),
                        )
                    )
            if position_seed == args.position_seeds[0]:
                for name, teacher in teachers.items():
                    teacher.config = config.model_copy(
                        update={"worlds": max(args.worlds)}
                    )
                    profiler = cProfile.Profile()
                    native_profile = hasattr(
                        managym.SelectedBranchRuntime, "start_benchmark"
                    )
                    if native_profile:
                        managym.SelectedBranchRuntime.start_benchmark()
                    try:
                        with profiler:
                            teacher.search(engine, history, seed=17)
                    finally:
                        if native_profile:
                            native_profiles[name] = json.loads(
                                managym.SelectedBranchRuntime.finish_benchmark_json()
                            )
                    profiles[name] = profile_costs(profiler)
        print(
            json.dumps(
                {
                    "reference": args.reference,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "checkpoint_sha256": digest,
                    "initialized_policy": args.checkpoint is None,
                    "model": teachers["after"].agent.config.model_dump(mode="json"),
                    "depth": 20,
                    "threads": 1,
                    "runtime_sha256": receipts["after"].runtime_sha256,
                    "native_instrumented_build": bool(native_profiles),
                    "reference_native": "shared with current teacher",
                    "singleton_ratio_reference": (
                        "retained original runtime"
                        if original_timings
                        else "shared runtime"
                    ),
                    "timings": [
                        {**asdict(row), "speedup": row.speedup} for row in results
                    ],
                    "original_runtime_sha256": original_runtime_sha256,
                    "original_runtime_timings": [
                        {**asdict(row), "speedup": row.speedup}
                        for row in original_timings
                    ],
                    "sixteen_worlds_over_reference_singleton": {
                        str(row.position_seed): row.after_seconds
                        / singleton.before_seconds
                        for row in (original_timings or results)
                        if row.worlds == 16
                        for singleton in (original_timings or results)
                        if singleton.position_seed == row.position_seed
                        and singleton.worlds == 1
                    },
                    "native_profile_at_max_worlds": native_profiles,
                    "profile_at_max_worlds": {
                        name: {key: asdict(cost) for key, cost in costs.items()}
                        for name, costs in profiles.items()
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
