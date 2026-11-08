"""Bounded fixed-position local-search timing and pre-change receipt comparison.

Run `uv run python scripts/bench_local_update.py --reference 841b1c50c`.
No games are played: both teachers search the same two seeded opening positions
at depth 20, on one CPU thread. By default the policy is initialized (64 x 2),
not trained; --checkpoint selects an admitted observation-only checkpoint.
The reference is loaded from git, never installed as another production path.
Timings are unprofiled; a separate search profiles the largest allocation.
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
from typing import cast

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


def reference_teacher(revision: str) -> type[LocalUpdateTeacher]:
    """Load retained code for timing; its module path uses the current checkout.

    Compare rollout content, not this temporary teacher's source receipt identity.
    """
    path = Path(__file__).resolve().parents[1] / "manabot/sim/search/local_update.py"
    source = subprocess.check_output(
        ["git", "show", f"{revision}:manabot/sim/search/local_update.py"], text=True
    )
    module = types.ModuleType("_local_update_benchmark_reference")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    # This git-loaded class is a benchmark-only untyped boundary.
    return cast(type[LocalUpdateTeacher], module.LocalUpdateTeacher)


@dataclass(frozen=True)
class Parity:
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


def compare(before: LocalUpdateReceipt, after: LocalUpdateReceipt) -> Parity:
    """All discrete evidence must match; only numeric predictions may round."""
    if len(before.rollouts) != len(after.rollouts):
        raise AssertionError("rollout counts differ")
    for left, right in zip(before.rollouts, after.rollouts, strict=True):
        assert asdict(replace(left, signed_value=right.signed_value)) == asdict(right)
    for key in ("base", "reference", "offer_ids", "allocation_counts"):
        assert getattr(before, key) == getattr(after, key), key
    value_error = max(
        abs(left.signed_value - right.signed_value)
        for left, right in zip(before.rollouts, after.rollouts, strict=True)
    )
    target_error = float(np.max(np.abs(np.asarray(before.target) - after.target)))
    np.testing.assert_allclose(
        [row.signed_value for row in before.rollouts],
        [row.signed_value for row in after.rollouts],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(before.target, after.target, rtol=1e-5, atol=1e-6)
    return Parity(value_error, target_error)


def profile_costs(profile: cProfile.Profile) -> dict[str, Cost]:
    """Inclusive costs; semantic serialization includes the native viewer hash."""
    groups: dict[str, Cost] = {}
    for entry in profile.getstats():
        code = entry.code
        if isinstance(code, str):
            name = (
                "semantic_json_and_hash"
                if "semantic_observation_json" in code
                else None
            )
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
        else:
            name = None
        if name is not None:
            bucket = groups.setdefault(name, Cost())
            bucket.calls += entry.callcount
            bucket.seconds += entry.totaltime
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="841b1c50c")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--worlds", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--position-seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(15)
    reference = reference_teacher(args.reference)
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
        teachers = {
            "before": reference(checkpoint, digest, config),
            "after": LocalUpdateTeacher(checkpoint, digest, config),
        }
        for teacher in teachers.values():
            validate_agent_setup(teacher.agent, match.to_rust())
        results: list[Timing] = []
        profiles: dict[str, dict[str, Cost]] = {}
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
            if position_seed == args.position_seeds[0]:
                for name, teacher in teachers.items():
                    teacher.config = config.model_copy(
                        update={"worlds": max(args.worlds)}
                    )
                    profiler = cProfile.Profile()
                    with profiler:
                        teacher.search(engine, history, seed=17)
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
                    "timings": [
                        {**asdict(row), "speedup": row.speedup} for row in results
                    ],
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
