"""Bounded native choice latency and Python allocation measurements.

Timing excludes fixture setup and allocation tracing. Warm two calls, retain
nine calls per component/root, then measure Python allocation peak separately.
Native Torch/Rust allocations are not counted by tracemalloc. Label and object
models have different untrained policies; complete games measure workflow cost,
not strength or an isolated architectural speedup.
"""

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import signal
from statistics import median
from time import perf_counter
import tracemalloc
from typing import Literal

import torch

from manabot.env import ObservationSpace
from manabot.infra import AgentSpec, ObservationSpaceHypers
from manabot.model.agent import Agent
from manabot.sim.compound import CompoundPolicy, sample_compound
from manabot.sim.structured_policy import RaggedOfferBatch, flatten_projection
import managym


@dataclass(frozen=True)
class Measurement:
    median_seconds: float
    p95_seconds: float
    python_peak_bytes: int


def _measure(call: Callable[[], object]) -> Measurement:
    for _ in range(2):
        call()
    samples: list[float] = []
    for _ in range(9):
        start = perf_counter()
        call()
        samples.append(perf_counter() - start)
    tracemalloc.start()
    try:
        call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return Measurement(median(samples), sorted(samples)[-1], peak)


def _root(count: int) -> tuple[managym.Env, managym.Observation]:
    env = managym.Env(seed=81, skip_trivial=False)
    env.reset(
        [
            managym.PlayerConfig(str(i), {"Mountain": 20, "Gray Ogre": max(20, count)})
            for i in range(2)
        ]
    )
    env.scenario_clear_hand(0)
    env.scenario_clear_hand(1)
    for _ in range(count):
        env.scenario_force_battlefield(0, "Gray Ogre", ready=True)
    raw = env.scenario_refresh()
    for _ in range(50):
        if (
            raw.action_space.action_space_type
            == managym.ActionSpaceEnum.DECLARE_ATTACKER
        ):
            return env, raw
        index = next(
            i
            for i, a in enumerate(raw.action_space.actions)
            if a.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
        )
        raw, _, _, _, _ = env.step(index)
    raise ValueError("fixture did not reach attackers")


def _agent(features: Literal["labels", "objects"]) -> Agent:
    torch.manual_seed(107)
    return Agent(
        ObservationSpace(
            ObservationSpaceHypers(
                max_cards_per_player=100,
                max_permanents_per_player=80,
                max_actions=128,
            )
        ),
        AgentSpec(
            compound_decisions=True,
            compound_features=features,
            hidden_dim=16,
            num_attention_heads=2,
        ),
    )


def run() -> dict[str, object]:
    torch.set_num_threads(1)
    rows: list[dict[str, object]] = []
    games: list[dict[str, object]] = []
    for features in ("labels", "objects"):
        agent = _agent(features)
        for count in (3, 35, 65):
            env, raw = _root(count)
            offers = env.compound_offers()
            projection = json.loads(offers.projection_json())
            enc = agent.observation_space.encoder.hypers

            def encode() -> tuple[dict[str, torch.Tensor], RaggedOfferBatch]:
                observation = {
                    k: torch.as_tensor(v).unsqueeze(0)
                    for k, v in agent.observation_space.encode(raw).items()
                }
                batch = flatten_projection(
                    projection,
                    viewer_json=raw.toJSON(),
                    object_rows=raw.object_row_indexes(
                        enc.max_cards_per_player, enc.max_permanents_per_player
                    )
                    if features == "objects"
                    else None,
                )
                return observation, batch

            observation, batch = encode()
            output = agent.compound(observation, batch, deterministic=True)
            calls: dict[str, Callable[[], object]] = {
                "projection": lambda: env.compound_offers().projection_json(),
                "encoding": encode,
                "decoding": lambda: agent.compound(
                    observation, batch, deterministic=True
                ),
                "lowering": lambda: env.compound_commands_json(
                    offers, output.submission.to_json()
                ),
                "complete_player": lambda: sample_compound(
                    agent, env, raw, deterministic=True
                ),
            }
            rows.append(
                {
                    "features": features,
                    "candidates": count,
                    "components": {
                        name: asdict(_measure(call)) for name, call in calls.items()
                    },
                }
            )
        for seed in (107, 108, 109):
            env = managym.Env(seed=seed)
            raw, _ = env.reset(
                [
                    managym.PlayerConfig(str(i), {"Mountain": 12, "Gray Ogre": 12})
                    for i in range(2)
                ]
            )
            policy = CompoundPolicy(agent)
            start = perf_counter()
            commands = 0
            while not raw.game_over:
                if commands >= 10000:
                    raise ValueError("complete-game command cap")
                raw, _, _, _, _ = env.step(policy.act(env, raw))
                commands += 1
            seconds = perf_counter() - start
            games.append(
                {
                    "features": features,
                    "seed": seed,
                    "seconds": seconds,
                    "commands": commands,
                    "commands_per_second": commands / seconds,
                }
            )
    return {
        "world": str(managym.WORLD_VERSION),
        "threads": 1,
        "warmup": 2,
        "samples": 9,
        "p95": "nearest rank (maximum of nine)",
        "rows": rows,
        "games": games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to replace an existing attempt")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def timeout(signum: int, frame: object) -> None:
        raise TimeoutError("900-second fixture budget exhausted")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 900)
    start = perf_counter()
    result: dict[str, object] = {"status": "failed"}
    try:
        with torch.inference_mode():
            result.update(run())
        result["status"] = "completed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result["seconds"] = perf_counter() - start
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
