"""Measure full paired evaluations on the fixed held-out TLA trophy deals.

One invocation measures one device and runner. It retains the ordinary arena
evidence and a summary of completed games, loader batching and wall time.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterable
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

from etude.checkpoints import CheckpointRef
from etude.eval.agents import ModelForward
from etude.eval.records import Code, Match, checkpoint_player
from etude.eval.result import Result
from etude.records import ArtifactRef
import torch

from experiments.mana_taps import OBSERVATION
from experiments.tla_frontier import HEURISTIC, model
from experiments.tla_trophies import (
    corpus_input,
    corpus_path,
    held_out_deal,
    held_out_pairing,
    playable,
)
from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.eval.agent_loader import (
    EVAL_DEVICES,
    device_available,
    load_with_precision,
)
from manabot.eval.agents import build, local_forward
from manabot.eval.pairing import PairingSpec, play_pairing
from manabot.eval.vector import LocalMatches, play_pairings
from manabot.infra.files import atomic_json
from manabot.infra.identity import file_sha256
from manabot.policy.checkpoint import read_container
from manabot.policy.prepare import Prepared
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.registration import EvalPrecision

ARENA = "eval-throughput"
DEALS = 32
RANDOM_SEED = 1


@dataclass(frozen=True)
class ForwardCall:
    rows: int
    seconds: float


def timed_forward(forward: ModelForward, calls: list[ForwardCall]) -> ModelForward:
    """Append one timing per completed loader call."""

    def call(inputs: list[object]) -> list[object]:
        rows = sum(item.batch for item in inputs if isinstance(item, Prepared))
        start = time.perf_counter()
        output = forward(inputs)
        calls.append(ForwardCall(rows, time.perf_counter() - start))
        return output

    return call


def random_checkpoint(
    path: Path, width: int, heads: int = 8, *, template: Path | None = None
) -> None:
    """Generate seeded weights, optionally keeping another policy's input contract."""
    torch.manual_seed(RANDOM_SEED)
    if template is None:
        space, config, readings = ObservationSpace(OBSERVATION), model("w160l8"), None
    else:
        source = read_container(template)
        space, config, readings = source.space, source.model, source.readings
    policy = ActorCritic(
        space,
        config.model_copy(update={"hidden_dim": width, "num_attention_heads": heads}),
        readings=readings,
    )
    save_checkpoint(policy, space, path, player_configs=None)


def checkpoint_ref(path: Path, experiment: str) -> CheckpointRef:
    path = path.resolve(strict=True)
    digest = file_sha256(path)
    return CheckpointRef(
        experiment=experiment,
        job="policy",
        name=digest,
        artifact=ArtifactRef(
            uri=path.as_uri(), sha256=digest, size_bytes=path.stat().st_size
        ),
    )


def trophy_deals(
    ref: CheckpointRef, count: int, precision: EvalPrecision
) -> tuple[Code, dict[str, PairingSpec], list[Match]]:
    source = corpus_input()
    path = corpus_path()
    if file_sha256(path) != source.sha256:
        raise ValueError(f"trophy corpus differs from the data manifest at {path}")
    pools = playable("held_out")
    if len(pools) < 2 * count:
        raise ValueError(f"the trophy split has fewer than {2 * count} playable pools")
    candidate = checkpoint_player(ref)
    heuristic = Code(
        name="heuristic", source=heuristic_source_paths(), config=HEURISTIC
    )
    specs: dict[str, PairingSpec] = {}
    matches: list[Match] = []
    for i in range(count):
        deal = held_out_deal(i, precision)
        # The throughput arms share the same match keys across precisions.
        match = Match(
            players=(candidate, heuristic),
            config=deal.model_dump(exclude={"precision"}),
        )
        matches.append(match)
        specs[match.key] = held_out_pairing(deal, pools, source.sha256, arena=ARENA)
    return heuristic, specs, matches


def completed_games(match: Match, result: Result) -> int:
    plays = [row for row in result.tables["plays"] if row["seat"] == 0]
    if len(plays) != 4 or any(not row["valid"] for row in plays):
        failures = [row["failure"] for row in plays if not row["valid"]]
        raise ValueError(
            f"deal {match.key} did not complete all four games: {failures}"
        )
    return len(plays)


async def count_vector_games(
    results: AsyncIterable[tuple[Match, Result | Exception]],
) -> int:
    games = 0
    async for match, outcome in results:
        if isinstance(outcome, Exception):
            raise RuntimeError(f"deal {match.key} failed") from outcome
        games += completed_games(match, outcome)
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=EVAL_DEVICES, required=True)
    parser.add_argument("--runner", choices=("scalar", "vector"), required=True)
    parser.add_argument("--precision", choices=("float32", "bf16"), default="float32")
    parser.add_argument("--slots", type=int, default=64, help="vector slots")
    parser.add_argument("--deals", type=int, default=DEALS, help=f"1 to {DEALS}")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--checkpoint", type=Path, help="local v11 policy")
    policy.add_argument("--random-width", type=int, choices=(160, 384), default=384)
    args = parser.parse_args()
    if args.slots < 1:
        parser.error("--slots must be positive")
    if not 1 <= args.deals <= DEALS:
        parser.error(f"--deals must be between 1 and {DEALS}")
    if not device_available(args.device):
        parser.error(f"--device {args.device} is unavailable on this host")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"--checkpoint {args.checkpoint} is not a file")
    out = args.output_dir.resolve()
    if out.exists():
        parser.error(f"--output-dir {out} exists; each arm needs a fresh directory")

    out.mkdir(parents=True)
    os.environ["MANABOT_EVAL_DEVICE"] = args.device
    os.environ["ETUDE_OUTPUT_DIR"] = str(out)
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    path = args.checkpoint
    if path is None:
        path = out / f"w{args.random_width}l8.safetensors"
        random_checkpoint(path, args.random_width)
    ref = checkpoint_ref(path, ARENA)
    checkpoint = read_container(path)
    if not checkpoint.space.one_token:
        raise ValueError("the throughput comparison requires a v11 checkpoint")
    model_name = f"w{checkpoint.model.hidden_dim}l{checkpoint.model.attention_layers}"
    parameters = sum(int(value.size) for value in checkpoint.parameters.values())
    del checkpoint

    heuristic, specs, matches = trophy_deals(ref, args.deals, args.precision)
    calls: list[ForwardCall] = []
    forward = timed_forward(load_with_precision(ref, args.precision), calls)
    agents = (forward, build(heuristic, {}))
    started = time.perf_counter()
    if args.runner == "scalar":
        games = 0
        for match in matches:
            spec = specs[match.key]
            result = play_pairing(
                match,
                agents,
                matchup=spec.matchup,
                deal=spec.deal,
                matchup_mode=spec.matchup_mode,
                game_seconds=spec.game_seconds,
                arena=spec.arena,
                precision=spec.precision,
            )
            games += completed_games(match, result)
    else:
        vector_agents = (local_forward(forward), agents[1])
        source = LocalMatches([(match, vector_agents) for match in matches])
        results = play_pairings(
            source, lambda match: specs[match.key], slots=args.slots, output_dir=out
        )
        games = asyncio.run(count_vector_games(results))
    seconds = time.perf_counter() - started
    if games != 4 * len(matches):
        raise RuntimeError(f"completed {games} games across {len(matches)} deals")

    replay_seconds = sum(
        json.loads((out / "arena" / match.key / "replay.json").read_bytes())["seconds"]
        for match in matches
    )
    forward_rows = sum(call.rows for call in calls)
    forward_seconds = sum(call.seconds for call in calls)
    stages = {
        "forward": forward_seconds,
        "replay": replay_seconds,
        "other_arena": seconds - forward_seconds - replay_seconds,
    }
    summary = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": args.device,
        "precision": args.precision,
        "runner": args.runner,
        "slots": 1 if args.runner == "scalar" else args.slots,
        "model": model_name,
        "parameters": parameters,
        "checkpoint_sha256": ref.artifact.sha256,
        "random_seed": RANDOM_SEED if args.checkpoint is None else None,
        "deals": len(matches),
        "games": games,
        "seconds": seconds,
        "games_per_hour": 3600 * games / seconds,
        "forward_calls": len(calls),
        "forward_rows": forward_rows,
        "mean_rows_per_forward": forward_rows / len(calls) if calls else 0,
        "max_rows_per_forward": max((call.rows for call in calls), default=0),
        "stage_seconds": stages,
        "largest_stage": max(stages, key=stages.__getitem__),
    }
    if args.device == "mlx":
        summary["mlx"] = version("mlx")
    if args.device == "cuda":
        summary["gpu"] = torch.cuda.get_device_name()
        summary["cuda"] = torch.version.cuda
    with (out / "forwards.jsonl").open("w") as receipt:
        for call in calls:
            receipt.write(json.dumps(asdict(call)) + "\n")
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
