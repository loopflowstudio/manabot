"""Training cost with decision history off, compared between two source trees on one host.

`case` times whole-game self-play collection and one damped-policy update per
iteration, using whichever `manabot` and `managym` are first on the import
path, and uses only APIs both trees share. `compare` fetches a reference
revision, builds its engine, runs fresh `case` processes in the order
candidate, reference, reference, candidate, and writes `summary.json`.

Hosts differ by more than the difference of interest, so only cases from one
`compare` are comparable. The policy is a seeded random initialization: this
measures software cost, never learning or strength.
"""

import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tarfile
import time
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "https://github.com/loopflowstudio/manabot/archive/{revision}.tar.gz"
ORDER = ("candidate", "reference", "reference", "candidate")


def case(args: argparse.Namespace) -> None:
    import torch

    import manabot
    import manabot.backends.torch as backend
    from manabot.backends.torch.collection import CollectionSession
    from manabot.backends.torch.learner import Learner
    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.match import Match, Reward
    from manabot.env.observation import ObservationSpace
    from manabot.infra.config import (
        MatchConfig,
        ModelConfig,
        ObservationConfig,
        RewardConfig,
    )
    from manabot.training.decks import UniformDecks
    from manabot.training.learning import DampedPolicyLearning
    from manabot.training.rollout.collection import (
        close_collector,
        collect,
        open_collector,
    )
    from manabot.training.updates.damped_policy import update_damped_iteration
    import managym

    space = ObservationSpace(ObservationConfig())
    torch.manual_seed(39100)
    model = ActorCritic(
        space,
        ModelConfig(
            world=managym.WORLD_SETUP,
            hidden_dim=64,
            attention_layers=2,
            num_attention_heads=4,
            policy_film=True,
            value_aggregation="masked_mean",
        ),
    )
    pack = "elemental-world-v2"
    decks = UniformDecks(
        decks=tuple(
            managym.Deck(player.decklist, player.sideboard)
            for key in managym.authored_deck_keys(pack)
            for player in [managym.authored_deck_setup(pack, key)]
        ),
        sampling="uniform-independent-v1",
        decklists="mixed",
    )
    collector = open_collector(
        space,
        Match(MatchConfig(world=managym.WORLD_SETUP)),
        Reward(RewardConfig()),
        num_envs=args.streams,
        numerics=CollectionSession(
            device=args.device, seed=111, precision=args.precision
        ),
        seed=111,
        opponent_mode="self",
        learning_seats="both",
        workers=args.workers,
        games_per_stream=1,
        deck_source=decks,
    )
    iterations: list[dict[str, float | int]] = []
    try:
        learner = Learner(
            model, args.device, minibatch_rows=256, precision=args.precision
        )
        learning = DampedPolicyLearning(
            gradient="damped_policy", advantage_quantile=0, min_advantage=0
        )
        for iteration in range(1, args.iterations + 1):
            started = time.perf_counter()
            batch = collect(collector, model, 64)
            collected = time.perf_counter()
            update_damped_iteration(backend, learner, batch, learning, iteration)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            finished = time.perf_counter()
            iterations.append(
                {
                    "decisions": int(batch.actions.size),
                    "collection_seconds": collected - started,
                    "update_seconds": finished - collected,
                }
            )
            print(json.dumps(iterations[-1]), flush=True)
    finally:
        close_collector(collector)
    extension = next(Path(managym.__file__).parent.glob("_managym*.so"))
    args.out.write_text(
        json.dumps(
            {
                "iterations": iterations,
                "python_source": str(Path(manabot.__file__).parent),
                "native_source": str(extension),
                "extension_sha256": sha256(extension.read_bytes()).hexdigest(),
                "device": args.device
                if not args.device.startswith("cuda")
                else torch.cuda.get_device_name(),
                "precision": args.precision,
                "streams": args.streams,
                "workers": args.workers,
            },
            indent=2,
        )
        + "\n"
    )


def _reference(out: Path, revision: str) -> Path:
    """Fetch the reference tree and build its engine for this interpreter."""
    with urlopen(ARCHIVE.format(revision=revision), timeout=120) as reply:
        archive = tarfile.open(fileobj=io.BytesIO(reply.read()), mode="r:gz")
    archive.extractall(out, filter="data")
    tree = out / f"manabot-{revision}"
    with (out / "reference-build.log").open("w") as log:
        subprocess.run(
            ["uv", "run", "--no-sync", "maturin", "build", "--release"]
            + ["-i", sys.executable, "-m", str(tree / "managym/Cargo.toml")]
            + ["-o", str(out / "wheels")],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    with ZipFile(next((out / "wheels").glob("*.whl"))) as wheel:
        name = next(n for n in wheel.namelist() if Path(n).match("_managym*.so"))
        (tree / "managym" / Path(name).name).write_bytes(wheel.read(name))
    return tree


def compare(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    trees = {"candidate": ROOT, "reference": args.reference_tree}
    if trees["reference"] is None:
        trees["reference"] = _reference(args.out, args.reference)
    seconds: dict[str, list[float]] = {"candidate": [], "reference": []}
    rows: list[dict[str, object]] = []
    for ordinal, side in enumerate(ORDER):
        receipt = args.out / f"{ordinal}-{side}.json"
        subprocess.run(
            [sys.executable, __file__, "case", "--out", str(receipt)]
            + ["--device", args.device, "--precision", args.precision]
            + ["--streams", str(args.streams), "--workers", str(args.workers)]
            + ["--iterations", str(args.iterations)],
            env=dict(os.environ, PYTHONPATH=str(trees[side])),
            check=True,
        )
        result = json.loads(receipt.read_text())
        if Path(result["python_source"]).parent != trees[side].resolve():
            raise RuntimeError(f"{side} imported {result['python_source']}")
        # The first iteration carries allocator and kernel warm-up.
        timed = result["iterations"][1:]
        decisions = sum(row["decisions"] for row in timed)
        total = sum(row["collection_seconds"] + row["update_seconds"] for row in timed)
        seconds[side].append(total / decisions)
        rows.append(
            {
                "order": ordinal,
                "side": side,
                "decisions": decisions,
                "collection_seconds": sum(r["collection_seconds"] for r in timed),
                "update_seconds": sum(r["update_seconds"] for r in timed),
                "decisions_per_second": decisions / total,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    summary = {
        "reference": args.reference,
        "cases": rows,
        "candidate_over_reference_seconds_per_decision": median(seconds["candidate"])
        / median(seconds["reference"]),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("case", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--precision", default="bf16")
        command.add_argument("--streams", type=int, default=256)
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--iterations", type=int, default=3)
    commands.choices["compare"].add_argument("--reference", required=True)
    commands.choices["compare"].add_argument("--reference-tree", type=Path)
    args = parser.parse_args()
    (case if args.command == "case" else compare)(args)


if __name__ == "__main__":
    main()
