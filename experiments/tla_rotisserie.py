"""Compare terminal game rewards with opening values after public TLA drafts.

Measured execution requires an admitted competent frozen player. --fixtures is a
controlled software proof, never a comparison or substitute player.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

import manabot.backends.torch as torch_backend
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.drafting import prepare_draft
from manabot.infra.config import MatchConfig
from manabot.infra.files import atomic_json
from manabot.infra.identity import file_sha256
from manabot.infra.source import source_commit, source_dirty
from manabot.training.execution import execute_regime
from manabot.training.regime import Execution, TrainingRegime, TrainRotisserie
from manabot.training.rotisserie import validate_software
from manabot.verify.store import VerifyStore
import managym
from managym import _managym


def compare(player: Path, out: Path, seed: int, updates: int, seconds: float) -> None:
    started = perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, object] = {
        "status": "preflight",
        "source_commit": source_commit(),
        "source_dirty": source_dirty(),
        "seed": seed,
        "updates": updates,
        "seconds": seconds,
        "expected_checkpoints": [
            "shared-initialization",
            *[f"{arm}-{i}" for arm in ("game", "value") for i in range(1, updates + 1)],
        ],
        "player": str(player),
        "native_sha256": file_sha256(_managym.__file__),
        "world": managym.world_identity(),
    }
    atomic_json(out / "comparison.json", receipt)
    try:
        digest = file_sha256(player)
        receipt["player_sha256"] = digest
        pilot, _ = load_checkpoint(str(player))
        pilot.eval().requires_grad_(False)
        if pilot.value_semantic not in ("signed_outcome", "win_logit"):
            raise ValueError("player critic has no supported signed semantic")
        if "world_digest" not in (pilot.world_binding or {}):
            raise ValueError(
                "Rotisserie requires a world-bound player admitting generated decks"
            )
    except Exception as error:
        receipt.update(
            status="admission_refused", error=f"{type(error).__name__}: {error}"
        )
        atomic_json(out / "comparison.json", receipt)
        raise
    reference = {
        "path": str(player.resolve()),
        "sha256": digest,
        "bytes": player.stat().st_size,
    }
    receipt.update(status="running", player=reference)
    atomic_json(out / "comparison.json", receipt)
    runs = []
    try:
        with VerifyStore(out / "training.sqlite") as store:
            for arm in ("game", "value"):
                left = seconds - (perf_counter() - started)
                if left <= 0:
                    raise TimeoutError(
                        "comparison deadline; remaining arm/checkpoints are missing"
                    )
                regime = TrainingRegime(
                    id=f"tla-rotisserie-{arm}",
                    world=managym.WORLD_VERSION,
                    match=MatchConfig(),
                    stages=[
                        TrainRotisserie(
                            id=arm,
                            player=reference,
                            reward=arm,
                            updates=updates,
                            evaluate_initial=arm == "game",
                            execution=Execution(wall_seconds=left, threads=1),
                        )
                    ],
                    wall_seconds=left,
                )
                run = execute_regime(
                    regime, seed, out / arm, store, experiment="tla_rotisserie"
                )
                initial_path = out / arm / arm / "drafts" / "initialization.json"
                initial = json.loads(initial_path.read_text())
                if (
                    "initial_parameter_digest" in receipt
                    and receipt["initial_parameter_digest"]
                    != initial["parameter_digest"]
                ):
                    raise ValueError(
                        "reward arms have different initialization tensors"
                    )
                receipt["initial_parameter_digest"] = initial["parameter_digest"]
                runs.append(
                    {
                        "arm": arm,
                        "id": run.id,
                        "status": run.status,
                        "error": run.error,
                        "metrics_run": run.metrics_run,
                    }
                )
                receipt["runs"] = runs
                atomic_json(out / "comparison.json", receipt)
                if run.status != "completed":
                    raise RuntimeError(f"{arm} arm {run.status}: {run.error}")
    except Exception as error:
        receipt.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    else:
        receipt["status"] = "completed_exploratory"
    finally:
        receipt["wall_seconds"] = perf_counter() - started
        atomic_json(out / "comparison.json", receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="retain one unscored reference draft")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=370)
    comparison = commands.add_parser(
        "compare", help="bounded rewards and fixed-reference evaluation"
    )
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--seed", type=int, default=370)
    comparison.add_argument("--player", type=Path)
    comparison.add_argument("--updates", type=int, default=2)
    comparison.add_argument("--seconds", type=float, default=1800)
    comparison.add_argument("--fixtures", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.command == "prepare":
            result = prepare_draft(args.out, args.seed)
            print(
                f"software validation: {result.draft_picks} draft picks, {len(result.decks)} admitted decks; supply {sum(result.initial)} -> {sum(result.remaining)}; receipt: {args.out / 'receipt.json'}"
            )
        elif args.fixtures:
            validate_software(torch_backend, args.out, args.seed)
            print(
                f"software validation only: one synthetic-return update; {args.out / 'receipt.json'}"
            )
        else:
            if args.player is None or args.updates < 1 or args.seconds <= 0:
                parser.error(
                    "compare requires --player, positive updates and positive seconds"
                )
            compare(args.player, args.out, args.seed, args.updates, args.seconds)
            print(f"exploratory comparison: {args.out / 'comparison.json'}")
    except (ValueError, RuntimeError, OSError, TimeoutError) as error:
        raise SystemExit(f"No completed comparison: {error}") from None


if __name__ == "__main__":
    main()
