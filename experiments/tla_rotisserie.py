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
from manabot.training.records import TrainingRun
from manabot.training.regime import Execution, TrainingRegime, TrainRotisserie
from manabot.training.rotisserie import validate_software
from manabot.verify.store import VerifyStore
import managym
from managym import _managym


def _arm_record(arm: str, run: TrainingRun) -> dict[str, object]:
    return {
        "arm": arm,
        "id": run.id,
        "status": run.status,
        "error": run.error,
        "metrics_run": run.metrics_run,
    }


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
            *[f"{arm}-{i}" for arm in ("value", "game") for i in range(1, updates + 1)],
        ],
        "arm_order": ["value", "game"],
        "budget_policy": "half remaining time reserved for each arm at admission",
        "drafts_per_update": 1,
        "evaluation_deals": 1,
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
            arm_seconds = (seconds - (perf_counter() - started)) / 2
            for arm in ("value", "game"):
                left = seconds - (perf_counter() - started)
                if left <= 0:
                    raise TimeoutError(
                        "comparison deadline; remaining arm/checkpoints are missing"
                    )
                # The first arm cannot consume the second arm's reservation.
                budget = min(left, arm_seconds) if arm == "value" else left
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
                            drafts_per_update=1,
                            evaluate_initial=arm == "value",
                            execution=Execution(wall_seconds=budget, threads=1),
                        )
                    ],
                    wall_seconds=budget,
                )
                try:
                    run = execute_regime(
                        regime, seed, out / arm, store, experiment="tla_rotisserie"
                    )
                except Exception as error:
                    # Deadline stops retain exported state and leave the other
                    # arm's reservation available. Other failures stop both arms.
                    record = out / arm / "run.json"
                    if not record.exists() or not isinstance(error, TimeoutError):
                        if record.exists():
                            failed = TrainingRun.model_validate_json(record.read_text())
                            runs.append(_arm_record(arm, failed))
                            receipt["runs"] = runs
                        raise
                    run = TrainingRun.model_validate_json(record.read_text())
                initial_path = out / arm / arm / "drafts" / "initialization.json"
                if initial_path.exists():
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
                elif run.status == "completed":
                    raise ValueError(f"{arm} arm has no initialization receipt")
                runs.append(_arm_record(arm, run))
                receipt["runs"] = runs
                atomic_json(out / "comparison.json", receipt)
                if run.status != "completed" and not (
                    run.status == "interrupted"
                    and (run.error or "").startswith("TimeoutError:")
                ):
                    raise RuntimeError(f"{arm} arm {run.status}: {run.error}")
    except Exception as error:
        receipt.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    else:
        receipt["status"] = (
            "completed_exploratory"
            if len(runs) == 2 and all(run["status"] == "completed" for run in runs)
            else "incomplete"
        )
        if receipt["status"] == "incomplete":
            receipt["error"] = (
                "arm deadline exceeded; see retained run errors and missing evaluation slots"
            )
            raise TimeoutError(receipt["error"])
    finally:
        checkpoints = []
        for name in receipt["expected_checkpoints"]:
            arm, update = (
                ("value", 0) if name == "shared-initialization" else name.rsplit("-", 1)
            )
            summary = (
                out / arm / arm / "drafts" / f"eval-{int(update):04d}" / "summary.json"
            )
            totals = {
                "planned": 32,
                "attempted": 0,
                "completed": 0,
                "failed": 0,
                "missing": 32,
            }
            if summary.exists():
                rows = json.loads(summary.read_text())
                totals = {key: sum(row[key] for row in rows) for key in totals}
            checkpoints.append({"checkpoint": name, **totals})
        receipt["checkpoints"] = checkpoints
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
    comparison.add_argument("--updates", type=int, default=1)
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
        failed = "preparation" if args.command == "prepare" else "comparison"
        raise SystemExit(f"No completed {failed}: {error}") from None


if __name__ == "__main__":
    main()
