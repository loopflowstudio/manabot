"""Local checkpoint comparisons with bounded preparation and CPU execution.

checkpoint_request admits local policies into the shared EvaluationRequest API.
The CLI performs that preparation in a killable process, then spends the same
absolute deadline on evaluate_local. It never starts callback transport or loads
an Etude checkpoint record that could fetch remote bytes.
"""

import argparse
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import time

import torch

from manabot.eval.admission import (
    OPPONENT_NAMES,
    AnchorKind,
    register_anchor,
    register_checkpoint,
)
from manabot.eval.evaluation import EvaluationRequest, evaluate_local
from manabot.eval.models import EvaluationKey
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.sim.execution import EvalExecution
from manabot.sim.matchup import SELECTED_SUITE, AuthoredMatchup, selected_match


def checkpoint_request(
    checkpoint: Path,
    *,
    opponent: Path | AnchorKind = "heuristic",
    deal_seeds: tuple[int, ...] = (101,),
    authored_matchup: AuthoredMatchup | None = None,
    include_mirrors: bool = False,
    game_seconds: float = 120,
    max_commands: int = 10_000,
) -> EvaluationRequest:
    """Prepare a small comparison from local bytes, without producer metadata.

    Preparation loads policies to admit their architecture, world and actual
    setup. CLI preparation is supervised; API callers may prepare requests ahead
    of allocating execution. Registered bytes are verified again by execution.
    """
    match = authored_matchup.match() if authored_matchup else selected_match()
    suite = authored_matchup.content_suite if authored_matchup else SELECTED_SUITE
    candidate = register_checkpoint(
        checkpoint,
        match=match,
        content_suite=suite,
        player_id="local-candidate",
        display_name="Local candidate",
        allow_deck_repetition=include_mirrors,
    ).registration
    paths = {candidate.player_id: str(checkpoint.resolve())}
    if isinstance(opponent, Path):
        other = register_checkpoint(
            opponent,
            match=match,
            content_suite=suite,
            player_id="local-opponent",
            display_name="Local opponent",
            allow_deck_repetition=include_mirrors,
        ).registration
        paths[other.player_id] = str(opponent.resolve())
    else:
        other = register_anchor(
            candidate, opponent, player_id="local-" + opponent.replace("_", "-")
        )
    key = EvaluationKey(
        world=candidate.world,
        content_suite=suite,
        viewer_boundary="acting-viewer",
        arena_version="local-evaluation-v1",
        rating_model_version="unrated-local-check",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(other.model_dump()),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    return EvaluationRequest(
        key=key,
        candidate=candidate,
        opponent=other,
        checkpoint_paths=paths,
        deal_seeds=deal_seeds,
        authored_matchup=authored_matchup,
        matchup_mode="all" if include_mirrors else "cross",
        game_seconds=game_seconds,
        max_commands=max_commands,
    )


def _prepare(
    sender: Connection,
    checkpoint: Path,
    opponent: Path | AnchorKind,
    seeds: tuple[int, ...],
    matchup: AuthoredMatchup | None,
    mirrors: bool,
    game_seconds: float,
    max_commands: int,
) -> None:
    try:
        torch.set_num_threads(1)
        request = checkpoint_request(
            checkpoint,
            opponent=opponent,
            deal_seeds=seeds,
            authored_matchup=matchup,
            include_mirrors=mirrors,
            game_seconds=game_seconds,
            max_commands=max_commands,
        )
        sender.send((request.model_dump_json(), None))
    except Exception as exc:
        sender.send((None, f"{type(exc).__name__}: {exc}"))
    finally:
        sender.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Local policy .pt file")
    source.add_argument(
        "--request", type=Path, help="Previously admitted EvaluationRequest JSON"
    )
    parser.add_argument(
        "--opponent",
        default="heuristic",
        help="heuristic, scripted_greedy, random, or a local policy .pt",
    )
    parser.add_argument(
        "--deal-seed",
        type=int,
        action="append",
        help="Explicit deal seed; repeat for multiple complete blocks (default 101)",
    )
    parser.add_argument(
        "--matchup",
        type=Path,
        help="AuthoredMatchup JSON; default selected Allies/Lessons setup",
    )
    parser.add_argument("--include-mirrors", action="store_true")
    parser.add_argument("--out", type=Path, required=True, help="New attempt directory")
    parser.add_argument(
        "--seconds",
        type=float,
        default=300,
        help="Whole lifetime including preparation, play and replay",
    )
    parser.add_argument("--game-seconds", type=float, default=120)
    parser.add_argument("--max-commands", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-rows", type=int, default=16)
    parser.add_argument("--resident-mib", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.request and (
        args.opponent != "heuristic"
        or args.deal_seed
        or args.matchup
        or args.include_mirrors
        or args.game_seconds != 120
        or args.max_commands != 10_000
    ):
        parser.error(
            "--request already fixes opponents, deals, matchup and game limits"
        )
    # EvalExecution rejects nonfinite/invalid budgets before creating output.
    limits = EvalExecution(
        device="cpu",
        deadline_unix=time.time() + args.seconds,
        max_active_games=args.workers,
        max_batch_rows=args.batch_rows,
        resident_model_bytes=args.resident_mib * 2**20,
    )
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out / "limits.json", limits.model_dump(mode="json"))
    try:
        if args.request:
            request = EvaluationRequest.model_validate_json(args.request.read_bytes())
        else:
            matchup = (
                AuthoredMatchup.model_validate_json(args.matchup.read_bytes())
                if args.matchup
                else None
            )
            opponent: Path | AnchorKind = (
                # Narrow the CLI string at the boundary; no URI resolution.
                args.opponent
                if args.opponent in OPPONENT_NAMES
                else Path(args.opponent)
            )
            context = mp.get_context("spawn")
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_prepare,
                args=(
                    sender,
                    args.checkpoint,
                    opponent,
                    tuple(args.deal_seed or [101]),
                    matchup,
                    args.include_mirrors,
                    args.game_seconds,
                    args.max_commands,
                ),
            )
            try:
                process.start()
                sender.close()
                if not receiver.poll(limits.remaining()):
                    raise TimeoutError(
                        "allocation deadline exhausted during checkpoint admission"
                    )
                payload, error = receiver.recv()
                if error:
                    raise ValueError(error)
                request = EvaluationRequest.model_validate_json(payload)
            finally:
                if process.is_alive():
                    process.kill()
                if process.pid is not None:
                    process.join()
                sender.close()
                receiver.close()
        atomic_json(args.out / "request.json", request.model_dump(mode="json"))
        result = evaluate_local(request, limits=limits, out_dir=args.out / "panel")
    except Exception as exc:
        atomic_json(
            args.out / "admission-failure.json",
            {
                "status": "incomplete",
                "error": f"{type(exc).__name__}: {exc}",
                "execution": limits.model_dump(mode="json"),
            },
        )
        print(f"incomplete: {exc}; evidence: {args.out}")
        return 1
    print(
        f"{result.status}: {len(result.rows)}/{request.expected_games} games; evidence: {args.out}"
    )
    if result.score is not None:
        deals = len(request.deal_seeds)
        # Resampling one correlated deal cannot estimate between-deal uncertainty.
        # Keep the shared/frozen statistics intact; qualify the local presentation.
        uncertainty = (
            "1 deal block; uncertainty unavailable from a single deal"
            if deals == 1
            else f"{deals} deal blocks; paired-deal 95% bootstrap interval "
            f"{result.score.lower:.3f}–{result.score.upper:.3f}"
        )
        print(f"score: {result.score.mean:.3f} ({uncertainty}); {request.purpose}")
    if result.error:
        print(result.error)
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
