"""Local checkpoint comparisons with bounded preparation and CPU execution.

checkpoint_request admits local policies into the shared EvaluationRequest API.
The CLI performs that preparation in a killable process, then spends the same
absolute deadline on evaluate_local. It never starts callback transport or loads
an Etude checkpoint record that could fetch remote bytes.
"""

import argparse
import math
from multiprocessing.connection import Connection
from pathlib import Path
import time

import torch

from manabot.backends import resolve
from manabot.eval.admission import (
    OPPONENT_NAMES,
    AnchorKind,
    register_anchor,
    register_checkpoint,
)
from manabot.eval.evaluation import (
    EvaluationRequest,
    EvaluationResult,
    continue_evaluation,
    evaluate_local,
    evaluation_status,
)
from manabot.eval.models import EvaluationKey
from manabot.infra.artifacts import StoredArtifact, materialize_artifact
from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.policy.binding import BindingReport
from manabot.policy.protocol import Backend
from manabot.sim.child import ChildProcess
from manabot.sim.matchup import (
    SELECTED_SUITE,
    AuthoredMatchup,
    DeckMatchup,
    selected_match,
)


class LocalComparison(StrictModel):
    request: EvaluationRequest
    report: BindingReport | None


def checkpoint_request(
    checkpoint: Path,
    *,
    backend: Backend,
    opponent: Path | AnchorKind = "heuristic",
    deal_seeds: tuple[int, ...] = (101,),
    authored_matchup: AuthoredMatchup | DeckMatchup | None = None,
    include_mirrors: bool = False,
    heuristic_actions: int | None = None,
    game_seconds: float = 120,
    max_commands: int = 10_000,
) -> EvaluationRequest:
    """Prepare a small comparison from local bytes, without producer metadata.

    Preparation loads policies to admit their architecture, world and actual
    setup. CLI preparation is supervised; API callers may prepare requests ahead
    of allocating execution. Registered bytes are verified again by execution.
    """
    return local_comparison(
        checkpoint,
        backend=backend,
        opponent=opponent,
        deal_seeds=deal_seeds,
        authored_matchup=authored_matchup,
        include_mirrors=include_mirrors,
        heuristic_actions=heuristic_actions,
        game_seconds=game_seconds,
        max_commands=max_commands,
    ).request


def local_comparison(
    checkpoint: Path,
    *,
    backend: Backend,
    opponent: Path | AnchorKind = "heuristic",
    deal_seeds: tuple[int, ...] = (101,),
    authored_matchup: AuthoredMatchup | DeckMatchup | None = None,
    include_mirrors: bool = False,
    heuristic_actions: int | None = None,
    game_seconds: float = 120,
    max_commands: int = 10_000,
) -> LocalComparison:
    match = authored_matchup.match() if authored_matchup else selected_match()
    suite = authored_matchup.content_suite if authored_matchup else SELECTED_SUITE
    admitted = register_checkpoint(
        checkpoint,
        backend=backend,
        match=match,
        content_suite=suite,
        player_id="local-candidate",
        display_name="Local candidate",
        allow_deck_repetition=include_mirrors,
    )
    candidate = admitted.registration
    paths = {candidate.player_id: str(checkpoint.resolve())}
    if isinstance(opponent, Path):
        other = register_checkpoint(
            opponent,
            backend=backend,
            match=match,
            content_suite=suite,
            player_id="local-opponent",
            display_name="Local opponent",
            allow_deck_repetition=include_mirrors,
        ).registration
        paths[other.player_id] = str(opponent.resolve())
    else:
        other = register_anchor(
            candidate,
            opponent,
            player_id="local-" + opponent.replace("_", "-"),
            heuristic_actions=heuristic_actions,
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
    request = EvaluationRequest(
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
    return LocalComparison(request=request, report=admitted.report)


def _prepare(
    sender: Connection,
    checkpoint: Path | StoredArtifact,
    opponent: Path | AnchorKind,
    seeds: tuple[int, ...],
    matchup: AuthoredMatchup | DeckMatchup | None,
    mirrors: bool,
    game_seconds: float,
    max_commands: int,
    heuristic_actions: int | None,
) -> None:
    try:
        if isinstance(checkpoint, StoredArtifact):
            checkpoint = materialize_artifact(checkpoint)
        torch.set_num_threads(1)
        comparison = local_comparison(
            checkpoint,
            backend=resolve("torch"),
            opponent=opponent,
            deal_seeds=seeds,
            authored_matchup=matchup,
            include_mirrors=mirrors,
            heuristic_actions=heuristic_actions,
            game_seconds=game_seconds,
            max_commands=max_commands,
        )
        sender.send((comparison.model_dump_json(), None))
    except Exception as exc:
        sender.send((None, f"{type(exc).__name__}: {exc}"))
    finally:
        sender.close()


def prepare_local_request(
    checkpoint: Path | StoredArtifact,
    opponent: Path | AnchorKind,
    seeds: tuple[int, ...],
    matchup: AuthoredMatchup | DeckMatchup | None,
    *,
    deadline: float,
    mirrors: bool = False,
    game_seconds: float = 120,
    max_commands: int = 10_000,
    heuristic_actions: int | None = None,
) -> LocalComparison:
    """Resolve and admit checkpoint bytes within the remaining allocation."""
    process = ChildProcess(
        "manabot.eval.local:_prepare",
        (
            checkpoint,
            opponent,
            seeds,
            matchup,
            mirrors,
            game_seconds,
            max_commands,
            heuristic_actions,
        ),
    )
    try:
        if not process.connection.poll(max(0, deadline - time.time())):
            raise TimeoutError(
                "allocation deadline exhausted during checkpoint admission"
            )
        payload, error = process.connection.recv()
        if error:
            raise ValueError(error)
        return LocalComparison.model_validate_json(payload)
    finally:
        process.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Local policy .pt file")
    source.add_argument(
        "--request", type=Path, help="Previously admitted EvaluationRequest JSON"
    )
    source.add_argument(
        "--continue",
        dest="continuation",
        type=Path,
        help="Evaluation panel directory whose missing blocks to finish",
    )
    source.add_argument(
        "--status", type=Path, help="Inspect retained evaluation progress"
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
    parser.add_argument(
        "--out", type=Path, help="New evaluation directory (fresh repeat)"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=300,
        help="Whole lifetime including preparation, play and replay",
    )
    parser.add_argument("--game-seconds", type=float, default=120)
    parser.add_argument("--max-commands", type=int, default=10_000)
    args = parser.parse_args(argv)
    if (args.request or args.continuation or args.status) and (
        args.opponent != "heuristic"
        or args.deal_seed
        or args.matchup
        or args.include_mirrors
        or args.game_seconds != 120
        or args.max_commands != 10_000
    ):
        parser.error(
            "saved evaluation already fixes opponents, deals, matchup and game limits"
        )
    if args.continuation or args.status:
        if args.out:
            parser.error("--out is only for a fresh evaluation")
        directory = args.continuation or args.status
        try:
            result = evaluation_status(directory)
            if args.continuation:
                if not math.isfinite(args.seconds) or args.seconds <= 0:
                    parser.error("--seconds must be positive")
                result = continue_evaluation(
                    directory, deadline=time.time() + args.seconds
                )
            return _describe(result, directory)
        except Exception as exc:
            print(f"incomplete: {exc}; evidence: {directory}")
            return 1
    if args.out is None:
        parser.error("a fresh evaluation requires --out")
    deadline = time.time() + args.seconds
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out / "limits.json", {"deadline_unix": deadline})
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
            comparison = prepare_local_request(
                args.checkpoint,
                opponent,
                tuple(args.deal_seed or [101]),
                matchup,
                deadline=deadline,
                mirrors=args.include_mirrors,
                game_seconds=args.game_seconds,
                max_commands=args.max_commands,
            )
            request = comparison.request
        atomic_json(args.out / "request.json", request.model_dump(mode="json"))
        result = evaluate_local(request, deadline=deadline, out_dir=args.out / "panel")
    except Exception as exc:
        atomic_json(
            args.out / "admission-failure.json",
            {
                "status": "incomplete",
                "error": f"{type(exc).__name__}: {exc}",
                "execution": {"deadline_unix": deadline},
            },
        )
        print(f"incomplete: {exc}; evidence: {args.out}")
        return 1
    return _describe(result, args.out / "panel")


def _describe(result: EvaluationResult, directory: Path) -> int:
    request = result.request
    print(
        f"{result.status}: {len(result.complete_deals())}/{len(request.deal_seeds)} complete deals; "
        f"{len(result.rows)}/{request.expected_games} games; evidence: {directory}"
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
