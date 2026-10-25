"""Card evaluation CLI: evaluate, rank, export-log, publish, and bounded verify.

Use the evaluation extra. Every evaluation has an explicit run ID and admitted
registrations; no command selects a checkpoint or remote destination implicitly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

from etude.run.storage import storage_from_uri

from manabot.eval.arena_log import CardMapping, export_arena_log
from manabot.eval.card_evaluation import (
    CardEvaluationSpec,
    evaluate_cards,
    publish_cards,
)
from manabot.eval.card_query import CardQuery, evidence_refs, load_game, rank_cards
from manabot.eval.evaluate import source_digest
from manabot.eval.models import EvaluationKey
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.sim.matchup import AuthoredMatchup
from manabot.sim.registration import PlayerRegistration
from manabot.sim.replay import replay_games


def verification_spec(games: int) -> CardEvaluationSpec:
    if games < 8 or games % 8:
        raise ValueError("games must be a positive multiple of eight")
    matrix = AuthoredMatchup(
        pack_key="elemental-world-v2", role_a="elemental_ug", role_b="elemental_bg"
    )
    runtime = runtime_fingerprints(match_config=matrix.match())
    players = tuple(
        PlayerRegistration(
            player_id=name,
            display_name=name,
            role="anchor",
            runner_kind="code",
            player_spec={"kind": "scripted_greedy"},
            compute_class_id="scripted-greedy-cpu-v1",
            information_boundary="acting-viewer",
            world=runtime["world"],
            content_suite=matrix.content_suite,
            observation_abi_sha256=runtime["observation_abi_sha256"],
            action_abi_sha256=runtime["action_abi_sha256"],
            matchup_sha256=runtime["matchup_sha256"],
            source_sha256=source_digest(["manabot/sim/players.py"]),
            player_seed_derivation_id="arena-pair-deal-player-v1",
        )
        for name in ("verify-one", "verify-two")
    )
    return CardEvaluationSpec(
        run=f"verify-{games}",
        key=EvaluationKey(
            world=runtime["world"],
            content_suite=matrix.content_suite,
            viewer_boundary="acting-viewer",
            arena_version="card-evaluation-v1",
            rating_model_version="unrated-card-evaluation",
            rating_prior_sha256=canonical_sha256({}),
            anchor_cohort_sha256=canonical_sha256([p.model_dump() for p in players]),
            evaluation_compute_envelope_id="scripted-greedy-cpu-v1",
        ),
        players=players,
        deals=tuple(range(230000, 230000 + games // 8)),
        authored_matchup=matrix,
        game_seconds=60,
        max_commands=10000,
    )


def verify(games: int, out: Path) -> dict[str, int | float | str]:
    started = time.monotonic()
    spec = verification_spec(games)
    evaluate_cards(spec, out)
    queries = [CardQuery(run=spec.run, player=p.player_id) for p in spec.players]
    # Each subject query rebuilds its rows; both participants must be checked.
    before = [rank_cards(out, query) for query in queries]
    refs = evidence_refs(out, spec.run)
    if len(refs) != games:
        raise ValueError("verification schedule did not retain every game")
    for ref in refs:
        game = load_game(out, spec.run, ref)
        if not replay_games([game.trace.data]).passed:
            raise ValueError(f"retained trace no longer verifies: {game.game}")
    if [rank_cards(out, query) for query in queries] != before:
        raise ValueError("ranking changed when rebuilt from evidence")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "games": games,
        "elapsed_seconds": time.monotonic() - started,
        "artifact_bytes": sum(ref.bytes for ref in refs),
        "peak_parent_memory_bytes": int(
            usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        ),
        "peak_child_memory_bytes": int(
            children.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        ),
        "run": spec.run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--spec", type=Path, required=True)
    evaluation.add_argument("--out", type=Path, required=True)
    ranking = commands.add_parser("rank")
    ranking.add_argument("--source", required=True)
    ranking.add_argument("--run", required=True)
    ranking.add_argument("--player", required=True)
    ranking.add_argument("--opponent", action="append", default=[])
    ranking.add_argument("--deck")
    ranking.add_argument("--opponent-deck")
    ranking.add_argument("--position", choices=["play", "draw"])
    ranking.add_argument("--partial", action="store_true")
    ranking.add_argument("--json", action="store_true")
    export = commands.add_parser("export-log")
    export.add_argument("--source", required=True)
    export.add_argument("--run", required=True)
    export.add_argument("--game", required=True)
    export.add_argument("--viewer", type=int, choices=[0, 1], required=True)
    export.add_argument("--mapping", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    publication = commands.add_parser("publish")
    publication.add_argument("--source", type=Path, required=True)
    publication.add_argument("--destination", required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("--games", type=int, default=1024)
    verification.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate_cards(
            CardEvaluationSpec.model_validate_json(args.spec.read_bytes()), args.out
        )
        print(result.model_dump_json(indent=2))
    elif args.command == "rank":
        result = rank_cards(
            args.source,
            CardQuery(
                run=args.run,
                player=args.player,
                opponents=tuple(args.opponent),
                deck=args.deck,
                opponent_deck=args.opponent_deck,
                on_play=None if args.position is None else args.position == "play",
                partial=args.partial,
            ),
        )
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print(
                f"{result.subject} / {result.selection.run}: {result.completeness.model_dump()}"
            )
            print(
                f"opponent mix: {result.opponent_mix}; excluded: {result.excluded_outcomes}"
            )
            print("card\tGIH WR\tcopies\tgames\t95% deal interval")
            for card in result.cards:
                rate = "unavailable" if card.rate is None else f"{card.rate:.3%}"
                print(
                    f"{card.name}\t{rate}\t{card.qualifying_copies}\t{card.games}\t{card.interval}"
                )
    elif args.command == "export-log":
        refs = [r for r in evidence_refs(args.source, args.run) if r.game == args.game]
        if len(refs) != 1:
            raise ValueError("game must identify exactly one retained artifact")
        log = export_arena_log(
            load_game(args.source, args.run, refs[0]),
            args.viewer,
            CardMapping.model_validate_json(args.mapping.read_bytes()),
        )
        args.out.write_text(log)
    elif args.command == "publish":
        publish_cards(args.source, storage_from_uri(args.destination))
    else:
        print(json.dumps(verify(args.games, args.out), indent=2))


if __name__ == "__main__":
    main()
