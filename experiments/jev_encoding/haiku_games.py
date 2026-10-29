"""Haiku or Jev against MTG-183's final checkpoint under a chosen encoding and history.

Arms share MTG-212's frozen deal schedule (seeds, seats, deck pairs), so arm k's
game i is the same deal and seat as every other arm's game i: comparisons are
paired by deal. Forced decisions are skipped natively. Each worker process owns
one `BudgetLedger` slice of the arm's budget, so spend stays capped while games
run in parallel. A failed game is retained unscored; no game is retried.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import json
from pathlib import Path
import random

import torch

from experiments.elemental_grid import DECKS, PACK
from experiments.jev_encoding.encodings import VARIANTS, DeckContext, chat_form
from manabot.infra.config import MatchConfig
from manabot.llm.budget import BudgetConfig, BudgetLedger
from manabot.llm.openrouter import OpenRouterConfig, OpenRouterSession
from manabot.llm.schema import DecisionInput
from manabot.semantic.text_observation import TextDecision
from manabot.verify.llm_game import CheckpointOpponent, GameConfig, play

CHECKPOINT = Path.home() / ".manabot/mtg212-jev/mtg183-f3748bcabac8-u5950.pt"
CHECKPOINT_SHA256 = "4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4"
CHECKPOINT_BYTES = 1_071_166
HAIKU = OpenRouterConfig(
    model_id="anthropic/claude-haiku-5.5",
    objective="win-game",
    max_output_tokens=8192,
    reasoning_effort="high",
    deadline_seconds=120,
    fallback_seed=37,
    cache_prompt=True,
    provider="Claude Platform on AWS",
)


def schedule(count: int) -> list[tuple[int, GameConfig]]:
    """MTG-212's frozen schedule (`jev_mtg183.schedule`), first `count` games."""
    games: list[tuple[int, tuple[str, str], int]] = []
    for model_deck in DECKS:
        for policy_deck in DECKS:
            for seat in (0, 1):
                for _ in range(5):
                    decks = (model_deck, policy_deck) if seat == 0 else (policy_deck, model_deck)
                    games.append((len(games), decks, seat))
    random.Random(212183).shuffle(games)
    return [
        (
            index,
            GameConfig(
                seed=212183000 + index,
                opponent_seed=212595000 + index,
                model_seat=seat,
                match=MatchConfig.authored(PACK, *decks),
                skip_trivial=True,
                max_decisions=2000,
                max_seconds=3600,
                max_reported_cost_usd=0.75,
            ),
        )
        for index, decks, seat in games[:count]
    ]


JEV = OpenRouterConfig.model_validate_json(Path("configs/llm/openrouter-jev.json").read_text())


def _encoder(variant: str, config: GameConfig, model: str = "haiku"):
    match = config.match
    assert match is not None
    own, other = (match.hero_deck, match.villain_deck) if config.model_seat == 0 else (match.villain_deck, match.hero_deck)
    decks = DeckContext(own=dict(own), opponent=dict(other))

    def encode(decision: TextDecision) -> DecisionInput:
        encoded = VARIANTS[variant](decision.capture_json, decks)
        if model == "jev":
            # Typed Decisions carry the question and choices themselves, exactly
            # as in the decision-level screen.
            text, prefix = encoded.text, None
        elif encoded.card_block is None:
            # Frozen-layout variants already list their labelled actions and
            # have no stable prefix to cache.
            text, prefix = encoded.text, None
        else:
            text, prefix = chat_form(encoded)
        descriptions = encoded.descriptions
        return DecisionInput(
            text=text,
            legal_references=tuple(a.reference for a in decision.actions),
            action_descriptions=descriptions,
            representation_id=f"jev-encoding-{variant}-{'decisions' if model == 'jev' else 'chat'}",
            source_identity={"capture_digest": decision.capture_digest},
            cache_prefix_chars=prefix,
        )

    return encode


_CACHEABLE = {"systematic", "player", "board", "library"}


def _inference(variant: str, model: str = "haiku") -> OpenRouterConfig:
    """Cache only encodings with a stable card-text prefix (chat_form)."""
    if model == "jev":
        return JEV
    return HAIKU.model_copy(update={"cache_prompt": variant.split("+")[0].split("@")[0] in _CACHEABLE})


def _worker(job: dict) -> list[dict]:
    torch.set_num_threads(1)
    opponent = CheckpointOpponent(CHECKPOINT, CHECKPOINT_SHA256, CHECKPOINT_BYTES)
    ledger = BudgetLedger(
        Path(job["ledger"]),
        BudgetConfig(
            ceiling_usd=Decimal(job["ceiling"]),
            request_bound_usd=Decimal("0.02"),
            overhead_usd=Decimal("0"),
            charge_multiplier=Decimal("1.06"),
            bound_evidence=(
                "MTG-212 evidence/billing-bound.json: Jev $0.042/M prompt, free output"
                if job["model"] == "jev"
                else "OpenRouter anthropic/claude-haiku-5.5 endpoints 2026-10-28: $0.10/M in, $0.50/M out, $0.125/M cache write; 8,192 max output"
            ),
        ),
    )
    results = []
    for index, config in job["games"]:
        config = GameConfig.model_validate(config).model_copy(update={"history_capacity": job["history"]})
        directory = Path(job["out"]) / f"game-{index:04d}"
        if directory.exists():
            continue  # attempted earlier; never replaced
        try:
            ledger.check_admission()
        except Exception as error:
            results.append({"game": index, "stopped": f"{type(error).__name__}: {error}"})
            break
        # Caching only pays when the encoding has a stable prefix.
        inference = _inference(job["variant"], job["model"])
        with OpenRouterSession(inference, budget=ledger, budget_owner=f"game-{index:04d}") as session:
            summary = play(config, directory, session, opponent=opponent, encoder=_encoder(job["variant"], config, job["model"]))
        (directory / "summary.json").write_text(summary.model_dump_json(indent=2))
        valid = summary.completed and summary.replay_matches and summary.failure is None
        results.append(
            {
                "game": index,
                "seat": config.model_seat,
                "valid": valid,
                "win": valid and summary.winner == config.model_seat,
                "failure": summary.failure,
                "model_decisions": summary.model_decisions,
                "prompt_tokens": summary.prompt_tokens,
                "cached_tokens": summary.cached_tokens,
                "cache_write_tokens": summary.cache_write_tokens,
                "completion_tokens": summary.completion_tokens,
                "cost": summary.reported_cost_usd,
                "unpriced": summary.unpriced_attempts,
            }
        )
        with open(Path(job["out"]) / "results.jsonl", "a") as sink:
            sink.write(json.dumps(results[-1]) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument("--history", type=int, choices=[0, 32], required=True)
    parser.add_argument("--model", choices=["haiku", "jev"], default="haiku")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--first", type=int, default=0, help="skip this many schedule games")
    parser.add_argument("--only", type=int, nargs="+", help="play exactly these schedule games (a marked retry pass)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget", required=True, help="USD for this arm, split across workers")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    games = schedule(args.first + args.games)[args.first:]
    if args.only:
        games = [(i, c) for i, c in schedule(1000) if i in set(args.only)]
    (args.out / "arm.json").write_text(
        json.dumps({"variant": args.variant, "history": args.history, "games": [i for i, _ in games], "model": args.model, "inference": _inference(args.variant, args.model).model_dump(mode="json"), "budget": args.budget, "workers": args.workers}, indent=2)
    )
    share = (Decimal(args.budget) / args.workers).quantize(Decimal("0.0001"))
    jobs = [
        {
            "variant": args.variant,
            "model": args.model,
            "history": args.history,
            "out": str(args.out),
            "ledger": str(args.out / f"spend-{worker}.sqlite"),
            "ceiling": str(share),
            "games": [(i, c.model_dump(mode="json")) for i, c in games[worker :: args.workers]],
        }
        for worker in range(args.workers)
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        results = [r for batch in pool.map(_worker, jobs) for r in batch]
    played = [r for r in results if "valid" in r]
    print(json.dumps({"played": len(played), "wins": sum(r["win"] for r in played), "valid": sum(r["valid"] for r in played), "cost": sum(r["cost"] for r in played), "stops": [r for r in results if "stopped" in r]}, indent=2))


if __name__ == "__main__":
    main()
