"""Ask Jev the same labelled positions under each encoding variant.

Positions come from `label.py` (MTG-212 campaign-3 Jev decisions, with the
checkpoint's distribution). Each (position, variant) pair is asked once and
appended to the results file; reruns skip pairs already answered. Every request
is admitted through a `BudgetLedger` before dispatch.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
from pathlib import Path
import random
import threading

from manabot.llm.budget import BudgetConfig, BudgetLedger
from manabot.llm.openrouter import OpenRouterConfig, OpenRouterSession
from manabot.llm.schema import DecisionInput

from experiments.jev_encoding.encodings import VARIANTS, DeckContext

CAMPAIGN = Path("/Users/jack/src/manabot.mtg-212-benchmark-mtg-183/experiments/data/mtg212-jev/campaign-3")


def sample_positions(labels: Path, count: int, seed: int, threshold: float) -> list[dict]:
    rows = [json.loads(line) for line in labels.open()]
    confident = [r for r in rows if max(r["checkpoint"].values()) >= threshold]
    return random.Random(seed).sample(confident, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("data/labels.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=974)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ledger", type=Path, default=Path("data/spend.sqlite"))
    parser.add_argument("--ceiling", default="5")
    parser.add_argument("--tag", default="", help="suffix recorded on variant names (a changed encoding)")
    args = parser.parse_args()

    config = OpenRouterConfig.model_validate_json(Path("configs/llm/openrouter-jev.json").read_text())
    ledger = BudgetLedger(
        args.ledger,
        BudgetConfig(
            ceiling_usd=Decimal(args.ceiling),
            request_bound_usd=Decimal("0.02"),
            overhead_usd=Decimal("0"),
            charge_multiplier=Decimal("1.06"),
            bound_evidence="MTG-212 evidence/billing-bound.json: $0.042/M prompt, free output, 262,144-token context",
        ),
    )
    done = set()
    if args.out.exists():
        done = {(r["game"], r["step"], r["variant"].removesuffix(args.tag) if args.tag else r["variant"]) for r in map(json.loads, args.out.open())}
    positions = sample_positions(args.labels, args.count, args.seed, args.threshold)
    # Position-major: every variant answers a position before the next one, so
    # partial results stay paired; variant order is shuffled within a position.
    rng = random.Random(args.seed + 1)
    work = []
    for p in positions:
        variants = [v for v in args.variants.split(",") if (p["game"], p["step"], v) not in done]
        rng.shuffle(variants)
        work += [(p, v) for v in variants]
    lock = threading.Lock()
    print(f"{len(work)} requests to send", flush=True)

    def ask(item: tuple[dict, str]) -> None:
        position, variant = item
        game = CAMPAIGN / position["game"]
        config_json = json.loads((game / "config.json").read_text())
        match = config_json["match"]
        seat = config_json["model_seat"]
        decks = DeckContext(
            own=match["villain_deck"] if seat else match["hero_deck"],
            opponent=match["hero_deck"] if seat else match["villain_deck"],
        )
        capture = (game / position["step"] / "capture.json").read_text()
        encoded = VARIANTS[variant](capture, decks)
        assert list(encoded.descriptions) == list(position["kinds"])
        decision = DecisionInput(
            text=encoded.text,
            legal_references=tuple(encoded.descriptions),
            action_descriptions=encoded.descriptions,
            representation_id=f"jev-encoding-{variant}{args.tag}",
            source_identity={"capture": f"{position['game']}/{position['step']}"},
        )
        with OpenRouterSession(config, budget=ledger, budget_owner=variant) as session:
            result = session.decide(decision)
        attempt = result.attempts[-1] if result.attempts else None
        reply = attempt.decision_reply if attempt else None
        answer = reply.answers["action"] if reply else None
        record = {
            "game": position["game"],
            "step": position["step"],
            "variant": variant + args.tag,
            "chars": len(encoded.text),
            "choice": result.selected_reference,
            "probabilities": getattr(answer, "probabilities", None),
            "confidence": getattr(answer, "confidence", None),
            "prompt_tokens": attempt.prompt_tokens if attempt else None,
            "cost": str(reply.usage.cost) if reply and reply.usage else None,
            "seconds": result.seconds,
            "failure": result.failure.model_dump_json() if result.failure else None,
        }
        with lock, args.out.open("a") as sink:
            sink.write(json.dumps(record) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        for index, _ in enumerate(pool.map(ask, work)):
            if index % 100 == 0:
                spend = ledger.snapshot(); print(index, "accounted", spend.accounted_usd, "pending", spend.pending_usd, flush=True)
    spend = ledger.snapshot(); print("done accounted", spend.accounted_usd, "pending", spend.pending_usd)


if __name__ == "__main__":
    main()
