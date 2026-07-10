"""Exp-08 (wave/rules Stage 4) runner: the first A-vs-B matchup table.

UR Lessons vs GW Allies (the Milestone-1 two-deck slice), seat-balanced, in
parallel across processes, with per-game decision-profile instrumentation
(exp-00 style). The "hero" side is ALWAYS the UR Lessons player; seat
balancing alternates which seat (play/draw) the UR player occupies, so the
hero win rate IS the UR per-deck win rate.

Usage:
    python -m manabot.verify.run_two_deck_matchup --games 400 --workers 8 \
        --out reports/data/exp-08-two-deck-matchup.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


def _worker(args: dict[str, Any]) -> dict[str, Any]:
    """Play a chunk of games in a subprocess; returns picklable results."""

    import torch

    torch.set_num_threads(1)
    from manabot.sim.flat_mc import play_games
    from manabot.verify.util import GW_ALLIES_DECK, UR_LESSONS_DECK

    result = play_games(
        args["hero_spec"],
        args["villain_spec"],
        num_games=args["num_games"],
        seed=args["seed"],
        game_offset=args["game_offset"],
        hero_deck=dict(UR_LESSONS_DECK),
        villain_deck=dict(GW_ALLIES_DECK),
    )
    return {
        "records": [
            {
                "game_index": r.game_index,
                "hero_seat": r.hero_seat,
                "hero_won": r.hero_won,
                "winner": r.winner,
                "steps": r.steps,
                "turns": r.turns,
                "hero_decisions": r.hero_decisions,
                "villain_decisions": r.villain_decisions,
            }
            for r in result.records
        ],
        "hero_search": result.hero_search.to_dict() if result.hero_search else None,
        "hero_decision_seconds": (
            result.hero_search.decision_seconds if result.hero_search else []
        ),
        "villain_search": (
            result.villain_search.to_dict() if result.villain_search else None
        ),
        "villain_decision_seconds": (
            result.villain_search.decision_seconds if result.villain_search else []
        ),
        "wall_seconds": result.wall_seconds,
    }


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _decision_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-deck decision profile: mean surfaced decisions per game by kind."""

    profile: dict[str, Any] = {}
    for side, label in (("hero_decisions", "ur"), ("villain_decisions", "gw")):
        kinds: set[str] = set()
        for record in records:
            kinds.update(record[side])
        by_kind = {
            kind: float(np.mean([record[side].get(kind, 0) for record in records]))
            for kind in sorted(kinds)
        }
        totals = [float(sum(record[side].values())) for record in records]
        profile[label] = {
            "decisions_per_game": _distribution(totals),
            "mean_by_kind": by_kind,
        }
    return profile


def run_matchup(
    hero_spec: dict[str, Any],
    villain_spec: dict[str, Any],
    *,
    num_games: int,
    workers: int,
    base_seed: int,
) -> dict[str, Any]:
    from manabot.sim.flat_mc import (
        GameRecord,
        aggregate_records,
        spec_name,
    )

    chunks = []
    per_worker = num_games // workers
    remainder = num_games % workers
    offset = 0
    for w in range(workers):
        chunk = per_worker + (1 if w < remainder else 0)
        if chunk == 0:
            continue
        chunks.append(
            {
                "hero_spec": hero_spec,
                "villain_spec": villain_spec,
                "num_games": chunk,
                "seed": base_seed + w * 1_000_000,
                "game_offset": offset,
            }
        )
        offset += chunk

    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        outputs = list(pool.map(_worker, chunks))
    wall = time.perf_counter() - start

    raw_records = [record for out in outputs for record in out["records"]]
    records = [GameRecord(**record) for record in raw_records]
    metrics = aggregate_records(records)

    # Per-deck win rates: hero == UR by construction; per-seat rates come
    # from aggregate_records (win_rate_on_play / win_rate_on_draw are the
    # UR player's). Also report the seat itself (seat-0 win rate regardless
    # of deck) so deck and seat effects can be separated.
    from manabot.sim.flat_mc import wilson_interval

    seat0_wins = sum(1 for r in records if r.winner == 0)
    decided = sum(1 for r in records if r.winner is not None)
    seat_lo, seat_hi = wilson_interval(seat0_wins, decided)
    metrics["on_play_win_rate_any_deck"] = seat0_wins / decided if decided else 0.0
    metrics["on_play_win_ci_lower"] = seat_lo
    metrics["on_play_win_ci_upper"] = seat_hi

    def merge_search(key: str, ds_key: str) -> dict[str, Any] | None:
        stats = [out[key] for out in outputs if out[key] is not None]
        if not stats:
            return None
        seconds = [s for out in outputs for s in out[ds_key]]
        total = {
            name: float(sum(s[name] for s in stats))
            for name in ("decisions", "seconds", "simulations", "cap_hits")
        }
        total["mean_seconds_per_decision"] = (
            total["seconds"] / total["decisions"] if total["decisions"] else 0.0
        )
        total["median_seconds_per_decision"] = (
            float(np.median(seconds)) if seconds else 0.0
        )
        return total

    return {
        "hero": f"{spec_name(hero_spec)} (UR Lessons)",
        "villain": f"{spec_name(villain_spec)} (GW Allies)",
        "hero_spec": hero_spec,
        "villain_spec": villain_spec,
        "metrics": metrics,
        "turns": _distribution([float(r.turns) for r in records]),
        "steps": _distribution([float(r.steps) for r in records]),
        "decision_profile": _decision_profile(raw_records),
        "hero_search": merge_search("hero_search", "hero_decision_seconds"),
        "villain_search": merge_search("villain_search", "villain_decision_seconds"),
        "wall_seconds": wall,
        "engine_seconds": float(sum(out["wall_seconds"] for out in outputs)),
    }


def build_matchups(games: int) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    """The pre-registered exp-08 matrix (hero = UR, villain = GW)."""

    search = lambda n: {"kind": "search", "sims": n}  # noqa: E731
    random_spec = {"kind": "random"}
    return [
        (random_spec, random_spec, games),  # raw deck advantage
        (search(16), search(16), games),
        (search(64), search(64), games),  # does search widen/narrow the gap?
        (search(64), random_spec, games),  # UR search vs GW random
        (random_spec, search(64), games),  # UR random vs GW search
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=str, default="reports/data/exp-08-two-deck-matchup.json"
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated list of matchup names (hero__villain) to run",
    )
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    from manabot.sim.flat_mc import spec_name

    only = set(args.only.split(",")) if args.only else None
    matchups = build_matchups(args.games)
    for index, (hero_spec, villain_spec, games) in enumerate(matchups):
        name = f"{spec_name(hero_spec)}__{spec_name(villain_spec)}"
        if only is not None and name not in only:
            continue
        if name in results:
            print(f"[skip] {name} (already in {out_path})")
            continue
        print(f"[run ] UR:{spec_name(hero_spec)} vs GW:{spec_name(villain_spec)}: "
              f"{games} games, {args.workers} workers")
        result = run_matchup(
            hero_spec,
            villain_spec,
            num_games=games,
            workers=args.workers,
            base_seed=args.seed + index * 97,
        )
        results[name] = result
        out_path.write_text(json.dumps(results, indent=2))
        m = result["metrics"]
        print(
            f"       UR win {m['win_rate']:.3f} "
            f"[{m['win_ci_lower']:.3f},{m['win_ci_upper']:.3f}] "
            f"play {m['win_rate_on_play']:.3f} draw {m['win_rate_on_draw']:.3f} "
            f"| turns {result['turns']['mean']:.1f} "
            f"| wall {result['wall_seconds']:.0f}s"
        )

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
