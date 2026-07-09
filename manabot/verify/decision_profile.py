"""Decision-profile measurement harness (experiment cycle C0, E0a/E0c).

Plays full games through the manabot Env and records, per game:

- surfaced decisions split by ActionSpaceKind and by player (hero/villain)
- total surfaced decisions
- skip_trivial_count (trivial decision points collapsed by the engine)
- collapse ratio = skipped / (skipped + surfaced)
- game length in turns
- winner (for the E0c baseline win-rate matrix)

Evaluation is seat-balanced by default (C0.5 protocol amendment A1): the hero
alternates between player seat 0 (on the play) and seat 1 (on the draw), and
win rates are reported overall and per seat. Multi-init untrained baselines
(amendment A2) via --untrained-inits / --baseline-inits.

Run as a script:

    python -m manabot.verify.decision_profile \
        --games-random 400 --games-untrained 400 --games-baseline 400 \
        --untrained-inits 5 --baseline-inits 3 --deck standard \
        --seed 0 --out scratch/exp00c-standard.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable

import numpy as np
import torch

from manabot.env import (
    Env,
    Match,
    ObservationSpace,
    Reward,
    build_opponent_policy,
)
from manabot.env.single_agent_env import RandomPolicy

from .util import (
    INTERACTIVE_DECK,
    STANDARD_DECK,
    build_hypers,
    step_with_fallback,
    suppress_truncation_logs,
    winner_from_info_or_obs,
)

HERO = 0
VILLAIN = 1

DECKS = {
    "standard": STANDARD_DECK,
    "interactive": INTERACTIVE_DECK,
}

# managym ActionSpaceKind (managym/src/agent/action.rs)
ACTION_SPACE_KIND_NAMES = {
    0: "game_over",
    1: "priority",
    2: "declare_attacker",
    3: "declare_blocker",
    4: "choose_target",
}
DECISION_KINDS = ("priority", "declare_attacker", "declare_blocker", "choose_target")


@dataclass
class GameProfile:
    """Per-game decision-profile record."""

    game_index: int
    winner: int | None  # player index (0 = on the play), NOT hero/villain role
    hero_seat: int  # player index the hero occupied this game (0 = on the play)
    turns: int
    surfaced_total: int
    surfaced_hero: int
    surfaced_villain: int
    skipped: int
    collapse_ratio: float
    # kind name -> count, per player
    hero_by_kind: dict[str, int] = field(default_factory=dict)
    villain_by_kind: dict[str, int] = field(default_factory=dict)
    # sanity: surfaced decisions with <=1 valid encoded action (should be ~0)
    single_valid_decisions: int = 0
    aborted: bool = False


def _untrained_agent(observation_space: ObservationSpace, hypers, seed: int):
    from manabot.model import Agent

    torch.manual_seed(seed)
    agent = Agent(observation_space, hypers.agent)
    agent.eval()
    return agent


def _agent_policy(agent) -> Callable[[dict[str, np.ndarray]], int]:
    def policy(obs: dict[str, np.ndarray]) -> int:
        tensor_obs = {
            key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
            for key, value in obs.items()
        }
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(
                tensor_obs, deterministic=False
            )
        return int(action.item())

    return policy


def build_hero_policy(
    name: str,
    observation_space: ObservationSpace,
    hypers,
    seed: int,
) -> Callable[[dict[str, np.ndarray]], int]:
    """Hero policies: 'random' (uniform over valid) or 'untrained' (fresh Agent)."""

    if name == "random":
        return RandomPolicy()
    if name == "untrained":
        return _agent_policy(_untrained_agent(observation_space, hypers, seed))
    raise ValueError(f"Unsupported hero policy: {name}")


def play_profile_games(
    *,
    hero_policy_name: str,
    villain_policy_name: str,
    num_games: int,
    seed: int = 0,
    max_decisions_per_game: int = 5000,
    hero_deck: dict[str, int] | None = None,
    villain_deck: dict[str, int] | None = None,
    seat_balanced: bool = False,
    agent_seed: int | None = None,
) -> list[GameProfile]:
    """Play games and record the decision profile for each one.

    With ``seat_balanced=True`` the hero occupies player seat 0 (on the play)
    in even-indexed games and seat 1 (on the draw) in odd-indexed games, by
    swapping the player configs handed to the engine — the engine itself
    always starts player 0. ``agent_seed`` controls the untrained-agent torch
    init independently of the game seed (multi-init baselines).
    """

    hero_deck = STANDARD_DECK if hero_deck is None else hero_deck
    villain_deck = STANDARD_DECK if villain_deck is None else villain_deck
    hypers = build_hypers(
        match={"hero_deck": hero_deck, "villain_deck": villain_deck}
    )
    observation_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    match_swapped = match.swapped()
    reward = Reward(hypers.reward)

    env = Env(
        match,
        observation_space,
        reward,
        seed=seed,
        auto_reset=False,
        enable_profiler=False,
        enable_behavior_tracking=False,
    )
    hero_policy = build_hero_policy(
        hero_policy_name,
        observation_space,
        hypers,
        seed if agent_seed is None else agent_seed,
    )
    villain_policy = build_opponent_policy(villain_policy_name)
    if villain_policy is None:
        raise ValueError("villain policy must not be 'none'")

    np.random.seed(seed)

    profiles: list[GameProfile] = []
    for game_index in range(num_games):
        hero_seat = game_index % 2 if seat_balanced else 0
        obs, _ = env.reset(
            seed=seed + game_index,
            options={"match": match_swapped} if hero_seat == 1 else None,
        )
        done = False
        aborted = False
        info: dict[str, Any] = {}
        counts = {
            HERO: {kind: 0 for kind in DECISION_KINDS},
            VILLAIN: {kind: 0 for kind in DECISION_KINDS},
        }
        single_valid = 0
        surfaced = 0

        while not done and surfaced < max_decisions_per_game:
            raw = env.last_raw_obs
            player = int(raw.agent.player_index)
            kind_id = int(raw.action_space.action_space_type)
            kind = ACTION_SPACE_KIND_NAMES.get(kind_id, f"unknown_{kind_id}")
            if kind == "game_over":
                break
            role = HERO if player == hero_seat else VILLAIN
            if kind in DECISION_KINDS:
                counts[role][kind] += 1
            surfaced += 1
            if int(np.flatnonzero(obs["actions_valid"] > 0).shape[0]) <= 1:
                single_valid += 1

            if role == HERO:
                action = hero_policy(obs)
            else:
                action = villain_policy(obs)

            try:
                obs, _, terminated, truncated, info = step_with_fallback(env, action)
            except Exception:
                aborted = True
                break
            done = bool(terminated or truncated)

        skipped = int(env.skip_trivial_count())
        turns = int(env.last_raw_obs.turn.turn_number)
        winner = None if aborted else winner_from_info_or_obs(info, env.last_raw_obs)
        hero_total = sum(counts[HERO].values())
        villain_total = sum(counts[VILLAIN].values())
        total = hero_total + villain_total
        denominator = skipped + total
        profiles.append(
            GameProfile(
                game_index=game_index,
                winner=winner,
                hero_seat=hero_seat,
                turns=turns,
                surfaced_total=total,
                surfaced_hero=hero_total,
                surfaced_villain=villain_total,
                skipped=skipped,
                collapse_ratio=(skipped / denominator) if denominator > 0 else 0.0,
                hero_by_kind=dict(counts[HERO]),
                villain_by_kind=dict(counts[VILLAIN]),
                single_valid_decisions=single_valid,
                aborted=aborted,
            )
        )

    env.close()
    return profiles


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    """Mean and normal-approximation 95% CI half-width bounds."""

    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    half = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(n)
    return mean, mean - half, mean + half


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a Bernoulli rate."""

    if total <= 0:
        return 0.0, 0.0
    p = wins / total
    denom = 1.0 + (z**2) / total
    center = p + (z**2) / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)
    return (
        max(0.0, (center - margin) / denom),
        min(1.0, (center + margin) / denom),
    )


def _distribution_summary(values: list[float]) -> dict[str, float]:
    mean, ci_lo, ci_hi = mean_ci95(values)
    return {
        "mean": mean,
        "ci95_lower": ci_lo,
        "ci95_upper": ci_hi,
        "median": float(statistics.median(values)) if values else 0.0,
        "p95": float(np.percentile(values, 95)) if values else 0.0,
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def summarize_profiles(profiles: list[GameProfile]) -> dict[str, Any]:
    """Aggregate per-game profiles into distribution summaries."""

    clean = [p for p in profiles if not p.aborted]
    summary: dict[str, Any] = {
        "num_games": len(profiles),
        "num_aborted": len(profiles) - len(clean),
    }
    summary["surfaced_total"] = _distribution_summary(
        [float(p.surfaced_total) for p in clean]
    )
    summary["surfaced_hero"] = _distribution_summary(
        [float(p.surfaced_hero) for p in clean]
    )
    summary["surfaced_villain"] = _distribution_summary(
        [float(p.surfaced_villain) for p in clean]
    )
    summary["skipped"] = _distribution_summary([float(p.skipped) for p in clean])
    summary["collapse_ratio"] = _distribution_summary(
        [p.collapse_ratio for p in clean]
    )
    summary["turns"] = _distribution_summary([float(p.turns) for p in clean])
    for kind in DECISION_KINDS:
        summary[f"hero_{kind}"] = _distribution_summary(
            [float(p.hero_by_kind.get(kind, 0)) for p in clean]
        )
        summary[f"villain_{kind}"] = _distribution_summary(
            [float(p.villain_by_kind.get(kind, 0)) for p in clean]
        )
    summary["single_valid_decisions_total"] = int(
        sum(p.single_valid_decisions for p in clean)
    )

    games = len(clean)
    hero_wins = sum(1 for p in clean if p.winner == p.hero_seat)
    decided = sum(1 for p in clean if p.winner is not None)
    ci_lo, ci_hi = wilson_interval(hero_wins, games)
    summary["hero_wins"] = hero_wins
    summary["decided_games"] = decided
    summary["hero_win_rate"] = hero_wins / games if games else 0.0
    summary["hero_win_ci95"] = [ci_lo, ci_hi]

    # Per-seat breakdown: seat 0 = hero on the play, seat 1 = hero on the draw.
    per_seat: dict[str, Any] = {}
    for seat in (0, 1):
        seat_games = [p for p in clean if p.hero_seat == seat]
        seat_wins = sum(1 for p in seat_games if p.winner == p.hero_seat)
        seat_lo, seat_hi = wilson_interval(seat_wins, len(seat_games))
        per_seat[str(seat)] = {
            "games": len(seat_games),
            "hero_wins": seat_wins,
            "hero_win_rate": seat_wins / len(seat_games) if seat_games else 0.0,
            "hero_win_ci95": [seat_lo, seat_hi],
        }
    summary["per_seat"] = per_seat

    # Raw seat advantage, role-agnostic: how often does player 0 (on the
    # play) win, whoever occupies that seat?
    otp_wins = sum(1 for p in clean if p.winner == 0)
    otp_lo, otp_hi = wilson_interval(otp_wins, decided)
    summary["on_the_play_wins"] = otp_wins
    summary["on_the_play_win_rate"] = otp_wins / decided if decided else 0.0
    summary["on_the_play_win_ci95"] = [otp_lo, otp_hi]
    return summary


def run_matchup(
    *,
    label: str,
    hero: str,
    villain: str,
    num_games: int,
    seed: int,
    deck: dict[str, int] | None = None,
    seat_balanced: bool = False,
    agent_seed: int | None = None,
) -> dict[str, Any]:
    profiles = play_profile_games(
        hero_policy_name=hero,
        villain_policy_name=villain,
        num_games=num_games,
        seed=seed,
        hero_deck=deck,
        villain_deck=deck,
        seat_balanced=seat_balanced,
        agent_seed=agent_seed,
    )
    summary = summarize_profiles(profiles)
    return {
        "label": label,
        "hero": hero,
        "villain": villain,
        "seed": seed,
        "agent_seed": agent_seed,
        "seat_balanced": seat_balanced,
        "summary": summary,
        "games": [asdict(p) for p in profiles],
    }


def _print_summary(label: str, summary: dict[str, Any]) -> None:
    print(f"\n=== {label} ({summary['num_games']} games, "
          f"{summary['num_aborted']} aborted) ===")
    rows = [
        ("surfaced decisions (both)", "surfaced_total"),
        ("surfaced decisions (hero)", "surfaced_hero"),
        ("surfaced decisions (villain)", "surfaced_villain"),
        ("skipped (trivial)", "skipped"),
        ("collapse ratio", "collapse_ratio"),
        ("turns", "turns"),
    ]
    for kind in DECISION_KINDS:
        rows.append((f"hero {kind}", f"hero_{kind}"))
        rows.append((f"villain {kind}", f"villain_{kind}"))
    for name, key in rows:
        stats = summary[key]
        print(
            f"  {name:30s} mean {stats['mean']:8.2f} "
            f"[{stats['ci95_lower']:.2f}, {stats['ci95_upper']:.2f}] "
            f"median {stats['median']:7.1f} p95 {stats['p95']:7.1f}"
        )
    lo, hi = summary["hero_win_ci95"]
    print(
        f"  hero win rate: {summary['hero_win_rate']:.3f} "
        f"(Wilson 95% [{lo:.3f}, {hi:.3f}], "
        f"{summary['hero_wins']}/{summary['num_games']} wins, "
        f"{summary['decided_games']} decided)"
    )
    for seat, seat_name in ((0, "on the play"), (1, "on the draw")):
        stats = summary["per_seat"][str(seat)]
        if stats["games"] == 0:
            continue
        seat_lo, seat_hi = stats["hero_win_ci95"]
        print(
            f"  hero win rate ({seat_name}): {stats['hero_win_rate']:.3f} "
            f"(Wilson 95% [{seat_lo:.3f}, {seat_hi:.3f}], "
            f"{stats['hero_wins']}/{stats['games']})"
        )
    otp_lo, otp_hi = summary["on_the_play_win_ci95"]
    print(
        f"  on-the-play win rate (either role): "
        f"{summary['on_the_play_win_rate']:.3f} "
        f"(Wilson 95% [{otp_lo:.3f}, {otp_hi:.3f}], "
        f"{summary['on_the_play_wins']}/{summary['decided_games']} decided)"
    )
    print(
        f"  surfaced decisions with <=1 valid action: "
        f"{summary['single_valid_decisions_total']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-random", type=int, default=500,
                        help="random-vs-random games (E0a + E0c)")
    parser.add_argument("--games-untrained", type=int, default=200,
                        help="untrained-vs-random games (E0a + E0c)")
    parser.add_argument("--games-baseline", type=int, default=200,
                        help="untrained-vs-passive games (E0c)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deck", choices=sorted(DECKS), default="standard",
                        help="deck for both players")
    parser.add_argument("--seat-balanced", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="alternate the hero between seats 0 and 1 "
                             "(--no-seat-balanced reproduces the C0 protocol)")
    parser.add_argument("--untrained-inits", type=int, default=1,
                        help="number of fresh untrained inits for "
                             "untrained-vs-random (each plays "
                             "--games-untrained games)")
    parser.add_argument("--baseline-inits", type=int, default=1,
                        help="number of fresh untrained inits for "
                             "untrained-vs-passive")
    parser.add_argument("--out", type=str, default=None,
                        help="write full JSON results to this path")
    args = parser.parse_args()

    suppress_truncation_logs()

    deck = DECKS[args.deck]
    matchups = []
    if args.games_random > 0:
        matchups.append(
            run_matchup(
                label=f"random-vs-random ({args.deck})",
                hero="random",
                villain="random",
                num_games=args.games_random,
                seed=args.seed,
                deck=deck,
                seat_balanced=args.seat_balanced,
            )
        )
    if args.games_untrained > 0:
        for init in range(args.untrained_inits):
            agent_seed = args.seed + 100_000 + 1_000 * init
            matchups.append(
                run_matchup(
                    label=(
                        f"untrained-vs-random ({args.deck}, "
                        f"init seed {agent_seed})"
                    ),
                    hero="untrained",
                    villain="random",
                    num_games=args.games_untrained,
                    seed=args.seed + 100_000 + 1_000 * init,
                    deck=deck,
                    seat_balanced=args.seat_balanced,
                    agent_seed=agent_seed,
                )
            )
    if args.games_baseline > 0:
        for init in range(args.baseline_inits):
            agent_seed = args.seed + 200_000 + 1_000 * init
            matchups.append(
                run_matchup(
                    label=(
                        f"untrained-vs-passive ({args.deck}, "
                        f"init seed {agent_seed})"
                    ),
                    hero="untrained",
                    villain="passive",
                    num_games=args.games_baseline,
                    seed=args.seed + 200_000 + 1_000 * init,
                    deck=deck,
                    seat_balanced=args.seat_balanced,
                    agent_seed=agent_seed,
                )
            )

    for matchup in matchups:
        _print_summary(matchup["label"], matchup["summary"])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(matchups, indent=2))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
