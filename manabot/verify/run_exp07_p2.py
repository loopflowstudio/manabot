"""Exp-07 (wave/search C7) Task 2 measurements: policy rollouts inside search.

Three phases, resumable via the output JSON:
    A. throughput probe — psearch-N self-play decisions: ms/decision,
       playouts/sec, net obs/sec; contrasted with random-rollout search cost
       (exp-02: ~0.44 ms/decision per sim at N=256).
    B. equal sims — psearch-N vs search-N head-to-head.
    C. equal wall-clock — psearch-N vs search-N*, where N* gives the
       random-rollout searcher the same measured wall-clock per decision
       (P2, the honest comparison).

Usage:
    python -m manabot.verify.run_exp07_p2 --student .runs/exp07/student_r0.pt \
        --sims 16 --games 100 --out reports/data/exp-07-p2.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

# exp-02 measured random-rollout search at 7/31/113 ms per decision for
# N=16/64/256 — linear in N at ~0.44 ms/decision/sim on this machine.
RANDOM_SEARCH_MS_PER_SIM = 113.0 / 256.0


def throughput_probe(
    spec: dict[str, Any],
    *,
    decisions: int,
    seed: int,
) -> dict[str, Any]:
    """Play search self-play decisions on one env; measure cost."""

    from manabot.env import Env, Match, ObservationSpace, Reward
    from manabot.infra.hypers import MatchHypers, RewardHypers
    from manabot.sim.flat_mc import make_player
    from manabot.verify.util import INTERACTIVE_DECK

    player, _ = make_player(dict(spec), seed=seed)
    obs_space = ObservationSpace()
    match = Match(
        MatchHypers(
            hero="a",
            villain="b",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(match, obs_space, Reward(RewardHypers()), seed=seed, auto_reset=False)
    obs, _ = env.reset(seed=seed)
    game = 0
    start = time.perf_counter()
    while player.stats.decisions < decisions:
        action = player.act(env, obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            game += 1
            obs, _ = env.reset(seed=seed + game)
    wall = time.perf_counter() - start

    stats = player.stats
    out = {
        "spec": {k: v for k, v in spec.items() if k != "checkpoint"},
        "decisions": stats.decisions,
        "wall_seconds": wall,
        "ms_per_decision": 1000.0 * stats.seconds / stats.decisions,
        "playouts": stats.simulations,
        "playouts_per_second": stats.simulations / stats.seconds,
        "cap_hits": stats.cap_hits,
        "random_search_ms_per_sim_ref": RANDOM_SEARCH_MS_PER_SIM,
    }
    if hasattr(stats, "net_obs"):
        out.update(
            net_obs=stats.net_obs,
            net_obs_per_second=stats.net_obs / stats.seconds,
            net_forwards=stats.net_forwards,
            rollout_steps=stats.rollout_steps,
            mean_plies_per_playout=stats.rollout_steps / max(1, stats.simulations),
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=str, required=True)
    parser.add_argument("--sims", type=int, default=16)
    parser.add_argument("--probe-decisions", type=int, default=60)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument(
        "--policy-plies",
        type=int,
        default=None,
        help="hybrid rollouts: policy plays this many plies, engine random-"
        "finishes the tail (None = policy to terminal)",
    )
    parser.add_argument(
        "--equal-wallclock-sims",
        type=int,
        default=None,
        help="override the N* derived from the probe",
    )
    parser.add_argument("--out", type=str, default="reports/data/exp-07-p2.json")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    def save() -> None:
        out_path.write_text(json.dumps(results, indent=2))

    psearch_spec = {
        "kind": "policy_search",
        "sims": args.sims,
        "checkpoint": args.student,
        "device": args.device,
        "epsilon": args.epsilon,
        "policy_plies": args.policy_plies,
        "name": f"psearch-{args.sims}",
    }

    # Phase A — throughput probes, both searchers, same box, same day.
    if "probe" not in results:
        print(
            f"[run ] probe: psearch-{args.sims} self-play, "
            f"{args.probe_decisions} decisions",
            flush=True,
        )
        results["probe"] = throughput_probe(
            psearch_spec, decisions=args.probe_decisions, seed=args.seed
        )
        save()
    if "probe_random" not in results:
        print(
            f"[run ] probe: search-{args.sims} (random rollouts) self-play, "
            f"{args.probe_decisions} decisions",
            flush=True,
        )
        results["probe_random"] = throughput_probe(
            {"kind": "search", "sims": args.sims},
            decisions=args.probe_decisions,
            seed=args.seed,
        )
        save()
    probe = results["probe"]
    probe_random = results["probe_random"]
    print(
        f"       policy: {probe['ms_per_decision']:.0f} ms/dec | "
        f"{probe['playouts_per_second']:.1f} playouts/s | "
        f"{probe['net_obs_per_second']:.0f} net obs/s | "
        f"{probe['mean_plies_per_playout']:.0f} plies/playout",
        flush=True,
    )
    print(
        f"       random: {probe_random['ms_per_decision']:.1f} ms/dec | "
        f"{probe_random['playouts_per_second']:.0f} playouts/s",
        flush=True,
    )

    from manabot.verify.run_flat_mc import run_matchup

    # Phase B — equal sims.
    key = f"equal_sims_{args.sims}"
    if key not in results:
        print(
            f"[run ] equal sims: psearch-{args.sims} vs search-{args.sims}, "
            f"{args.games} games",
            flush=True,
        )
        results[key] = run_matchup(
            psearch_spec,
            {"kind": "search", "sims": args.sims},
            num_games=args.games,
            workers=args.workers,
            base_seed=args.seed + 17,
        )
        save()
        m = results[key]["metrics"]
        print(
            f"       win {m['win_rate']:.3f} "
            f"[{m['win_ci_lower']:.3f},{m['win_ci_upper']:.3f}]",
            flush=True,
        )

    # Phase C — equal wall-clock (P2). N* from the same-day probes:
    # random-rollout search cost is linear in sims, so its per-sim cost is
    # probe_random ms/dec divided by the probe's sims.
    ms_per_sim_today = probe_random["ms_per_decision"] / args.sims
    n_star = args.equal_wallclock_sims or max(
        args.sims,
        int(round(probe["ms_per_decision"] / ms_per_sim_today)),
    )
    results["ms_per_sim_today"] = ms_per_sim_today
    results["equal_wallclock_sims"] = n_star
    key = f"equal_wallclock_{n_star}"
    if key not in results:
        print(
            f"[run ] equal wall-clock: psearch-{args.sims} vs search-{n_star}, "
            f"{args.games} games",
            flush=True,
        )
        results[key] = run_matchup(
            psearch_spec,
            {"kind": "search", "sims": n_star},
            num_games=args.games,
            workers=args.workers,
            base_seed=args.seed + 31,
        )
        save()
        m = results[key]["metrics"]
        hero_ms = (results[key]["hero_search"] or {}).get(
            "mean_seconds_per_decision", 0.0
        ) * 1000
        villain_ms = (results[key]["villain_search"] or {}).get(
            "mean_seconds_per_decision", 0.0
        ) * 1000
        print(
            f"       win {m['win_rate']:.3f} "
            f"[{m['win_ci_lower']:.3f},{m['win_ci_upper']:.3f}] | "
            f"hero {hero_ms:.0f} ms/dec vs villain {villain_ms:.0f} ms/dec",
            flush=True,
        )

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
