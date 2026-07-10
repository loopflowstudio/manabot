"""E2c judging eval: seat-balanced 400-game eval vs random for each final checkpoint.

Usage:
    python scratch/judge_c2_potential.py \
        --out reports/data/exp-04-potential-judging.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")

import torch

from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import (
    AgentHypers,
    MatchHypers,
    ObservationSpaceHypers,
    RewardHypers,
)
from manabot.model import Agent
from manabot.verify.util import (
    INTERACTIVE_DECK,
    capture_evaluation,
    wilson_lower_bound,
)

CHECKPOINT_TEMPLATE = ".runs/first-light-c2-potential-s{seed}-final/step_65536.pt"

REPORT_KEYS = (
    "win_rate",
    "win_ci_lower",
    "win_rate_on_play",
    "win_rate_on_draw",
    "wins",
    "wins_on_play",
    "wins_on_draw",
    "games_on_play",
    "games_on_draw",
    "landed_when_able",
    "cast_when_able",
    "passed_when_able",
    "attacked_when_able",
    "mean_steps",
)


def load_agent(path: str) -> tuple[Agent, ObservationSpace]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hypers = checkpoint["hypers"]
    obs_space = ObservationSpace(
        ObservationSpaceHypers(**hypers["observation_hypers"])
    )
    agent = Agent(obs_space, AgentHypers(**hypers["agent_hypers"]))
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    return agent, obs_space


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--eval-seed-base", type=int, default=4001)
    parser.add_argument("--template", default=CHECKPOINT_TEMPLATE)
    parser.add_argument("--prefix", default="c2-potential")
    parser.add_argument(
        "--out", default="reports/data/exp-04-potential-judging.json"
    )
    args = parser.parse_args()

    match = Match(
        MatchHypers(hero_deck=INTERACTIVE_DECK, villain_deck=INTERACTIVE_DECK)
    )
    reward = Reward(RewardHypers())

    results: dict[str, dict] = {}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        results = json.loads(out_path.read_text())

    for offset, s in enumerate(int(x) for x in args.seeds.split(",")):
        name = f"{args.prefix}-s{s}"
        if name in results:
            print(f"[skip] {name}")
            continue
        path = args.template.format(seed=s)
        agent, obs_space = load_agent(path)
        artifacts = capture_evaluation(
            agent,
            obs_space,
            match,
            reward,
            num_games=args.games,
            opponent_policy="random",
            deterministic=False,
            seed=args.eval_seed_base + offset,
            capture_actions=False,
            seat_balanced=True,
        )
        m = artifacts.metrics
        row = {k: m[k] for k in REPORT_KEYS}
        row["wilson_lb_on_play"] = wilson_lower_bound(
            int(m["wins_on_play"]), int(m["games_on_play"])
        )
        row["wilson_lb_on_draw"] = wilson_lower_bound(
            int(m["wins_on_draw"]), int(m["games_on_draw"])
        )
        row["checkpoint"] = path
        row["eval_seed"] = args.eval_seed_base + offset
        results[name] = row
        out_path.write_text(json.dumps(results, indent=2))
        print(
            f"{name}: overall {m['win_rate']:.3f} (LB {m['win_ci_lower']:.3f}) "
            f"play {m['win_rate_on_play']:.3f} draw {m['win_rate_on_draw']:.3f} "
            f"cast {m['cast_when_able']:.2f} pass {m['passed_when_able']:.2f} "
            f"land {m['landed_when_able']:.2f}"
        )

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
