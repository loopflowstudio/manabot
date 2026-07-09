"""E2c runner (wave/search C2): potential-based shaping on the interactive deck.

Dev-preset first-light run with the pay-per-event shaping replaced by
potential-based shaping (gamma * Phi(s') - Phi(s), Phi(terminal) = 0).
Weights chosen so per-event shaping magnitudes match E2b's exactly:
land 0.03, creature 0.06, life 0.2/20-life (= 0.01 per life point).

Usage:
    python scratch/run_c2_potential.py --seed 1 --label c2-potential-s1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")

import manabot.verify.first_light as fl
from manabot.verify.util import INTERACTIVE_DECK

# C1/C2 deck: both players on the interactive UR list.
fl.STANDARD_DECK = INTERACTIVE_DECK

# E2c reward: terminal win/lose plus potential-based shaping only.
# potential_gamma matches train.gamma (0.99) — required for policy invariance.
fl.FIRST_LIGHT_REWARD = {
    "win_reward": 1.0,
    "lose_reward": -1.0,
    "land_play_reward": 0.0,
    "creature_play_reward": 0.0,
    "opponent_life_loss_reward": 0.0,
    "potential_enabled": True,
    "potential_gamma": 0.99,
    "potential_land_weight": 0.03,
    "potential_creature_weight": 0.06,
    "potential_life_weight": 0.2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--db", default=".runs/verify.sqlite")
    args = parser.parse_args()

    config = fl.resolve_run_config(
        db=Path(args.db),
        seed=args.seed,
        mode="dev",
        label=args.label,
        opponent_policy="random",
        num_envs=None,
        num_steps=None,
        total_timesteps=None,
        eval_interval=None,
        eval_num_games=None,
        baseline=True,
        report=False,
        report_path=None,
        notes="E2c potential-based shaping (exp-04)",
    )
    summary = fl.run_first_light(config)
    final = summary.get("final") or {}
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "label": args.label,
                "final_win_rate": final.get("win_rate"),
                "final_cast_when_able": final.get("cast_when_able"),
                "final_passed_when_able": final.get("passed_when_able"),
                "final_landed_when_able": final.get("landed_when_able"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
