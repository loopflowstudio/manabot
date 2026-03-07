"""Legacy wrapper around the first-light harness for random-opponent runs."""

from __future__ import annotations

import argparse
import json

from .first_light import resolve_run_config, run_first_light


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=".runs/verify.sqlite")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label", default="step4-beat-random")
    parser.add_argument("--total-timesteps", type=int, default=1_048_576)
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=262_144)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)

    summary = run_first_light(
        resolve_run_config(
            db=args.db,
            seed=args.seed,
            mode="decision",
            label=args.label,
            opponent_policy="random",
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            total_timesteps=args.total_timesteps,
            eval_interval=args.eval_interval,
            eval_num_games=args.num_games,
            baseline=True,
            report=args.report,
            report_path=None,
            notes="Invoked via legacy step4_beat_random wrapper.",
        )
    )

    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "recommendation": summary["recommendation"],
                "final_win_rate": summary["final"]["win_rate"],
                "final_win_ci_lower": summary["final"]["win_ci_lower"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
