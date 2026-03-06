"""Command-line entrypoint for the first-light experiment harness."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .first_light import MODE_PRESETS, resolve_run_config, run_first_light


def _build_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default=".runs/verify.sqlite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_PRESETS),
        default="dev",
    )
    parser.add_argument(
        "--opponent",
        dest="opponent_policy",
        choices=["random", "passive"],
        default="random",
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-num-games", type=int)
    parser.add_argument("--notes")
    parser.add_argument(
        "--baseline",
        dest="baseline",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-baseline",
        dest="baseline",
        action="store_false",
    )
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--report-path")
    return parser


def _build_config(args: argparse.Namespace, *, evaluation_only: bool = False):
    total_timesteps = 0 if evaluation_only else args.total_timesteps
    eval_interval = None if evaluation_only else args.eval_interval
    return resolve_run_config(
        db=args.db,
        seed=args.seed,
        mode=args.mode,
        label=args.label,
        opponent_policy=args.opponent_policy,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=total_timesteps,
        eval_interval=eval_interval,
        eval_num_games=args.eval_num_games,
        baseline=False if evaluation_only else args.baseline,
        report=args.report,
        report_path=args.report_path,
        notes=args.notes,
    )


def _print_summary(summary: dict) -> None:
    final = summary.get("final") or {}
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "recommendation": summary["recommendation"],
                "final_win_rate": final.get("win_rate"),
                "final_landed_when_able": final.get("landed_when_able"),
                "final_cast_when_able": final.get("cast_when_able"),
                "report_path": summary.get("report_path"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    common = _build_common_parser()
    parser = argparse.ArgumentParser(description=__doc__, parents=[common])

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "train",
        help="Run the full harness",
        parents=[common],
    )
    subparsers.add_parser(
        "eval",
        help="Record an evaluation-only run with no training",
        parents=[common],
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evaluation_only = args.command == "eval"
    summary = run_first_light(_build_config(args, evaluation_only=evaluation_only))
    _print_summary(summary)


if __name__ == "__main__":
    main()
