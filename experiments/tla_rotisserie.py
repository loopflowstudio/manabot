"""Prepare the public TLA Rotisserie reward comparison's native foundations.

The uniform reference draft is software validation. Training and reward scoring
require the maintained learner/supervision integration and a supported competent
frozen player; this entry point does not substitute a player or bypass admission.
"""

import argparse
from pathlib import Path

from manabot.drafting import prepare_draft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="retain one unscored reference draft")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=370)
    args = parser.parse_args()
    if args.command == "prepare":
        try:
            result = prepare_draft(args.out, args.seed)
        except (ValueError, FileExistsError) as error:
            raise SystemExit(f"No completed preparation: {error}") from None
        print(
            f"software validation: {result.draft_picks} draft picks, "
            f"{len(result.decks)} admitted decks; "
            f"supply {sum(result.initial)} -> {sum(result.remaining)}; "
            f"receipt: {args.out / 'receipt.json'}"
        )


if __name__ == "__main__":
    main()
