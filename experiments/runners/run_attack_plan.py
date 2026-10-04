"""Execute a separately budgeted frozen-policy attack plan."""

import argparse
from pathlib import Path

from manabot.training.attack_execution import execute_attack_plan
from manabot.training.attacks import AttackPlan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = AttackPlan.model_validate_json(args.plan.read_text())
    execute_attack_plan(plan, args.out)


if __name__ == "__main__":
    main()
