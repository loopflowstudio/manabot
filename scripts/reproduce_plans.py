"""Regenerate the eight ETU-80 cells without training or network access.

uv run --no-project --python 3.12 python scripts/reproduce_plans.py [--check]
"""

import argparse
import json

from budget_plan import ROOT, Inputs, build_report, markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "experiments/planning"
    report = build_report(Inputs(), destination / "prices.json", {})
    outputs = {"grid.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
               "grid.md": markdown(report) + "\n"}
    for name, content in outputs.items():
        path = destination / name
        if args.check:
            if path.read_text() != content:
                raise SystemExit(f"stale planning output: {path}")
        else:
            path.write_text(content)
    print("Eight cells reproduced; no training launched or spending authorized.")


if __name__ == "__main__":
    main()
