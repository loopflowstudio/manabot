"""Render retained evaluation performance receipts without executing games.

Training reports belong to Etude experiment files. This evaluation-only builder
keeps the accepted activity overview, resource clocks and evidence links until
Etude exposes equivalent evaluation-receipt reporting. `write_evaluations` is
also the calibration runner's output adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from manabot.reports.evaluation import project_evaluations

ROOT = Path(__file__).parent
ASSETS = ("report.html", "model.js", "charts.js")


def write_evaluations(directories: list[Path], out: Path) -> Path:
    """Render real arena receipts using the same page, controls and icicles."""
    experiment = project_evaluations(directories)
    out.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS:
        shutil.copyfile(ROOT / asset, out / asset)
    evidence = out / "evidence"
    evidence.mkdir(exist_ok=True)
    copied: dict[str, str] = {}

    def retain(source: str) -> str:
        if source not in copied:
            target = evidence / f"{len(copied)}-{Path(source).name}"
            shutil.copyfile(source, target)
            copied[source] = str(target.relative_to(out))
        return copied[source]

    for run in experiment["eval_performance"]["runs"]:
        run["evidence"] = retain(run["evidence"])
        run["optimization"]["evidence"] = run["evidence"]
    for allocation in experiment["eval_performance"]["allocations"]:
        allocation["evidence"] = retain(allocation["evidence"])
        allocation["serving"]["traces"] = [
            retain(path) for path in allocation["serving"]["traces"]
        ]
    (out / "data.js").write_text(
        "window.REPORT=" + json.dumps({"experiments": [experiment]}) + ";\n"
    )
    return out / "report.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(".runs/eval/report"))
    parser.add_argument("--eval-receipt", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(write_evaluations(args.eval_receipt, args.out))


if __name__ == "__main__":
    main()
