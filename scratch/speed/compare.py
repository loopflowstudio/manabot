"""Summarize ABAB speed-check arms: per-arm median of process medians, spread, gap.

uv run python scratch/speed/compare.py DIR   # DIR holds base-N.json, head-N.json
"""

import json
from pathlib import Path
import statistics
import sys

# (label, reader, larger_is_better)
METRICS = [
    ("damped steps/s", lambda r: r["damped"]["steps_per_second"]["median"], True),
    ("ppo steps/s", lambda r: r["ppo"]["steps_per_second"]["median"], True),
    ("inference obs/s", lambda r: r["inference_obs_per_second"]["median"], True),
    ("host preparation s", lambda r: r["host_preparation_seconds"]["median"], False),
    ("save s", lambda r: r["save_seconds"]["median"], False),
    ("load s", lambda r: r["load_seconds"]["median"], False),
    ("collection s", lambda r: r["collection_seconds"], False),
]

folder = Path(sys.argv[1])
names = sorted({p.stem.rsplit("-", 1)[0] for p in folder.glob("*-*.json")})
arms = {
    arm: [json.loads(p.read_text()) for p in sorted(folder.glob(f"{arm}-*.json"))]
    for arm in ["base"] + [n for n in names if n != "base"]
}
for arm, results in arms.items():
    rows = {r["learner_rows"] for r in results}
    steps = {tuple(r["damped"]["steps"]) for r in results}
    print(f"{arm}: {len(results)} processes, learner rows {rows}, damped steps {steps}")
for arm, results in arms.items():
    print(
        f"{arm}: forward sync warnings {[r.get('forward_sync_warning_count') for r in results]}"
    )
lines = []
for label, read, larger in METRICS:
    values = {arm: [read(r) for r in results] for arm, results in arms.items()}
    median = {arm: statistics.median(v) for arm, v in values.items()}
    spread = max(max(v) - min(v) for v in values.values())
    for arm in arms:
        if arm == "base":
            continue
        gap = median[arm] - median["base"]
        slower = -gap if larger else gap
        verdict = "REGRESSION" if slower > spread else "within spread"
        line = (
            f"{label:18s} base {median['base']:.4g} [{min(values['base']):.4g}, "
            f"{max(values['base']):.4g}]  {arm} {median[arm]:.4g} "
            f"[{min(values[arm]):.4g}, {max(values[arm]):.4g}]  "
            f"{arm}/base {median[arm] / median['base']:.3f}  {verdict}"
        )
        lines.append(line)
        print(line)
# Learner sections (seconds per update, medians over processes).
for kind in ("damped", "ppo"):
    labels = sorted(
        {k for rs in arms.values() for r in rs for k in r[kind].get("sections", {})}
    )
    for label in labels:
        cells = []
        for arm, results in arms.items():
            totals = [
                r[kind]["sections"][label]["total_time"]
                for r in results
                if label in r[kind].get("sections", {})
            ]
            cells.append(
                f"{arm} {statistics.median(totals):.3f}" if totals else f"{arm} -"
            )
        line = f"{kind} {label:55s} " + "  ".join(cells)
        lines.append(line)
        print(line)
(folder / "summary.txt").write_text("\n".join(lines) + "\n")
