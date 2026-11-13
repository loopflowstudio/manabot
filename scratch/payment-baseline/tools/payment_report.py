"""Summarize `payment_metrics` lines from an instrumented bench run.

Usage: payment_report.py <bench output> [...]. Sums the metrics of every
`payment_metrics` line in each file (bench_search prints one per seed).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def merge(into: dict, more: dict) -> None:
    for key, value in more.items():
        if isinstance(value, dict):
            merge(into.setdefault(key, {}), value)
        elif isinstance(value, list):
            have = into.setdefault(key, [0] * len(value))
            for i, v in enumerate(value):
                have[i] += v
        elif key.startswith("max_"):
            into[key] = max(into.get(key, 0), value)
        else:
            into[key] = into.get(key, 0) + value


def load(path: Path) -> tuple[dict, float]:
    metrics: dict = {}
    elapsed = 0.0
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if "payment_metrics" in row:
            merge(metrics, row["payment_metrics"])
        elif "elapsed_s" in row:
            elapsed += row["elapsed_s"]
        elif "total_s" in row and "seed" in row:
            elapsed += row["total_s"]
    return metrics, elapsed


def hist(values: list[int]) -> str:
    total = sum(values)
    if not total:
        return "none"
    mean = sum(i * v for i, v in enumerate(values)) / total
    cum, marks = 0, {}
    for i, v in enumerate(values):
        cum += v
        for q in (0.5, 0.9, 0.99):
            if q not in marks and cum >= q * total:
                marks[q] = i
    top = max(i for i, v in enumerate(values) if v)
    zero = values[0] / total
    return (f"n={total} mean={mean:.2f} p50={marks[0.5]} p90={marks[0.9]} "
            f"p99={marks[0.99]} max={'>=' if top == len(values) - 1 else ''}{top} zero={zero:.1%}")


def report(path: Path) -> None:
    m, elapsed = load(path)
    plans, models = m["plans"], m["models"]
    print(f"== {path.name}: elapsed {elapsed:.2f}s (timed region), steps {m['steps']}")
    print(f"models {models} ({models/m['steps']:.2f}/step)  plans {plans} ({plans/m['steps']:.2f}/step)  "
          f"calls {m['calls']} ({m['calls']/plans:.2f}/plan, max {m['max_plan_calls']})  states {m['states']}")
    mns, pns = m["model_ns"] / 1e9, m["plan_ns"] / 1e9
    print(f"timer share of elapsed: model {mns:.3f}s = {mns/elapsed:.1%} ({m['model_ns']/models:.0f} ns each); "
          f"plan {pns:.3f}s = {pns/elapsed:.1%} ({m['plan_ns']/plans:.0f} ns each); together {(mns+pns)/elapsed:.1%}")
    for name in ("calls_per_plan", "sources_per_model", "plans_per_step", "models_per_step"):
        print(f"{name}: {hist(m[name])}")
    for phase, values in m["plans_per_computation"].items():
        print(f"plans per {phase}: {hist(values)}")
        print(f"models per {phase}: {hist(m['models_per_computation'][phase])}")

    print("-- plans by arithmetic class (all phases and sites)")
    by_class: dict = defaultdict(lambda: defaultdict(int))
    by_site: dict = defaultdict(lambda: defaultdict(int))
    by_phase: dict = defaultdict(lambda: defaultdict(int))
    for key, row in m["plan_rows"].items():
        phase, site, cls = key.split("/")
        for target in (by_class[cls], by_site[site], by_phase[phase]):
            for k, v in row.items():
                target[k] += v
    def table(rows: dict) -> None:
        for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["plans"]):
            print(f"  {name:32s} plans {r['plans']:>9} {r['plans']/plans:6.1%}  payable {r['payable']/r['plans']:6.1%}  "
                  f"calls/plan {r['calls']/r['plans']:5.2f}  time {r['ns']/m['plan_ns']:6.1%} ({r['ns']/r['plans']:.0f} ns)  "
                  f"mispredicted {r['mispredicted']}")
    table(by_class)
    print("-- plans by entry point"); table(by_site)
    print("-- plans by phase"); table(by_phase)

    print("-- models by phase/site")
    total = defaultdict(int)
    for key, r in sorted(m["model_rows"].items(), key=lambda kv: -kv[1]["builds"]):
        for k, v in r.items():
            total[k] += v
        print(f"  {key:58s} builds {r['builds']:>9} {r['builds']/models:6.1%}  {r['ns']/r['builds']:.0f} ns  "
              f"sources {r['sources']/r['builds']:.1f}  abil {r['with_abilities']/r['builds']:.0%}  "
              f"same-model-in-step {r['same_model_in_step']/r['builds']:.0%}  same-rev-in-step {r['same_revision_in_step']/r['builds']:.0%}  "
              f"same-rev-as-prev {r['same_revision_as_previous']/r['builds']:.0%}  rev-differs {r['revision_match_differs']}")
    b = total["builds"]
    print(f"  ALL: with abilities {total['with_abilities']/b:.1%}; same final model already built this step {total['same_model_in_step']/b:.1%}; "
          f"same (player, characteristics, tapped) revisions already built this step {total['same_revision_in_step']/b:.1%}; "
          f"same as player's previous build {total['same_revision_as_previous']/b:.1%}; "
          f"revision matches with different sources {total['revision_match_differs']}")


for arg in sys.argv[1:]:
    report(Path(arg))
