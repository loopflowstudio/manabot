"""Summarize the encoding screen against the checkpoint reference.

Metrics per variant, over positions every listed variant answered:
- agree: Jev's choice equals the checkpoint's most likely action.
- ckpt_mass: the checkpoint's probability of Jev's choice (higher is better).
- jev_on_ref: Jev's own probability on the checkpoint's choice.
Differences against ``base`` are paired by position, with a 95% percentile
bootstrap over positions.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def _ref(label: dict) -> str:
    return max(label["checkpoint"], key=label["checkpoint"].get)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--labels", type=Path, default=Path("data/labels.jsonl"))
    parser.add_argument("--baseline", default="base")
    args = parser.parse_args()
    labels = {(r["game"], r["step"]): r for r in map(json.loads, args.labels.open())}
    answers: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    failures: dict[str, int] = defaultdict(int)
    for path in args.results:
        for record in map(json.loads, path.open()):
            if record["failure"] or record["choice"] is None:
                failures[record["variant"]] += 1
                continue
            answers[record["game"], record["step"]][record["variant"]] = record
    variants = sorted({v for a in answers.values() for v in a})
    keys = [k for k, a in answers.items() if set(a) == set(variants)]
    print(f"positions answered by all {len(variants)} variants: {len(keys)}; failures {dict(failures)}")

    def metrics(key: tuple[str, str], variant: str) -> tuple[float, float, float]:
        label, answer = labels[key], answers[key][variant]
        ref = _ref(label)
        return (
            float(answer["choice"] == ref),
            label["checkpoint"].get(answer["choice"], 0.0),
            (answer["probabilities"] or {}).get(ref, 0.0),
        )

    table = {v: [metrics(k, v) for k in keys] for v in variants}
    rng = random.Random(0)
    boots = [[rng.randrange(len(keys)) for _ in keys] for _ in range(2000)]
    logged = statistics.mean(float(labels[k]["jev_choice"] == _ref(labels[k])) for k in keys)
    print(f"logged campaign choice agreement on these positions: {logged:.3f}")
    print(f"{'variant':<10} {'agree':>6} {'Δagree vs base [95%]':>24} {'ckpt_mass':>9} {'jev_on_ref':>10} {'tokens':>7} {'$/1k dec':>8}")
    for v in variants:
        rows = table[v]
        agree = statistics.mean(r[0] for r in rows)
        diffs = [rows[i][0] - table[args.baseline][i][0] for i in range(len(keys))]
        deltas = sorted(statistics.mean(diffs[i] for i in sample) for sample in boots)
        tokens = statistics.median(answers[k][v]["prompt_tokens"] for k in keys)
        cost = statistics.mean(float(answers[k][v]["cost"] or 0) for k in keys) * 1000
        print(
            f"{v:<10} {agree:6.3f} {statistics.mean(diffs):+7.3f} [{deltas[50]:+.3f}, {deltas[1949]:+.3f}]"
            f" {statistics.mean(r[1] for r in rows):9.3f} {statistics.mean(r[2] for r in rows):10.3f} {tokens:7.0f} {cost:8.3f}"
        )
    kinds: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for k in keys:
        kinds[labels[k]["kinds"][_ref(labels[k])].split(" ")[0]].append(k)
    print("\nagreement by the checkpoint's action kind (n):")
    print("kind".ljust(10) + "".join(v[:9].rjust(10) for v in variants))
    for kind, members in sorted(kinds.items(), key=lambda item: -len(item[1])):
        if len(members) < 10:
            continue
        print(
            f"{kind} ({len(members)})".ljust(10)
            + "".join(f"{statistics.mean(metrics(k, v)[0] for k in members):10.2f}" for v in variants)
        )


if __name__ == "__main__":
    main()
