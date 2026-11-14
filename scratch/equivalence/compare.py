"""Compare two captures bitwise; print counts and the largest differences."""

from collections import Counter
import sys

import torch

base = torch.load(sys.argv[1])
head = torch.load(sys.argv[2])
assert base.keys() == head.keys(), set(base) ^ set(head)
kinds: Counter[str] = Counter()
differ = []
for key in base:
    a, b = base[key], head[key]
    kind = (
        "train"
        if key.startswith("train/")
        else key.split("/")[-2 if "infer" in key else 2]
    )
    kinds[kind] += 1
    if a.shape != b.shape or a.dtype != b.dtype:
        differ.append((key, "shape/dtype"))
        continue
    same = (
        torch.equal(a, b)
        if not a.is_floating_point()
        else bool(((a == b) | (a.isnan() & b.isnan())).all())
    )
    if not same:
        differ.append((key, float((a.double() - b.double()).abs().max())))
worst: dict[str, float] = {}
for key, delta in differ:
    if isinstance(delta, float):
        kind = (
            "train"
            if key.startswith("train/")
            else key.split("/")[-2 if "infer" in key else 2]
        )
        worst[kind] = max(worst.get(kind, 0.0), delta)
print("tensors", len(base), dict(kinds))
print("max abs diff by kind", worst)
print("differing", len(differ))
for key, delta in sorted(differ, key=lambda d: -d[1] if isinstance(d[1], float) else 0)[
    :15
]:
    print(" ", key, delta)
