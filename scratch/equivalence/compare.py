"""Compare two captures bitwise; print counts and the largest differences."""

import sys
from collections import Counter

import torch

base = torch.load(sys.argv[1])
head = torch.load(sys.argv[2])
assert base.keys() == head.keys(), (set(base) ^ set(head))
kinds: Counter[str] = Counter()
differ = []
for key in base:
    a, b = base[key], head[key]
    kind = "train" if key.startswith("train/") else key.split("/")[2]
    kinds[kind] += 1
    if a.shape != b.shape or a.dtype != b.dtype:
        differ.append((key, "shape/dtype"))
        continue
    same = torch.equal(a, b) if not a.is_floating_point() else bool(
        ((a == b) | (a.isnan() & b.isnan())).all()
    )
    if not same:
        differ.append((key, float((a.double() - b.double()).abs().max())))
print("tensors", len(base), dict(kinds))
print("differing", len(differ))
for key, delta in differ[:40]:
    print(" ", key, delta)
