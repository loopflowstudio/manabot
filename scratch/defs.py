from manabot.verify.action_cases import curriculum
import torch
from collections import defaultdict
seen = defaultdict(set)
for c in curriculum():
    o = c.observation
    ids = set(o["semantic_cards"][o["semantic_cards"]>0].long().tolist())
    for k in ("agent_permanents","opponent_permanents"):
        ids |= set(o[k][...,24].long().flatten().tolist()) - {0}
    seen[c.split] |= ids
    if c.family == "learn": print(c.split, c.variant, sorted(ids))
print("held-only", sorted(seen["held_out"] - seen["fit"]))
