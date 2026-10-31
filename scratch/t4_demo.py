"""Headless T4 demo: seeded games between decks of T4 cards through managym.Env."""

import json
import random
from collections import Counter

import manabot.managym as managym

rows = [json.loads(l) for l in open("docs/rules/tla/allocation.jsonl")]
t4 = sorted({r["card_name"] for r in rows if r.get("planned_pr") == "T4"})
assert len(t4) == 75, len(t4)
assert managym.unsupported_cards(t4) == [], managym.unsupported_cards(t4)
print("world", managym.world_identity(), "T4 names", len(t4), "unsupported 0")

basics = ["Plains", "Island", "Swamp", "Mountain", "Forest"]


def deck(names: list[str]) -> dict[str, int]:
    d = {n: 1 for n in names}
    for b in basics:
        d[b] = 8
    return d


chunks = [t4[i::4] for i in range(4)]
games = errors = 0
turns = []
frame_kinds: Counter[int] = Counter()
action_desc: Counter[str] = Counter()
stack_seen: set[str] = set()
for seed in range(40):
    a, b = chunks[seed % 4], chunks[(seed + 1) % 4]
    env = managym.Env(seed=seed, skip_trivial=True)
    rng = random.Random(seed)
    try:
        obs, _ = env.reset([managym.PlayerConfig("A", deck(a)), managym.PlayerConfig("B", deck(b))])
        for _ in range(3000):
            for f in obs.continuation:
                frame_kinds[f.kind] += 1
            for s in obs.stack_objects:
                if s.description:
                    stack_seen.add(s.description.split(":")[0][:40])
            acts = obs.action_space.actions
            i = rng.randrange(len(acts))
            action_desc[str(acts[i].action_type).split(".")[-1]] += 1
            obs, _, term, trunc, _ = env.step(i)
            if term or trunc:
                break
        turns.append(obs.turn.turn_number)
        games += 1
    except Exception as e:  # report every failure
        errors += 1
        print("seed", seed, type(e).__name__, str(e)[:300])
print("games", games, "errors", errors, "mean turns", sum(turns) / max(1, len(turns)))
print("continuation frame kinds", dict(sorted(frame_kinds.items())))
print("actions", action_desc.most_common(12))
print("distinct stack descriptions", len(stack_seen))

# Policy input: a waiting delayed trigger reaches the current (v5) tensor.
from manabot.env.observation import ObservationSpace

space = ObservationSpace()
found = 0
for seed in range(40):
    a, b = chunks[seed % 4], chunks[(seed + 1) % 4]
    env = managym.Env(seed=seed, skip_trivial=True)
    rng = random.Random(seed)
    obs, _ = env.reset([managym.PlayerConfig("A", deck(a)), managym.PlayerConfig("B", deck(b))])
    for _ in range(3000):
        if any(f.kind == 10 for f in obs.continuation):
            enc = space.encode(obs)
            print("seed", seed, "tensor keys", sorted(enc)[:12])
            found = 1
            break
        acts = obs.action_space.actions
        obs, _, term, trunc, _ = env.step(rng.randrange(len(acts)))
        if term or trunc:
            break
    if found:
        break
