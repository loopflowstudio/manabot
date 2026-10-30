"""Headless afbd6a demo: definitions, tokens, statics and a played w17 game."""
import json
import managym

defs = json.loads(managym.card_definitions_json())
items = defs if isinstance(defs, list) else list(defs.values()) if isinstance(defs, dict) else defs
print("definitions:", len(items))
sample = items[0]
print("definition keys:", sorted(sample.keys()) if isinstance(sample, dict) else type(sample))
tokens = [d for d in items if d.get("is_token")]
print("tokens:", sorted(d.get("name") for d in tokens))
statics = {}
for d in items:
    for s in d.get("statics", []) or []:
        k = next(iter(s)) if isinstance(s, dict) else s
        statics.setdefault(k, d.get("name"))
print("static variants:", statics)
costs = {}
for d in items:
    for c in d.get("additional_costs", []) or []:
        k = next(iter(c)) if isinstance(c, dict) else c
        costs.setdefault(k, d.get("name"))
print("cost variants:", costs)

import random
from manabot.env.match import Match
from manabot.infra.config import MatchConfig

config = MatchConfig.authored("ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies")
rng = random.Random(0)
for seed in range(4):
    env = managym.Env(seed=seed)
    obs, _ = env.reset(Match(config).to_rust())
    steps, done, reward = 0, False, 0.0
    while not done and steps < 5000:
        n = len(obs.action_space.actions)
        obs, reward, term, trunc, info = env.step(rng.randrange(n))
        done = term or trunc
        steps += 1
    print(f"seed {seed}: decisions={steps} done={done} reward={reward}")
