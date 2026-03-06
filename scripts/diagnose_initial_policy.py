"""Diagnose initial policy behavior: does a fresh agent play lands/spells?"""

import torch
import numpy as np

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.env.observation import ActionEnum
from manabot.infra import Hypers
from manabot.model.agent import Agent
from manabot.verify.util import (
    STANDARD_DECK,
    _select_agent_action,
    _action_type_name,
    _land_available,
    _spell_available,
    _attack_available,
    _is_land_action,
    _is_spell_action,
    _is_attack_action,
    step_with_fallback,
)

hypers = Hypers(
    match={"hero_deck": STANDARD_DECK, "villain_deck": STANDARD_DECK},
    train={"opponent_policy": "random"},
    agent={"attention_on": False},
)

obs_space = ObservationSpace(hypers.observation)
match = Match(hypers.match)
reward = Reward(hypers.reward)

agent = Agent(obs_space, hypers.agent)
agent.eval()

env = Env(match, obs_space, reward, seed=42, auto_reset=False,
          enable_profiler=False, enable_behavior_tracking=False)

num_games = 20
totals = {
    "hero_actions": 0,
    "land_available": 0, "land_played": 0,
    "spell_available": 0, "spell_cast": 0,
    "attack_available": 0, "attack_declared": 0,
}

# Also track first 5 actions per game to see what's happening
for g in range(num_games):
    obs, _ = env.reset(seed=42 + g)
    done = False
    steps = 0
    game_log = []

    while not done:
        active = int(env.last_raw_obs.agent.player_index)
        if active == 0:
            # Check what's available
            has_land = _land_available(obs)
            has_spell = _spell_available(obs)
            has_attack = _attack_available(obs)

            # Get agent's action (deterministic = argmax)
            action = _select_agent_action(agent, obs, deterministic=True)
            action_name = _action_type_name(obs, action)
            played_land = _is_land_action(obs, action)
            cast_spell = _is_spell_action(obs, action)
            declared_attack = _is_attack_action(obs, action)

            totals["hero_actions"] += 1
            if has_land:
                totals["land_available"] += 1
            if played_land:
                totals["land_played"] += 1
            if has_spell:
                totals["spell_available"] += 1
            if cast_spell:
                totals["spell_cast"] += 1
            if has_attack:
                totals["attack_available"] += 1
            if declared_attack:
                totals["attack_declared"] += 1

            if len(game_log) < 8:
                avail = []
                if has_land: avail.append("land")
                if has_spell: avail.append("spell")
                if has_attack: avail.append("attack")
                game_log.append(f"  step {steps}: chose={action_name}, available=[{','.join(avail)}]")
        else:
            from manabot.env import build_opponent_policy
            opponent = build_opponent_policy("random")
            action = opponent(obs)

        try:
            obs, _, terminated, truncated, info = step_with_fallback(env, action)
        except Exception:
            break
        steps += 1
        done = bool(terminated or truncated)

    if g < 5:
        print(f"\nGame {g} ({steps} steps):")
        for line in game_log:
            print(line)

env.close()

print("\n--- Totals across {} games ---".format(num_games))
print(f"Hero actions: {totals['hero_actions']}")
for key in ["land", "spell", "attack"]:
    avail = totals[f"{key}_available"]
    took = totals[f"{key}_{'played' if key == 'land' else 'cast' if key == 'spell' else 'declared'}"]
    rate = took / avail if avail > 0 else 0
    print(f"  {key}: available={avail}, took={took}, rate={rate:.2%}")

# Also check: what do the raw logits look like for a typical decision?
print("\n--- Sample logits for first decision ---")
env2 = Env(match, obs_space, reward, seed=99, auto_reset=False,
           enable_profiler=False, enable_behavior_tracking=False)
obs, _ = env2.reset(seed=99)

device = torch.device("cpu")
tensor_obs = {
    k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
    for k, v in obs.items()
}
with torch.no_grad():
    logits, value = agent.forward(tensor_obs)
    raw_logits = agent.last_raw_logits[0]

print(f"  Value estimate: {value.item():.4f}")

# Check valid actions and their types with logits
valid = obs["actions_valid"]
actions = obs["actions"]
print(f"\n  Valid actions (with raw logits):")
for i in range(actions.shape[0]):
    if valid[i] > 0:
        name = _action_type_name(obs, i)
        logit = raw_logits[i].item()
        print(f"    [{i}] {name:30s}  logit={logit:+.4f}  features={actions[i][:6]}")

# Sample 20 stochastic actions to see distribution
print(f"\n  Stochastic sampling (20 draws):")
action_counts = {}
for _ in range(20):
    a, _, _, _ = agent.get_action_and_value(tensor_obs, deterministic=False)
    name = _action_type_name(obs, int(a.item()))
    action_counts[name] = action_counts.get(name, 0) + 1
for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
    print(f"    {name}: {count}/20")

env2.close()
