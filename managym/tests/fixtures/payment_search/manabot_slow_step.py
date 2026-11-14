"""Reproduce a slow engine step in manabot: one `Env.step` that takes tens of seconds.

Needs only manabot (tested on 0.2.1 from PyPI, Python 3.13, Apple M3 Max):

    pip install manabot==0.2.1
    python manabot_slow_step.py

It replays one game of two 40-card Avatar: The Last Airbender decks from a fixed seed and a fixed list of
actions, timing every step. The last action activates a mana ability with 11 lands and two other mana
sources in play, and that step is the slow one. Stacks sampled during it are in
`Game::tick_priority -> Game::payment_model`.
"""

import time

from manabot import managym

SEED = 5064105440874770157
DECKS = [
    {
        "Airbending Lesson": 1,
        "Appa, Loyal Sky Bison": 1,
        "Avatar Enthusiasts": 2,
        "Cat-Owl": 1,
        "Compassionate Healer": 2,
        "Curious Farm Animals": 1,
        "Deadly Precision": 2,
        "Hakoda, Selfless Commander": 1,
        "Invasion Reinforcements": 1,
        "Kyoshi Battle Fan": 1,
        "Kyoshi Warriors": 1,
        "Merchant of Many Hats": 2,
        "Misty Palms Oasis": 1,
        "Path to Redemption": 1,
        "Plains": 9,
        "Rabaroo Troop": 1,
        "Razor Rings": 1,
        "Swamp": 7,
        "Swampsnare Trap": 1,
        "Team Avatar": 1,
        "The Walls of Ba Sing Se": 1,
        "Water Tribe Captain": 1
    },
    {
        "Aang, at the Crossroads": 1,
        "Airship Engine Room": 1,
        "Bender's Waterskin": 1,
        "Cycle of Renewal": 1,
        "Earth Kingdom General": 1,
        "Earth Rumble": 1,
        "First-Time Flyer": 2,
        "Foggy Swamp Vinebender": 1,
        "Forest": 7,
        "Hermitic Herbalist": 2,
        "Island": 5,
        "Katara, Water Tribe's Hope": 1,
        "Kyoshi Village": 1,
        "Meditation Pools": 1,
        "Mountain": 1,
        "Plains": 1,
        "Raucous Audience": 2,
        "Rocky Rebuke": 2,
        "Spirit Water Revival": 1,
        "Suki, Kyoshi Warrior": 1,
        "The Boulder, Ready to Rumble": 1,
        "Toph, the First Metalbender": 1,
        "Walltop Sentries": 2,
        "Waterbending Lesson": 1,
        "White Lotus Reinforcements": 1
    }
]
ACTIONS = [0, 1, 1, 3, 0, 3, 1, 1, 0, 1, 0, 1, 4, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 2, 0, 0, 1, 1, 1, 3, 0, 6, 0, 0, 0, 0, 2, 0, 1, 0, 1, 2, 0, 0, 1, 2, 2, 0, 1, 1, 0, 1, 1, 1, 0, 2, 1, 0, 0, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 2, 1, 2, 1, 2, 1, 3, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 3, 1, 1, 1, 0, 0, 1, 6, 5, 1, 0, 1, 1, 5, 0, 1, 1, 4, 1, 1, 1, 1, 0, 0, 1, 2, 1, 0, 1, 2, 0, 0, 1, 1, 1, 0, 4, 0, 1, 2, 5, 0, 0, 3, 1, 0, 0, 4, 2, 2, 0, 0, 1, 0, 1, 1, 4, 1, 2, 9, 2, 0, 1, 2, 4, 0, 11, 0, 3, 1, 2, 1, 1, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1, 9, 1, 1, 1, 0, 0, 5, 5, 2, 0, 2, 1, 2, 11, 5, 1, 2, 2, 2, 2, 2, 0, 1, 0, 1, 0, 2, 2, 2, 1, 3, 1, 1, 0, 2, 0, 0, 8, 1, 0, 3, 1, 2, 2, 4, 6, 4, 1, 3, 0, 11, 0, 0, 0, 0, 1, 1, 1, 0, 2, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 4, 5, 3, 5, 4, 0, 3, 0, 3, 3, 3, 3, 4, 0, 1, 3, 4, 0, 2, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 2, 0, 2, 1, 1, 1, 2, 13, 5, 4, 4, 2, 0, 0, 1, 3, 1, 1, 6, 1, 0, 1, 2, 5, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 2, 1, 7, 7, 1, 0, 5, 3, 4, 5, 4, 4, 2, 4, 2, 3, 0, 4, 1, 0, 0, 0, 5, 1, 0, 1, 1, 4, 2, 1, 1, 0, 1, 1, 0, 2, 0, 1, 2, 0, 1, 3]

players = [
    managym.PlayerConfig(f"seat {seat}", deck, world=managym.WORLD_SETUP, disclosure="closed")
    for seat, deck in enumerate(DECKS)
]
env = managym.Env(seed=SEED)
env.reset(players)
slowest = 0.0
for number, action in enumerate(ACTIONS, start=1):
    began = time.perf_counter()
    env.step(action)
    seconds = time.perf_counter() - began
    slowest = max(slowest, seconds)
    if seconds > 0.25:
        print(f"step {number} of {len(ACTIONS)}: action {action} took {seconds:.2f} s")
print(f"slowest step: {slowest:.2f} s")
