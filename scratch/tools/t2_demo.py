"""Headless T2 demo: random-policy games with T2 card decks through managym.Env.

Tallies T2 action kinds by description, checks the opponent's view of
face-down foretold cards, and encodes each observation at policy input v5.
"""

import collections
import json
import random
import sys

import managym

T2 = json.load(open(sys.argv[1]))
RED = {"Sozin's Comet": 2, "Fire Nation Attacks": 2, "Zuko, Exiled Prince": 2,
       "Iroh, Grand Lotus": 2, "Mongoose Lizard": 2, "Rough Rhino Cavalry": 2,
       "Barrels of Blasting Jelly": 2, "Azula, Cunning Usurper": 2,
       "Canyon Crawler": 2, "Hama, the Bloodbender": 2, "Professor Zei, Anthropologist": 2,
       "Mountain": 10, "Swamp": 4, "Island": 4}
GREEN = {"Rockalanche": 2, "Badgermole Cub": 2, "Toph, the Blind Bandit": 2,
         "Earthbending Lesson": 2, "Saber-Tooth Moose-Lion": 2, "Giant Koi": 2,
         "Hermitic Herbalist": 2, "White Lotus Hideout": 2, "Jasmine Dragon Tea Shop": 2,
         "Serpent of the Pass": 2, "Spirit Water Revival": 2, "Solstice Revelations": 2,
         "Forest": 10, "Island": 6}

def main() -> None:
    names = sorted(set(RED) | set(GREEN))
    print("unsupported among deck cards:", managym.unsupported_cards(names))
    print("unsupported among all 51 T2 cards:", managym.unsupported_cards(T2))
    kinds = collections.Counter()
    samples = {}
    seen_cards = collections.Counter()
    leaks = 0
    face_down_checks = 0
    wins = collections.Counter()
    errors = []
    for seed in range(int(sys.argv[2])):
        env = managym.Env(seed=seed)
        obs, _ = env.reset([managym.PlayerConfig("red", RED), managym.PlayerConfig("green", GREEN)])
        rng = random.Random(seed)
        done = False
        steps = 0
        try:
            while not done and steps < 3000:
                env.encode_observation(obs, 5)
                for c in obs.opponent_cards:
                    if c.face_down:
                        face_down_checks += 1
                        if c.name or c.registry_key:
                            leaks += 1
                for a in obs.action_space.actions:
                    d = (a.description or "").lower()
                    for k in ("foretell", "flashback", "cycling", "kick", "from exile", "graveyard", "waterbend", "exhaust"):
                        if k in d:
                            kinds[k] += 1
                            samples.setdefault(k, a.description)
                i = rng.randrange(len(obs.action_space.actions))
                obs, _, done, trunc, _ = env.step(i)
                done = done or trunc
                steps += 1
            wins[env.winner_index()] += 1
        except Exception as e:  # report, keep going
            errors.append((seed, repr(e)[:200]))
    print("games:", sys.argv[2], "winners:", dict(wins), "errors:", errors[:5], len(errors))
    print("face-down opponent card views:", face_down_checks, "with name/key leaked:", leaks)
    print("offered T2 action kinds:", dict(kinds))
    for k, v in samples.items():
        print(f"  {k}: {v}")

main()
