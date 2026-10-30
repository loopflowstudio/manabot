"""Tally offered and taken casts/activations by the zone of the focused card in random T2-deck games."""
import collections, random, sys
import managym
exec(open("scratch/tools/t2_demo.py").read().split("def main")[0].replace("T2 = json.load(open(sys.argv[1]))", "T2 = []"))
offered = collections.Counter(); taken = collections.Counter()
for seed in range(int(sys.argv[1])):
    env = managym.Env(seed=seed)
    obs, _ = env.reset([managym.PlayerConfig("red", RED), managym.PlayerConfig("green", GREEN)])
    rng = random.Random(seed); done = False; steps = 0
    while not done and steps < 3000:
        cards = {c.id: c for c in obs.agent_cards}
        acts = obs.action_space.actions
        keys = []
        for a in acts:
            t = str(a.action_type).split(".")[-1]
            z = [cards[f] for f in a.focus if f in cards]
            k = None
            if t in ("PriorityCastSpell", "PriorityPlayLand", "PriorityActivateAbility") and z and not str(z[0].zone).endswith("Hand") and not str(z[0].zone).endswith("Battlefield"):
                k = (t, str(z[0].zone).split(".")[-1], z[0].name)
                offered[k] += 1
            keys.append(k)
        i = rng.randrange(len(acts))
        if keys[i]: taken[keys[i]] += 1
        obs, _, done, tr, _ = env.step(i); done = done or tr; steps += 1
print("offered (action, zone, card): count  [taken]")
for k, n in sorted(offered.items()):
    print(f"  {k}: {n} [{taken[k]}]")
