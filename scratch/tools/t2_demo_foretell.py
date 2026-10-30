"""Trace one Foretell to completion and compare the owner's and opponent's views."""
import random
import managym
exec(open("scratch/tools/t2_demo.py").read().split("def main")[0].replace("T2 = json.load(open(sys.argv[1]))", "T2 = []"))
for seed in range(40):
    env = managym.Env(seed=seed)
    obs, _ = env.reset([managym.PlayerConfig("red", RED), managym.PlayerConfig("green", GREEN)])
    rng = random.Random(seed); done = False; owner = None; steps = 0
    while not done and steps < 3000:
        acts = obs.action_space.actions
        f = [i for i, a in enumerate(acts) if (a.description or "").startswith("Foretell")]
        i = f[0] if f and owner is None else rng.randrange(len(acts))
        if f and owner is None:
            owner = obs.agent.player_index; print("seed", seed, "turn", obs.turn.turn_number, "choose:", acts[i].description)
        obs, _, done, _, _ = env.step(i); steps += 1
        if owner is not None and not done:
            mine = env.observation_for_player(owner)
            ex = [c for c in mine.agent_cards if c.face_down]
            if ex:
                theirs = env.observation_for_player(1 - owner)
                c = ex[0]
                print("owner sees:", c.name, "zone", c.zone, "face_down", c.face_down, "foretold grant", c.cast_permission and c.cast_permission.foretold, "after_turn", c.cast_permission and c.cast_permission.after_turn)
                print("opponent sees exile cards:", [(o.name, o.face_down) for o in theirs.opponent_cards if "EXILE" in str(o.zone).upper()], "opponent zone_counts", list(theirs.opponent.zone_counts))
                js = theirs.toJSON(); print("'Sozin' in opponent JSON:", "Sozin" in js)
                enc = env.encode_observation(theirs, 5); print("v5 buffers:", {k: v.shape for k, v in enc.items()})
                # later: owner casts it from exile?
                cast = None
                while not done and steps < 3000:
                    acts = obs.action_space.actions
                    cs = [j for j, a in enumerate(acts) if obs.agent.player_index == owner and str(a.action_type).endswith("PriorityCastSpell") and any(cc.id in a.focus and cc.face_down for cc in obs.agent_cards)]
                    if cs:
                        cast = (obs.turn.turn_number, acts[cs[0]].focus); obs, _, done, _, _ = env.step(cs[0]); break
                    obs, _, done, _, _ = env.step(rng.randrange(len(acts))); steps += 1
                print("cast from foretell offered/taken at (turn, focus):", cast)
                raise SystemExit
print("no foretell completed")
