"""Print the distinct action descriptions offered in random T2-deck games, plus foretell viewer checks."""
import collections, random, sys, importlib.util
spec = importlib.util.spec_from_file_location("d", "scratch/tools/t2_demo.py")
import managym
src = open("scratch/tools/t2_demo.py").read().split("def main")[0]
exec(src.replace("T2 = json.load(open(sys.argv[1]))", "T2 = []"))
descs = collections.Counter()
fore = []
for seed in range(int(sys.argv[1])):
    env = managym.Env(seed=seed)
    obs, _ = env.reset([managym.PlayerConfig("red", RED), managym.PlayerConfig("green", GREEN)])
    rng = random.Random(seed); done = False; steps = 0
    while not done and steps < 3000:
        for a in obs.action_space.actions:
            descs[(str(a.action_type), a.description)] += 1
        i = rng.randrange(len(obs.action_space.actions))
        before = obs
        obs, _, done, trunc, _ = env.step(i); done = done or trunc; steps += 1
        a = before.action_space.actions[i]
        if a.description and a.description.startswith("Foretell") and len(fore) < 3 and not done:
            owner = env.observation_for_player(before.agent.player_index)
            other = env.observation_for_player(1 - before.agent.player_index)
            mine = [(c.name, c.face_down, c.cast_permission.foretold if c.cast_permission else None) for c in owner.agent_cards if str(c.zone).endswith("EXILE")]
            theirs = [(c.name, c.face_down) for c in other.opponent_cards if str(c.zone).endswith("EXILE")]
            fore.append({"owner_exile": mine, "opponent_view_exile_cards": theirs, "opponent_view_zone_counts": list(other.opponent.zone_counts)})
for (t, d), n in sorted(descs.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
    if "PASS" not in t:
        print(n, t, d)
print("FORETELL VIEWS"); [print(f) for f in fore]
