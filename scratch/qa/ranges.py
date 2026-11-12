import random, managym, numpy as np
layout = managym.observation_layout()
defs = [d for d in managym.content_pool("world")["definitions"] if d.get("key") and not d["key"].startswith("token.")]
names = sorted({d["registry_name"] for d in defs})
print("world cards", len(names))
viol = {}
bad_games = 0
for g in range(12):
    rng = random.Random(g)
    def deck():
        picks = rng.sample(names, 10)
        lands = ["Mountain","Forest","Island","Plains","Swamp"]
        d = {l: 4 for l in lands}
        d.update({p: d.get(p, 0) + 4 for p in picks})
        return d
    env = managym.Env(seed=g)
    try:
        obs, info = env.reset([managym.PlayerConfig("A", deck()), managym.PlayerConfig("B", deck())])
    except Exception as e:
        bad_games += 1; print("reset", type(e).__name__, str(e)[:120]); continue
    for step in range(600):
        try:
            arr = env.encode_observation(obs)
        except Exception as e:
            print("encode", type(e).__name__, str(e)[:160]); break
        for name, spec in layout.items():
            a = arr[name]
            validk = f"{name}_valid"
            rows = a[arr[validk].astype(bool)] if validk in arr and a.ndim >= 2 and arr[validk].shape == a.shape[:-1] else a
            for cname, c in spec["columns"].items():
                v = rows[..., c["index"]]
                if v.size == 0: continue
                r = c["range"]
                if r == "0 or 1" and not np.isin(v, [0, 1]).all():
                    viol.setdefault((name, cname, r), set()).update(np.unique(v)[:5].tolist())
                import re
                m = re.fullmatch(r"(-?\d+)\.\.(\d+)", r)
                if m and ((v < int(m[1])) | (v > int(m[2]))).any():
                    viol.setdefault((name, cname, r), set()).update(np.unique(v)[:5].tolist())
                if r.startswith("nonnegative") and (v < 0).any():
                    viol.setdefault((name, cname, r), set()).update(v[v < 0][:5].tolist())
        try:
            obs, rew, term, trunc, info = env.step(rng.randrange(len(obs.action_space.actions)))
        except Exception as e:
            print("step", type(e).__name__, str(e)[:120]); break
        if term or trunc: break
print("violations:", viol)
ranges = sorted({c["range"] for s in layout.values() for c in s["columns"].values()})
print("range vocab:", ranges)
