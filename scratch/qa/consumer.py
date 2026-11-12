import sys, warnings, json
import managym

layout = managym.observation_layout()
definition = layout["agent_permanents"]["columns"]["definition"]
print(definition["index"], definition["kind"])
print(definition["encoding"])
print("torch loaded:", "torch" in sys.modules)

game = managym.Env(seed=42)
observation, info = game.reset([
    managym.PlayerConfig("Alice", {"Mountain": 40, "Lightning Bolt": 20}),
    managym.PlayerConfig("Bob", {"Forest": 40, "Llanowar Elves": 20}),
])
pool = {d["card_def_id"]: d for d in managym.content_pool("world")["definitions"]}
import random
rng = random.Random(0)
seen = set()
for step in range(400):
    arrays = game.encode_observation(observation)
    for side in ("agent_permanents", "opponent_permanents"):
        col = layout[side]["columns"]["definition"]["index"]
        valid = arrays[f"{side}_valid"].astype(bool)
        for ref in arrays[side][valid, col]:
            if ref:
                seen.add((side, pool[int(ref) - 1]["registry_name"]))
    # check every array named in the layout matches its shape/dtype
    for name, spec in layout.items():
        a = arrays[name]
        assert str(a.dtype) == spec["dtype"], (name, a.dtype, spec["dtype"])
        assert len(a.shape) == len(spec["shape"]), (name, a.shape, spec["shape"])
        for got, want in zip(a.shape, spec["shape"]):
            assert want is None or got == want, (name, a.shape, spec["shape"])
        if spec["columns"]:
            assert a.shape[-1] == len(spec["columns"]), (name, a.shape, len(spec["columns"]))
    missing = set(arrays) - set(layout)
    n = len(observation.action_space.actions)
    observation, reward, term, trunc, info = game.step(rng.randrange(n))
    if term or trunc:
        print("game over at step", step); break
print("unlisted arrays:", sorted(missing))
print("seen:", sorted(seen))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    assert managym.learning_catalog() == managym.catalog_tokens() or True
    managym.learning_record_names(); managym.learning_token_value("Flying") if False else None
    for x in w: print("warn:", x.category.__name__, x.message)
print(sorted(layout)[:50])
