"""Random-play admitted bonus-sheet corpus decks; encode v9/v10 and render text each decision."""
from __future__ import annotations
import collections, json, random, re, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from experiments.tla_corpus import Corpus
import managym
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.semantic.text_observation import capture_text_decision, TextEncodingError

corpus = Corpus.model_validate_json(Path("experiments/data/mtg213/corpus-v1/corpus.json").read_text())
bonus = {json.loads(l)["card_name"] for l in Path("managym/cards/tests/fixtures/tle-bonus-sheet.jsonl").read_text().splitlines() if l.strip()}
decks = []
for t in corpus.trophies:
    names = set(re.findall(r"^\d+ (.+)$", t.decklist.source, re.M))
    if not names & bonus: continue
    try:
        (imp,) = managym.admit_decklists([(t.event_id, managym.Decklist(t.decklist.source, t.decklist.filename))])
    except managym.DeckImportError:
        continue
    decks.append((t.event_id, imp.deck, names & bonus))
def debug(env, obs, msg):
    print("DEBUG", msg)
    v = env.current_agent_index()
    print(" opp hand visible:", [(c.name, c.id) for c in obs.opponent_cards if int(c.zone) == 1])
    print(" my perms:", [p.name for p in obs.agent_permanents][:30])
    print(" opp perms:", [p.name for p in obs.opponent_permanents][:30])
    print(" stack:", [s.description for s in obs.stack_objects])
    try:
        cap = json.loads(env.capture_decision(v, 0).to_json()); print(" stage:", cap["continuation"].get("stage"))
        print(" cast perms:", json.dumps([o for o in json.dumps(cap["observation"])[:0]]))
    except Exception as e: print(" capture:", e)
    try:
        ctx = json.loads(env.decision_context_json(v)); print(" ctx stage:", ctx.get("stage"), str(ctx)[:600])
    except Exception as e: print(" ctx:", e)
spaces = {v: ObservationSpace(ObservationConfig(policy_input_version=v)) for v in (9, 10)}
DUMP = True; dumped = set(); full_val = collections.Counter(); xnames = collections.Counter(); errors = collections.Counter(); samples = {}
games = int(sys.argv[1]) if len(sys.argv) > 1 else len(decks)
rng = random.Random(286)
played = collections.Counter(); steps_total = 0; finished = 0; text_n = 0
for g in range(games):
    a = decks[g % len(decks)]; b = decks[rng.randrange(len(decks))]
    env = managym.Env(seed=g, skip_trivial=True)
    try:
        obs, _ = env.reset([a[1].player(), b[1].player()])
    except Exception as e:
        errors["reset:" + type(e).__name__] += 1; samples.setdefault("reset", f"{a[0]} {e}"); continue
    for step in range(4000):
        for v, sp in spaces.items():
            try:
                sp.encode(obs)
            except Exception as e:
                k = f"encode-v{v}:{type(e).__name__}"; errors[k] += 1; samples.setdefault(k, str(e)[:300])
        if step % 7 == 0:
            try:
                capture_text_decision(env); text_n += 1
            except TextEncodingError as e:
                msg = str(e).split("\n")[0][:120]
                if "validation error" in msg: full_val[str(e).split("\n")[1:3].__repr__()[:200]] += 1
                if DUMP and "validation" not in msg and msg not in dumped:
                    dumped.add(msg); traceback.print_exc(); debug(env, obs, msg); print("CAPTURE", env.current_agent_index(), [str(x)[:200] for x in obs.action_space.actions][:6])
                errors["text:" + msg] += 1; samples.setdefault("text:" + msg, f"game {g} {a[0]} vs {b[0]} step {step}")
                try:
                    cap = json.loads(env.capture_decision(env.current_agent_index(), 0).to_json())
                    ds = cap["definitions"]; ds = ds.values() if isinstance(ds, dict) else ds
                    for d in ds:
                        if isinstance(d, dict) and (d.get("mana_cost") or {}).get("x"): xnames[d.get("name")] += 1
                except Exception:
                    pass
        if obs.game_over: finished += 1; break
        n = len(obs.action_space.actions)
        try:
            obs, *_ = env.step(rng.randrange(n))
        except Exception as e:
            errors["step:" + type(e).__name__] += 1; samples.setdefault("step:" + type(e).__name__, f"{a[0]} vs {b[0]} step {step}: {e}"[:500]); break
        steps_total += 1
    for n in a[2] | b[2]: played[n] += 1
print(f"games {games} finished {finished} steps {steps_total} text renders {text_n}")
print("errors", dict(errors))
for k, v in samples.items(): print(" ", k, "->", v)
print("validation kinds", dict(full_val)); print("x cards", dict(xnames)); print("bonus names in games:", len(played))
