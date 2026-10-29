import json, re, collections, sys
sys.path.insert(0, "scratch/cardtests")
from analyze import name_sets
ns = name_sets()
BASICS = {"Plains","Island","Swamp","Mountain","Forest"}
def snake(n): return re.sub(r"[^a-z0-9]+","_",n.lower()).strip("_")
def words(n): return [w for w in snake(n).split("_") if len(w) > 2 and w not in {"the","of","and"}]
OVERRIDE = json.load(open("scratch/cardtests/overrides.json")) if __import__("os").path.exists("scratch/cardtests/overrides.json") else {}
EMPTY_DEFAULT = {"stage2_cards":"tla","stage3_cards":"tla","ur_lessons_increment":"tla"}
def classify(x):
    key = f"{x['file']}::{x['fn']}"
    if key in OVERRIDE: return OVERRIDE[key], "override"
    real = [c for c in x["cards"] if ns[c] != "token" and c not in BASICS]
    fn = x["fn"]
    hits = [c for c in real if any(w in fn.split("_") for w in words(c)[:1]) or snake(c) in fn]
    hit_sets = {ns[c] for c in hits}
    if len(hit_sets) == 1: return hit_sets.pop(), "name"
    if not real:
        if x["file"] in EMPTY_DEFAULT: return EMPTY_DEFAULT[x["file"]], "empty-default"
        return "rules", "no-card"
    cnt = collections.Counter(ns[c] for c in real)
    top = cnt.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]: return top[0][0], f"majority {dict(cnt)}"
    return "?", f"tie {dict(cnt)} hits={hits}"
if __name__ == "__main__":
    items = json.load(open("scratch/cardtests/items.json"))
    out = {}
    for x in items:
        if not x["test"]: continue
        s, why = classify(x)
        out[f"{x['file']}::{x['fn']}"] = s
        if why != "name" or "-v" in sys.argv:
            print(f"{s:6} {x['file']}::{x['fn']}  [{why}] {x['cards']}")
    json.dump(out, open("scratch/cardtests/assign.json","w"), indent=1)
    print(collections.Counter(out.values()))
