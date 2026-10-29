"""Compiled IR before/after is identical apart from basic-land key renames."""
import json, sys
RENAME = {f"tla.{b}": f"lea.{b}" for b in ("plains", "island", "swamp", "mountain", "forest")}
def norm(ir, rename):
    keys = [rename.get(d["semantic_key"], d["semantic_key"]) for d in ir["definitions"]]
    def fix(v, field=None):
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                if k in ("definition_index", "semantic_index") and isinstance(x, int):
                    out[k] = keys[x]
                elif k == "program_index" or k == "program_indexes":
                    continue
                elif k in ("semantic_key", "definition") and isinstance(x, str):
                    out[k] = fix(rename.get(x, x))
                elif k == "program_key" and isinstance(x, str):
                    head, _, tail = x.rpartition(".")
                    out[k] = rename.get(head, head) + "." + tail
                else:
                    out[k] = fix(x)
            return out
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, str):
            for old, new in rename.items():
                if v.startswith(old + "."):
                    return new + v[len(old):]
        return v
    defs = {}
    for d, key in zip(ir["definitions"], keys):
        row = fix(d)
        row["programs"] = sorted((fix(ir["programs"][i]) for i in d["program_indexes"]), key=json.dumps)
        defs[key] = row
    rest = {k: fix(v) for k, v in ir.items() if k not in ("definitions", "programs", "ir_hash", "source_hash")}
    for name in ("pool", "token_templates"):
        if isinstance(rest.get(name), list):
            rest[name] = sorted(rest[name])
    for deck in rest.get("decks", []):
        for field in ("cards", "sideboard"):
            if field in deck:
                deck[field] = sorted(deck[field], key=json.dumps)
    return {"definitions": defs, "rest": rest}
bad = 0
for pack in sys.argv[3:]:
    before = json.load(open(f"{sys.argv[1]}/{pack}.json"))
    after = json.load(open(f"{sys.argv[2]}/{pack}.json"))
    a, b = norm(before, RENAME), norm(after, {})
    same = a == b
    bad += not same
    print(f"{pack}: {'equivalent' if same else 'DIFFERENT'} (hash changed: {before['ir_hash'] != after['ir_hash']})")
    if not same:
        for k in sorted(set(a['definitions']) | set(b['definitions'])):
            if a['definitions'].get(k) != b['definitions'].get(k): print('  def', k)
        if a['rest'] != b['rest']:
            for k in set(a['rest'])|set(b['rest']):
                if a['rest'].get(k)!=b['rest'].get(k): print('  field', k, str(a['rest'].get(k))[:200], '|', str(b['rest'].get(k))[:200])
sys.exit(bad)
