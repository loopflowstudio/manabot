"""Split per-card rules test files into top-level items and map tests to sets."""
import json, re, sys
from pathlib import Path
import managym

ROOT = Path("managym")
FILES = ["elemental_cards","interaction_spells","sacrifice_cards","stage2_cards","stage3_cards","temur_cards","tla_cards","tla_content","ur_lessons_increment"]

def name_sets():
    m = {}
    for row in managym.content_pool("tla-pool-v1")["definitions"]:
        k = row["semantic_key"]
        if k: m[row["registry_name"]] = k.split(".")[0]
    for f in sorted((ROOT/"src/cardsets").glob("*.rs")):
        for n in re.findall(r'name: "([^"]+)"\.to_string\(\)', f.read_text()):
            m.setdefault(n, f.stem)
    return m

def items(text):
    """Top-level items with their preceding attributes/doc comments."""
    lines = text.split("\n")
    out, i, n = [], 0, len(lines)
    header = []
    # header: use lines until first item
    while i < n:
        start = i
        # collect leading comments/attrs/blank
        while i < n and (lines[i].startswith("//") or lines[i].startswith("#[") or lines[i].strip()=="" ) and not lines[i].startswith("//!"):
            i += 1
        if i >= n: break
        if lines[i].startswith("//!") :
            out.append(("doc", "\n".join(lines[start:i+1]))); i += 1; continue
        # item starting at i: find end by brace depth, or semicolon for single-line
        j = i; depth = 0; seen = False
        while j < n:
            l = re.sub(r'"(\\.|[^"\\])*"', '""', lines[j])
            l = re.sub(r"//.*", "", l)
            depth += l.count("{") - l.count("}")
            if "{" in l: seen = True
            if (seen and depth == 0) or (not seen and l.rstrip().endswith(";") and depth==0):
                break
            j += 1
        out.append(("item", "\n".join(lines[start:j+1])))
        i = j + 1
    return out

if __name__ == "__main__":
    ns = name_sets()
    names = sorted(ns, key=len, reverse=True)
    report = []
    for f in FILES:
        text = (ROOT/f"tests/rules/{f}.rs").read_text()
        for kind, body in items(text):
            m = re.search(r"\bfn (\w+)", body)
            fn = m.group(1) if m else None
            is_test = "#[test]" in body
            lits = set(re.findall(r'"([^"]+)"', body))
            cards = sorted({l for l in lits if l in ns})
            sets = sorted({ns[c] for c in cards})
            report.append(dict(file=f, fn=fn, test=is_test, kind=kind, cards=cards, sets=sets, first=body.strip().split("\n")[0][:80]))
    json.dump(report, open("scratch/cardtests/items.json","w"), indent=1)
    for r in report:
        if not r["test"]:
            print("HELPER", r["file"], r["fn"], r["first"])
