"""Move per-card rules tests into managym/tests/cards/<set>.rs (one-off)."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "scratch/cardtests")
from analyze import items, FILES

T = Path("managym/tests")
ORIGIN = {"elemental_cards":"elemental","interaction_spells":"interaction","sacrifice_cards":"sacrifice",
          "stage2_cards":"stage2","stage3_cards":"stage3","temur_cards":"temur","tla_cards":"triggers",
          "tla_content":"clauses","ur_lessons_increment":"ur_lessons"}
assign = json.load(open("scratch/cardtests/assign.json"))

def rewrite(code):
    code = code.replace("super::elemental_cards::", "crate::support::elemental::")
    code = code.replace("use super::{", "use crate::support::{")
    code = code.replace("use super::helpers", "use crate::support::helpers")
    code = re.sub(r"\belemental_cards::", "elemental::", code)
    assert "super::" not in code, code[code.index("super::")-200:code.index("super::")+80]
    return code

def publicize(item):
    lines = item.split("\n")
    for i, l in enumerate(lines):
        if l.startswith("//") or l.startswith("#[") or not l.strip():
            continue
        lines[i] = re.sub(r"^(pub\(super\) |pub )?(fn|struct|const|enum|type|static) ", r"pub(crate) \2 ", l)
        break
    return "\n".join(lines)

def item_name(item):
    m = re.search(r"^(?:pub(?:\(\w+\))? )?(?:fn|struct|const|enum|type|static) (\w+)", item, re.M)
    return m.group(1) if m else None

origin_parts = {}
for f in FILES:
    text = (T / f"rules/{f}.rs").read_text()
    parts = items(text)
    header, uses, helpers, tests = [], [], [], []
    for kind, body in parts:
        b = body.strip("\n")
        first = [l for l in b.split("\n") if not l.startswith("//") and l.strip()]
        if kind == "doc" or not first:
            header.append(b.replace("//!", "//"))
        elif first[0].startswith("use "):
            # leading comment lines of a use item are file header
            lines = b.split("\n"); k = next(i for i, l in enumerate(lines) if l.startswith("use "))
            if k: header.append("\n".join(lines[:k]))
            uses.append(rewrite("\n".join(lines[k:])))
        elif "#[test]" in b:
            tests.append((item_name(b), rewrite(b)))
        else:
            helpers.append(rewrite(b))
    assert sum("#[test]" in t for _, t in tests) == text.count("#[test]"), f
    header = [h for h in "\n".join(header).split("\n") if not re.match(rf"^// {f}\.rs\s*$", h)]
    while header and not header[0].strip(): header.pop(0)
    origin_parts[f] = dict(header="\n".join(header).strip("\n"), uses=uses, helpers=helpers, tests=tests)

# support modules
sup = T / "support"; sup.mkdir(exist_ok=True)
(T / "rules/helpers.rs").rename(sup / "helpers.rs")
mods = ["helpers"]
for f, p in origin_parts.items():
    if not p["helpers"]:
        continue
    name = ORIGIN[f]; mods.append(name)
    body = "\n".join(p["uses"]) + "\n\n" + "\n\n".join(publicize(h) for h in p["helpers"]) + "\n"
    (sup / f"{name}.rs").write_text(body)
(sup / "mod.rs").write_text(
    "//! Scenario builders and directed-play helpers shared by the rules and card\n"
    "//! test binaries. Each binary uses a subset, so unused helpers are expected.\n"
    "#![allow(dead_code)]\n\n" + "".join(f"pub mod {m};\n" for m in sorted(mods)))

# destination files
dest = defaultdict(lambda: defaultdict(list))
for f, p in origin_parts.items():
    for name, code in p["tests"]:
        dest[assign[f"{f}::{name}"]][f].append(code)

def section(f, codes, flat):
    p = origin_parts[f]
    helper_names = sorted(n for n in (item_name(h) for h in p["helpers"]) if n)
    uses = list(p["uses"])
    if helper_names:
        uses.append(f"use crate::support::{ORIGIN[f]}::{{{', '.join(helper_names)}}};")
    body = "\n".join(uses) + "\n\n" + "\n\n".join(codes) + "\n"
    if flat:
        return body
    head = ""
    return f"{head}mod {ORIGIN[f]} {{\n{body}}}\n"

cards = T / "cards"; cards.mkdir(exist_ok=True)
for s, by_origin in sorted(dest.items()):
    flat = len(by_origin) == 1
    text = "\n".join(section(f, codes, flat) for f, codes in sorted(by_origin.items()))
    if s == "rules":
        (T / "rules/engine_events.rs").write_text(text)
    else:
        (cards / f"{s}.rs").write_text(text)
for f in FILES:
    (T / f"rules/{f}.rs").unlink()
print({s: sum(len(c) for c in d.values()) for s, d in dest.items()})
print(sorted(s for s in dest if s != "rules"))
