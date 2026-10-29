"""One-time generator: lowered CardDefinition Debug dumps -> engine-type Rust set files.

Usage: python3 scratch/conversion/generate.py <managym dir> <before.json> <ids.json>
Writes managym/src/cardsets/<set>.rs for every set present in the dump.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
DUMP = json.loads(Path(sys.argv[2]).read_text())
IDS = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else {}
FIRST = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else {}
# Cards filed under a later printing move to their first printing's set.
REMAP = {
    key: row["first_set"] + "." + key.split(".", 1)[1]
    for key, row in FIRST.items()
    if row["first_set"] != key.split(".")[0]
}

# ---------------------------------------------------------------- Rust types


def split_top(text: str, sep: str = ",") -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x.strip() for x in out if x.strip()]


def strip(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"#\[[^\]]*\]", "", text)
    return text


STRUCTS: dict[str, dict[str, str]] = {}
ENUMS: dict[str, dict[str, object]] = {}
for name in ["card", "ability", "mana", "predicate"]:
    src = strip((ROOT / f"src/state/{name}.rs").read_text())
    for m in re.finditer(r"pub struct (\w+) \{(.*?)\n\}", src, re.S):
        fields = {}
        for item in split_top(m.group(2)):
            fm = re.match(r"pub (\w+): (.+)$", item, re.S)
            if fm:
                fields[fm.group(1)] = fm.group(2).strip()
        STRUCTS[m.group(1)] = fields
    for m in re.finditer(r"pub enum (\w+) \{(.*?)\n\}", src, re.S):
        variants: dict[str, object] = {}
        for item in split_top(m.group(2)):
            item = re.sub(r"=\s*\d+$", "", item).strip()
            vm = re.match(r"(\w+)\s*\{(.*)\}$", item, re.S)
            if vm:
                fields = {}
                for f in split_top(vm.group(2)):
                    fm = re.match(r"(\w+): (.+)$", f, re.S)
                    fields[fm.group(1)] = fm.group(2).strip()
                variants[vm.group(1)] = fields
                continue
            vm = re.match(r"(\w+)\((.*)\)$", item, re.S)
            if vm:
                variants[vm.group(1)] = split_top(vm.group(2))
                continue
            variants[item] = None
        ENUMS[m.group(1)] = variants

# ---------------------------------------------------------------- Debug parser

TOKEN = re.compile(r'\s*("(?:[^"\\]|\\.)*"|-?\d+|[A-Za-z_]\w*|[{}()\[\],:])')


def tokenize(text: str) -> list[str]:
    pos, out = 0, []
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip() == "":
                break
            raise ValueError(text[pos : pos + 40])
        out.append(m.group(1))
        pos = m.end()
    return out


class Parser:
    def __init__(self, text: str):
        self.t = tokenize(text)
        self.i = 0

    def peek(self) -> str:
        return self.t[self.i]

    def take(self, expect: str | None = None) -> str:
        tok = self.t[self.i]
        if expect is not None and tok != expect:
            raise ValueError(f"expected {expect} got {tok} at {self.t[self.i-5:self.i+5]}")
        self.i += 1
        return tok

    def value(self):
        tok = self.take()
        if tok.startswith('"'):
            return ("str", tok)
        if re.fullmatch(r"-?\d+", tok):
            return ("num", tok)
        if tok == "[":
            return ("list", self.items("]"))
        if tok == "{":
            return ("set", self.items("}"))
        # identifier
        if self.i < len(self.t) and self.peek() == "{":
            self.take("{")
            fields = []
            while self.peek() != "}":
                name = self.take()
                self.take(":")
                fields.append((name, self.value()))
                if self.peek() == ",":
                    self.take(",")
            self.take("}")
            return ("struct", tok, fields)
        if self.i < len(self.t) and self.peek() == "(":
            self.take("(")
            return ("tuple", tok, self.items(")"))
        return ("ident", tok)

    def items(self, close: str):
        out = []
        while self.peek() != close:
            out.append(self.value())
            if self.peek() == ",":
                self.take(",")
        self.take(close)
        return out


def debug(v) -> str:
    kind = v[0]
    if kind in ("str", "num"):
        return v[1]
    if kind == "ident":
        return v[1]
    if kind == "list":
        return "[" + ", ".join(map(debug, v[1])) + "]"
    if kind == "set":
        return "{" + ", ".join(map(debug, v[1])) + "}"
    if kind == "tuple":
        return v[1] + "(" + ", ".join(map(debug, v[2])) + ")"
    return v[1] + " { " + ", ".join(f"{n}: {debug(x)}" for n, x in v[2]) + " }"


# ---------------------------------------------------------------- emitter

USED: set[str] = set()
COLOR_SYMBOLS = ["W", "U", "B", "R", "G", "C"]
COLOR_NAMES = {"White": "W", "Blue": "U", "Black": "B", "Red": "R", "Green": "G"}


def generic(ty: str) -> tuple[str, list[str]]:
    m = re.match(r"(\w+)<(.*)>$", ty)
    if m:
        return m.group(1), split_top(m.group(2))
    return ty, []


def mana_cost(v) -> str:
    fields = dict(v[2])
    assert debug(fields["spend_on_lesson"]) == "false"
    cost = [int(x[1]) for x in fields["cost"][1]]
    text = ""
    if cost[6]:
        text += "{" + str(cost[6]) + "}"
    for index, symbol in enumerate(COLOR_SYMBOLS):
        text += ("{" + symbol + "}") * cost[index]
    for pair in fields["hybrid"][1]:
        a, b = (COLOR_NAMES[c[1]] for c in pair[1])
        text += "{" + a + "/" + b + "}"
    USED.add("ManaCost")
    return f'ManaCost::parse("{text}")'


def mana(v) -> str:
    fields = dict(v[2])
    amounts = [int(x[1]) for x in fields["mana"][1]]
    lesson = [int(x[1]) for x in fields["lesson"][1]]
    USED.add("Mana")
    if any(lesson):
        return f"Mana {{ mana: {amounts}, lesson: {lesson} }}"
    text = ""
    for index, symbol in enumerate(["W", "U", "B", "R", "G"]):
        text += symbol * amounts[index]
    if amounts[5]:
        text = str(amounts[5]) + text
    return f'Mana::parse("{text}")'


DEFAULTISH = {"false", "[]", "None", "0", '""', "{}"}
DEFAULT_STRUCTS = {"CardPredicate", "ActivatedAbilityDefinition", "CardDefinition"}


def is_default(field_debug: str) -> bool:
    if field_debug in DEFAULTISH:
        return True
    if field_debug.startswith("Keywords {") and "true" not in field_debug:
        return True
    if field_debug == "ManaCost { spend_on_lesson: false, cost: [0, 0, 0, 0, 0, 0, 0], mana_value: 0, hybrid: [] }":
        return True
    if field_debug == "CardTypes { types: {} }":
        return True
    return False


def emit(v, ty: str) -> str:
    ty = ty.strip()
    head, args = generic(ty)
    if head == "Vec":
        assert v[0] == "list", (v, ty)
        return "vec![" + ", ".join(emit(x, args[0]) for x in v[1]) + "]"
    if head == "Option":
        if v == ("ident", "None"):
            return "None"
        assert v[0] == "tuple" and v[1] == "Some", v
        return "Some(" + emit(v[2][0], args[0]) + ")"
    if head == "Box":
        return "Box::new(" + emit(v, args[0]) + ")"
    if ty in ("Colors", "BTreeSet<Color>"):
        USED.update({"Colors", "Color"})
        return "Colors::from([" + ", ".join(f"Color::{x[1]}" for x in v[1]) + "])"
    if ty == "String":
        return f"{v[1]}.into()"
    if ty in ("i32", "u32", "u8", "usize", "bool"):
        return v[1]
    if ty.startswith("["):
        inner = ty[1:].split(";")[0].strip()
        return "[" + ", ".join(emit(x, inner) for x in v[1]) + "]"
    if ty == "ManaCost":
        return mana_cost(v)
    if ty == "Mana":
        return mana(v)
    if ty == "CardTypes":
        USED.update({"CardTypes", "CardType"})
        return "CardTypes::new([" + ", ".join(f"CardType::{x[1]}" for x in v[2][0][1][1]) + "])"
    if ty == "Keywords":
        USED.add("Keywords")
        on = [n for n, x in v[2] if debug(x) == "true"]
        if not on:
            return "Keywords::default()"
        return "Keywords { " + ", ".join(f"{n}: true" for n in on) + ", ..Default::default() }"
    if ty in STRUCTS:
        USED.add(ty)
        assert v[0] == "struct" and v[1] == ty, (v, ty)
        fields = STRUCTS[ty]
        parts = []
        for name, value in v[2]:
            if ty in DEFAULT_STRUCTS and is_default(debug(value)):
                continue
            parts.append(f"{name}: {emit(value, fields[name])}")
        if ty in DEFAULT_STRUCTS:
            if not parts:
                return f"{ty}::default()"
            return f"{ty} {{ " + ", ".join(parts) + ", ..Default::default() }"
        return f"{ty} {{ " + ", ".join(parts) + " }"
    if ty in ENUMS:
        USED.add(ty)
        variants = ENUMS[ty]
        name = v[1]
        shape = variants[name]
        if v[0] == "ident":
            assert shape is None, (ty, name)
            return f"{ty}::{name}"
        if v[0] == "tuple":
            return f"{ty}::{name}(" + ", ".join(emit(x, t) for x, t in zip(v[2], shape)) + ")"
        assert isinstance(shape, dict), (ty, name)
        return f"{ty}::{name} {{ " + ", ".join(
            f"{n}: {emit(x, shape[n])}" for n, x in v[2]
        ) + " }"
    raise ValueError(f"unhandled {ty}: {debug(v)[:80]}")


ORDER = [
    "name", "mana_cost", "types", "supertypes", "subtypes", "power", "toughness",
    "keywords", "is_token", "color_override",
]


def slug(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


def card_function(key: str, debug_text: str) -> tuple[str, str]:
    v = Parser(debug_text).value()
    fields = dict(v[2])
    name = json.loads(fields["name"][1])
    fn = key.split(".", 1)[1]
    body = []
    text_box = fields["text_box"][1]
    body.append(f'key: "{key}"')
    scryfall = IDS.get(key, "")
    body.append(f'scryfall_id: "{scryfall}"')
    rest = [(n, x) for n, x in v[2] if n not in ("text_box",)]
    rest.sort(key=lambda item: ORDER.index(item[0]) if item[0] in ORDER else len(ORDER))
    for n, x in rest:
        if is_default(debug(x)):
            continue
        body.append(f"{n}: {emit(x, STRUCTS['CardDefinition'][n])}")
    body.append("text_box: text_box.into()")
    text = (
        f"fn {fn}() -> CardDefinition {{\n"
        f"    let text_box = {text_box};\n\n"
        f"    CardDefinition {{ " + ", ".join(body) + ", ..Default::default() }\n}\n"
    )
    return fn, text


# ---------------------------------------------------------------- set files

by_set: dict[str, list[tuple[str, str]]] = {}
for row in DUMP:
    key = row["key"]
    if key.endswith(".native"):
        name = json.loads(Parser(row["debug"]).value()[2][[n for n, _ in Parser(row["debug"]).value()[2]].index("name")][1][1])
        key = key.split(".")[0] + "." + slug(name)
    scryfall = FIRST[key]["id"] if key in REMAP else (IDS.get(key) or FIRST.get(key, {}).get("id", ""))
    key = REMAP.get(key, key)
    IDS[key] = scryfall
    USED.add("CardDefinition")
    fn, text = card_function(key, row["debug"])
    file = key.split(".")[0]
    if file == "token":
        file = "tokens" if key in ("token.human_soldier", "token.treasure") else "tla"
    by_set.setdefault(file, []).append((fn, text))

MODULE_OF = {
    "CardDefinition": "card", "CardTypes": "card", "CardType": "card", "Keywords": "card",
    "ManaAbility": "card", "TriggeredManaAbility": "card", "PowerCda": "card",
    "StaticScope": "card", "StaticPtBuff": "card", "ReplacementEffect": "card",
    "SacrificeCost": "card", "ActivatedAbilityDefinition": "card", "SpellCostReduction": "card",
    "Ability": "ability", "TriggerCondition": "ability", "StaticCondition": "ability",
    "TriggerSubject": "ability", "EffectValue": "ability", "PaymentCost": "ability",
    "Effect": "ability", "TargetSpec": "ability", "TargetRequirement": "ability",
    "Mana": "mana", "ManaCost": "mana", "Color": "mana", "Colors": "mana",
    "CardPredicate": "predicate",
}

for set_code, functions in by_set.items():
    path = ROOT / f"src/cardsets/{set_code}.rs"
    old = path.read_text() if path.exists() else ""
    header = "".join(line + "\n" for line in old.splitlines() if line.startswith("//!"))
    functions.sort()
    body = "".join(text + "\n" for _, text in functions)
    used = sorted(t for t in MODULE_OF if re.search(rf"\b{t}\b", body))
    groups: dict[str, list[str]] = {}
    for t in used:
        groups.setdefault(MODULE_OF[t], []).append(t)
    imports = "use crate::state::{\n" + "".join(
        f"    {module}::{{{', '.join(names)}}},\n" for module, names in sorted(groups.items())
    ) + "};\n"
    listing = "".join(f"        {fn}(),\n" for fn, _ in functions)
    incomplete = ""
    if set_code == "tla":
        m = re.search(r"(/// Defined cards whose behavior is incomplete.*?\n\];\n)", old, re.S)
        incomplete = "\n" + m.group(1).replace("&[Incomplete]", "&[super::Incomplete]")
        for old, new in REMAP.items():
            incomplete = incomplete.replace(f'"{old}"', f'"{new}"')
    out = (
        header + imports + incomplete + "\n"
        "pub(super) fn cards() -> Vec<CardDefinition> {\n    vec![\n" + listing + "    ]\n}\n\n" + body
    )
    path.write_text(out)
    print(set_code, len(functions))

mod_path = ROOT / "src/cardsets/mod.rs"
mod = mod_path.read_text()
sets = sorted(by_set)
for stale in set(re.findall(r"^mod (\w+);$", mod, re.M)) - set(sets):
    (ROOT / f"src/cardsets/{stale}.rs").unlink()
    print("removed", stale)
mod = re.sub(r"(?:^mod \w+;\n)+", "".join(f"mod {x};\n" for x in sets), mod, count=1, flags=re.M)
mod = re.sub(r"\nconst SETS.*?\];\n", "\n", mod, flags=re.S)
mod += (
    f"\nconst SETS: [fn() -> Vec<CardDefinition>; {len(sets)}] = [\n"
    + "".join(f"    {x}::cards,\n" for x in sets)
    + "];\n"
)
mod_path.write_text(mod)
json.dump(REMAP, open(Path(sys.argv[2]).parent / "remap.json", "w"), indent=1)
