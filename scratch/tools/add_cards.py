"""Insert card functions into a set file in name order, and list them in cards().

Usage: python3 scratch/tools/add_cards.py managym/src/cardsets/tla.rs new_cards.rs
Each top-level `fn name() -> CardDefinition` in new_cards.rs is placed before
the first existing card function whose name sorts after it. Helper functions
(other return types) go before the "// ---- Mana constructors ----" line.
"""
import re
import sys

target, source = sys.argv[1], sys.argv[2]
text = open(target).read()
new = open(source).read()

blocks = re.split(r"\n(?=fn |/// )", "\n" + new.strip() + "\n")
items = []
current = ""
for block in blocks:
    if not block.strip():
        continue
    current += block.strip("\n") + "\n"
    if block.lstrip().startswith("fn "):
        items.append(current)
        current = ""
assert not current.strip(), current

for item in items:
    name = re.search(r"fn (\w+)\(", item).group(1)
    is_card = "-> CardDefinition {" in item.split("\n", 2)[0] + item.split("\n", 2)[1] and re.search(r"fn \w+\(\) -> CardDefinition", item)
    if is_card:
        fns = [(m.start(), m.group(1)) for m in re.finditer(r"^fn (\w+)\(\) -> CardDefinition", text, re.M)]
        after = [pos for pos, fn in fns if fn > name]
        pos = after[0] if after else text.index("// ---- Mana constructors ----")
        # include preceding doc comments of that function
        while True:
            prev = text.rfind("\n", 0, pos - 1)
            line = text[prev + 1 : pos]
            if line.startswith("///"):
                pos = prev + 1
            else:
                break
        text = text[:pos] + item + "\n" + text[pos:]
        start = text.index("pub(super) fn cards() -> Vec<CardDefinition> {")
        end = text.index("    ]\n}", start)
        entries = re.findall(r"^        (\w+)\(\),$", text[start:end], re.M)
        assert name not in entries, name
        entries.append(name)
        entries.sort()
        body = "pub(super) fn cards() -> Vec<CardDefinition> {\n    vec![\n" + "".join(f"        {e}(),\n" for e in entries)
        text = text[:start] + body + text[end:]
    else:
        pos = text.index("// ---- Mana constructors ----")
        text = text[:pos] + item + "\n" + text[pos:]
open(target, "w").write(text)
print("added", [re.search(r"fn (\w+)\(", i).group(1) for i in items])
