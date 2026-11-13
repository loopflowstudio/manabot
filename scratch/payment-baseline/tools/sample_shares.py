"""Inclusive sample shares from a macOS `sample` call tree.

Counts the main thread's samples under the top-most frame of each group, so
recursion is counted once. Groups: `mana` is any symbol of managym::flow::mana;
within it `search` is PaymentModel/PaymentSearch code (plan, witness, bounds)
and `model` is the rest (payment_model and what it calls).
"""
import re
import sys
from pathlib import Path

LINE = re.compile(r"^([ +!:|]*?)(\d+) (\S+)")


def frames(path: Path) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    in_tree = False
    for line in path.read_text().splitlines():
        if line.startswith("Call graph:"):
            in_tree = True
            continue
        if line.startswith("Total number in stack"):
            break
        if not in_tree:
            continue
        match = LINE.match(line)
        if match:
            rows.append((len(match.group(1)), int(match.group(2)), match.group(3)))
    return rows


def main() -> None:
    rows = frames(Path(sys.argv[1]))
    total = rows[0][1]
    is_mana = lambda s: "4flow4mana" in s
    is_search = lambda s: is_mana(s) and ("12PaymentModel" in s or "13PaymentSearch" in s or "18PaymentSearchState" in s)
    sums = {"mana": 0, "search": 0}
    callers: dict[str, int] = {}
    stack: list[tuple[int, str]] = []
    for depth, count, symbol in rows:
        while stack and stack[-1][0] >= depth:
            stack.pop()
        above = [s for _, s in stack]
        if is_mana(symbol) and not any(is_mana(s) for s in above):
            sums["mana"] += count
            name = re.sub(r"^_RNv\w*?4flow\d+", "", above[-1])[:60] if above else "?"
            callers[above[-1][-70:]] = callers.get(above[-1][-70:], 0) + count
        if is_search(symbol) and not any(is_search(s) for s in above):
            sums["search"] += count
        stack.append((depth, symbol))
    model = sums["mana"] - sums["search"]
    print(f"samples {total}")
    print(f"payment (flow::mana inclusive) {sums['mana']} = {sums['mana']/total:.1%}")
    print(f"  plan/witness search          {sums['search']} = {sums['search']/total:.1%}")
    print(f"  model construction and rest  {model} = {model/total:.1%}")
    print(f"everything else                {total-sums['mana']} = {(total-sums['mana'])/total:.1%}")
    for name, count in sorted(callers.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  entered from ...{name}: {count}")


main()
