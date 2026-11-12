"""Apply exact (old, new) replacements from a JSON list; fail on any miss.

uv run python scratch/bench/edit.py <file> <<< '[["old","new",count?], ...]'
Adds `import_line` after the first matching anchor if given as ["@import", line, anchor].
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
s = path.read_text()
for item in json.load(sys.stdin):
    if item[0] == "@import":
        line, anchor = item[1], item[2]
        if line not in s:
            assert anchor in s, anchor
            s = s.replace(anchor, anchor + line, 1)
        continue
    old, new = item[0], item[1]
    count = item[2] if len(item) > 2 else 1
    found = s.count(old)
    assert found == count, f"{old!r}: found {found}, expected {count}"
    s = s.replace(old, new)
path.write_text(s)
