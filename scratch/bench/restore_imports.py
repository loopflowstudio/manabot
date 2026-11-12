"""Restore HEAD import order, keeping only imports this change added."""
import subprocess
import sys
from pathlib import Path


def statements(lines: list[str]) -> tuple[list[str], int, int]:
    """Top-level import statements (joined text) and the header span."""
    out, i, start, end = [], 0, None, 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(("import ", "from ")):
            if start is None:
                start = i
            stmt = [line]
            if "(" in line and ")" not in line:
                while ")" not in lines[i]:
                    i += 1
                    stmt.append(lines[i])
            out.append("".join(stmt))
            end = i + 1
        elif start is not None and line.strip() and not line.startswith("#"):
            if line.startswith("if TYPE_CHECKING"):
                break
            break
        i += 1
    return out, start, end


def module(stmt: str) -> str:
    return stmt.split()[1]


for name in sys.argv[1:]:
    head = subprocess.run(["git", "show", f"HEAD:{name}"], capture_output=True, text=True, check=True).stdout
    current = Path(name).read_text()
    h_lines = head.splitlines(keepends=True)
    c_lines = current.splitlines(keepends=True)
    h_stmts, h_start, h_end = statements(h_lines)
    c_stmts, c_start, c_end = statements(c_lines)
    added = [s for s in c_stmts if s not in h_stmts]
    removed = [s for s in h_stmts if s not in c_stmts]
    assert not removed, (name, removed)
    header = h_lines[h_start:h_end]
    for stmt in added:
        mod = module(stmt)
        assert mod.startswith("manabot"), (name, stmt)
        # insert before first first-party statement sorting after it
        text = "".join(header)
        pos = None
        idx = 0
        joined = []
        j = 0
        while j < len(header):
            line = header[j]
            if line.startswith(("import ", "from ")):
                k = j
                if "(" in line and ")" not in line:
                    while ")" not in header[k]:
                        k += 1
                s = "".join(header[j : k + 1])
                m = module(s)
                if (m.startswith("manabot") or m.startswith("academy") or m.startswith("experiments")) and m.lower() > mod.lower() and pos is None:
                    pos = j
                j = k + 1
            else:
                j += 1
        if pos is None:
            # after the last manabot statement
            last = max(i for i, l in enumerate(header) if l.startswith(("from manabot", "import manabot")))
            k = last
            if "(" in header[k] and ")" not in header[k]:
                while ")" not in header[k]:
                    k += 1
            pos = k + 1
        header[pos:pos] = [stmt]
    Path(name).write_text("".join(c_lines[:c_start] + header + c_lines[c_end:]))
    print(name, "restored", len(added), "added")
