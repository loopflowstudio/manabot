# Authored match authority receipt v1

`release-stack-ur-vs-gw-seed-0.json` is the checked authority-private receipt
for one deterministic UR Lessons versus GW Allies match. It drives both actors
from Etude server-built offers through revision-bound commands and records the
compiled semantic manifest, every deliberate decision, ordered semantic and
presentation event groups, state witnesses, encountered admitted programs, and
explicit fallback counters.

Regenerate or verify it from the repository root:

```bash
uv run --extra dev python scripts/generate_authored_match_receipt.py
uv run --extra dev python scripts/generate_authored_match_receipt.py --check
uv run --extra dev pytest -q tests/etude/test_authored_match_authority.py::test_release_stack_authored_match_receipt_is_terminal_and_authoritative
```

This receipt proves one terminal production-authority match. It does not prove
cross-surface replay parity, broader card conformance, UI behavior, or workload
performance budgets.
