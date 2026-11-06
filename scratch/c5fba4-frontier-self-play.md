# c5fba4 — frontier self-play on TLA trophy decks at two sizes

Status: protocol v1 drafted 2026-11-05 by the first implement step (headless,
no design review yet). Brief: Task c5fba4; direction comment 694f4190 in
[direction.md](direction.md). Contract: `experiments/protocols/tla-frontier-c5fba4-v1.{md,json}`;
runner `experiments/tla_frontier.py`; tests `tests/eval/test_tla_frontier.py`.

## Design (as implemented)

- Two regimes, identical to MTG-213 v10 except: input v9 / 512 facts, wall
  54 h, update cap 3,000. One regime file (w64) is hash-pinned; w256 is derived
  in `frozen()` from the protocol's `large` size (width 256, 4 layers, 8 heads;
  15.0× compute parameters of w64 = 648k → 9.71M, hashed tables excluded).
  Static declarations of those bytes live in tests, not `frozen()`.
- Damped learning rule kept unchanged (absolute iteration schedule, so the
  update cap does not alter learning rates).
- Wall-bounded runs → equal compute per size; shared update milestones give
  equal-update head-to-heads while both are running; endpoint head-to-head is
  the equal-compute comparison. Endpoint vs own initialization shows learning.
- 16 fixed pairs × 4 legs per comparison; per-pass eval cap USD2, cumulative
  USD12; training USD38 / 56 h per size; USD12 reserve under the USD100 cap.

## Delete — do not maintain

Nothing: new capability. `experiments/tla_pilot_v10.py` is executed evidence
and stays unchanged (its callback is mirrored, not shared, because its source
is hashed into retained environments).

## Remaining work

- Review of the protocol before freezing; re-freeze `runtime_world` if the card
  catalog moves before launch (`frozen()` rejects a drifted world).
- Freeze at a merged revision, then launch `runs(revision)` (paid; needs an
  execution decision) and drive `eval` passes during training.
- Analysis (bootstrap intervals over whole pairs) and the report are not
  written yet; `report()` gives descriptive curves only. Schedule slots carry
  tag, update and matchup; analysis derives endpoints from the runs.

Check: `uv run --extra dev pytest tests/eval/test_tla_frontier.py -q` — 8 passed.
