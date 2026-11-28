---
extends: implement
---
## Card coverage

**One ~30–50 minute cohort slice per session.** Build the whole next mechanism
slice (one shared primitive and every card in the batch it unlocks, or 10–30
cards that already compose), or resolve the whole `card-review` fix and
simplification list in one pass. Stop after finishing that slice or fix pass
and return `iterate` if more slices remain—don't roll into a second planned
slice in the same session.

**Design orthogonal, CR-faithful primitives on the first pass.** Compose small
orthogonal pieces (selectors, conditions, triggers, effects, replacements) and
reshape cramped existing types when needed instead of adding multi-axis
special variants or derived flags where direct Comprehensive Rules state
belongs. Keep offers complete under `skip_trivial`, preserve replay/forks and
viewer privacy, append (never reorder) v11 observation columns, leave frozen
v9/v10 inputs untouched, and retire removed enum/record tokens in
`managym/src/cards/token_renames.rs` so `tests/model/test_pinned_pilot.py`
stays at zero drift.

**Update ledgers and defer branch-wide refreshes.** Update the set ledger
(`fdn.jsonl`, `fra.jsonl`, etc.) for the cards built in this slice; mark cards
waiting on another Task's authority pending rather than adding shortcuts. Run
the slice's focused filter (`cargo test --manifest-path managym/Cargo.toml -p
managym-cards --test cards <set>::`) and directed interaction tests. Defer
`scripts/refresh_academy_fixtures.py` and `scratch/tour.yaml` updates until
the final build slice or review-fix pass.
