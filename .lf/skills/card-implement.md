---
requires: a slice-ordered card plan or `card-review` fix pass in `scratch/plan.md`
produces: committed code, tests and ledger updates for the next card slice or fix pass
action_style: procedural
---
Build the next card-cohort slice (or the full `card-review` fix pass) from `scratch/plan.md`.

1. **One ~30–50 minute slice per session.** Read the plan's next unbuilt slice
   and the files it names (finished slices are one line each; their detail is
   in git). Build the whole slice in this pass: the shared primitive and every
   card in the batch it unlocks, or the whole group of 10–30 cards that already
   compose. When addressing `card-review` findings, resolve the whole fix and
   simplification list in one pass rather than splitting it into small
   follow-ups. Stop after finishing that one slice or fix pass—don't roll into
   a second planned slice in the same session.
2. **Design orthogonal, CR-faithful types on the first pass.** Compose small
   orthogonal primitives (selectors, conditions, triggers, effects,
   replacements) and reshape cramped existing types when needed instead of
   adding multi-axis special variants or derived flags where direct
   Comprehensive Rules state belongs. Keep offers complete under
   `skip_trivial`, preserve replay/forks and viewer privacy, append (never
   reorder) v11 observation columns, leave frozen v9/v10 inputs untouched, and
   retire removed enum/record tokens in `managym/src/cards/token_renames.rs` so
   `tests/model/test_pinned_pilot.py` stays at zero drift.
3. **Update ledgers and delete replaced paths.** Update the set ledger
   (`fdn.jsonl`, `fra.jsonl`, etc.) for the cards built in this slice; mark
   cards waiting on another Task's authority pending rather than adding
   shortcuts. Delete obsolete symbols named in the slice together with their
   tests, keeping behavior on the surviving owner.
4. **Build once and defer branch-wide refreshes.** Finish the slice's Rust
   edits, then run its focused filter (`cargo test --manifest-path
   managym/Cargo.toml -p managym-cards --test cards <set>::`) and directed
   interaction tests. Rebuild the Python extension only when a named Python
   check reads changed native state, and only once. Defer
   `scripts/refresh_academy_fixtures.py` and `scratch/tour.yaml` updates until
   the final build slice or review-fix pass.
5. **Collapse the plan and commit.** Replace the completed slice in
   `scratch/plan.md` with one line naming its commit, note any discovery that
   changes a remaining slice (other ideas go in `scratch/questions.md`), and
   commit your edits cleanly before finishing.

When deciding whether to continue:

- **Iterate** while `scratch/plan.md` still has unbuilt slices for the current
  pass.
- **Advance** when no unbuilt slices remain (or when the next slice is blocked
  on another Task's pending release—record the unblocking artifact and first
  command without waiting).
