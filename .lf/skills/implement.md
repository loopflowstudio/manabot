---
extends: builtin
---
## In this repository

**Build whole ~30–50 minute slices.** When `scratch/plan.md` cuts cards into
slices, build the whole next slice in this pass: the shared primitive and every
card it unlocks, or the whole group of cards that already compose. When
addressing `card-review` findings, fix the whole review list (concepts,
composability, simplifications and failing checks) in one pass instead of
splitting it into tiny follow-ups. Don't stop after one card because it works.

**Read the slice, not the history.** Read the plan's next slice and the files
it names. Finished slices are one line each and their detail is in git.

**Compose orthogonal primitives on the first pass.** Reshape how effects,
triggers, replacements, conditions and selectors compose rather than adding a
specialized type or variant that fuses multiple axes or emits derived flags
where direct Comprehensive Rules state belongs.

**Build once and defer branch-wide refreshes.** Finish the slice's Rust edits,
then run its focused filter (`cargo test --manifest-path managym/Cargo.toml -p
managym-cards --test cards <set>::`). Rebuild the Python extension only when a
named Python check reads the changed native state, and only once. Defer
`scripts/refresh_academy_fixtures.py` and `scratch/tour.yaml` updates until the
final slice or review-fix pass—don't refresh fixtures or tours after every
intermediate slice.

**Keep `scratch/plan.md` short.** Before finishing, collapse the completed
slice to one line with its commit in `scratch/plan.md` and commit your edits.

When deciding whether to continue:

- **Iterate** while `scratch/plan.md` still has unbuilt slices for the current
  phase.
- **Advance** when no unbuilt slices remain (or when the next slice is blocked
  on another Task's pending release—record the unblocking artifact and first
  command without waiting).
