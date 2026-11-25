---
extends: builtin
---
## In this repository

**Read the slice, not the history.** Read the plan's next slice and the files
it names. Finished slices are one line each and their detail is in git.

**Build native code once per pass.** Finish Rust edits before running the
slice's focused `cargo test` filter. Rebuild the Python extension only when a
named Python check reads changed native state, and only once. After a failure,
rerun what failed, not the whole list.

**Keep `scratch/plan.md` short.** Before finishing, collapse the completed
slice to one line with its commit in `scratch/plan.md` and commit your edits.

When deciding whether to continue:

- **Iterate** while `scratch/plan.md` still has unbuilt slices for the current
  phase.
- **Advance** when no unbuilt slices remain (or when the next slice is blocked
  on another Task's pending release—record the unblocking artifact and first
  command without waiting).
