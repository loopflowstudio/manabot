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

When the next slice is blocked on another Task's pending release, record the
unblocking artifact and first command in `scratch/plan.md` and return `advance`
without waiting.
