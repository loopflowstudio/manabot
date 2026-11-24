---
extends: builtin
---
## In this repository

**A card slice is a cohort.** When the plan cuts cards into slices, build the
whole next slice in this pass: the shared primitive and every card it unlocks,
or the whole group of cards that already compose. Don't stop after one card
because it works; every pass cold-starts a fresh session, syncs and realigns.

**Read the slice, not the history.** Read the plan's next slice and the files
it names. Finished slices are one line each and their detail is in git.

**Build native code once per pass.** Finish the slice's Rust edits, then run
its focused filter (`cargo test --manifest-path managym/Cargo.toml -p
managym-cards --test cards <set>::`). Rebuild the extension, as AGENTS.md
describes, only when a named Python check reads the changed native state, and
only once. After a failure, rerun what failed, not the whole list.
