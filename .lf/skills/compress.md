---
extends: builtin
---
## In this repository

**Compress what changed since the last compress.** On a branch that already
has an `lf compress:` commit, take the diff since the latest one and follow it
outward; earlier passes already simplified the rest. Read the whole branch
only on its first compress. Start your own commit subjects with
`lf compress:` so the next pass finds where you stopped.

**Check with the slice's filter.** A cut to Rust runs the focused `cargo test`
filter the plan names. Rebuild the Python extension only when a Python check
is the one thing covering the cut.
