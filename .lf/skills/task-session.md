---
extends: builtin
---
## In this repository

**Let `deliver` and CI run the test matrix.** When asked to ship or deliver
after an edit, commit, push and launch `deliver` instead of running local
clippy or full Rust/Python test matrices first; `gate` and CI catch issues on
real infrastructure.
