---
extends: builtin
---

**Let `deliver` and CI run the test matrix.** When someone asks to ship or
deliver after an edit, commit, push, and launch `deliver` instead of running
local clippy or full Rust/Python test matrices first; `gate` and CI will catch
issues on real infrastructure.
