---
extends: builtin
---
## In this repository

Never induce load to test anything ([TESTING.md](../../TESTING.md)); a large
constructed board run once is a workload. Keep runs, builds and `ETUDE_HOME`
under ignored `.runs/`. Finish the Rust edits before building the native
extension, then build it once per pass: a release build takes minutes.
