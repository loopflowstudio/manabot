---
extends: lf:kickoff
---
Before proposing a compatibility shim for a released policy input, check what
the engine emitted and what the policy admitted in that release and on main.
If main has stopped emitting a value the released policy already rejected,
plan a regression test at the emission point. Widen admission only for a
supported input that needs it.
