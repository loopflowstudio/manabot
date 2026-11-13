# MTG-320 demo decision — 2026-11-13

The demo did not happen: this run is headless and nobody is available to use
the product. No human reactions or approval were collected. No demo processes
were started.

Decision: advance. This is the caller-authorized stand-in decision, not a
reaction attributed to Jack Heart. The preceding pass intended checked card
identities with unchanged gameplay. Its recorded QA at 6b28b642d6 reports 568
seeded games and 137,518 decisions matching main's observation and state hashes,
and three Academy games reaching Game Over without page or console errors.
The tap-cost failure also occurred on main. The engine and Academy trees have
not changed since that QA. Subsequent gate receipts at 349d643e52 report 22
focused debug tests passing, Clippy and formatting passing, current-head CI
passing, and Engine Budget passing with unchanged digests.

These results support the scoped outcome; no concrete repairable gap justifies
the backward edge to implementation. Human feel remains unobserved. Preserve
the recorded direction: "Stop at review. Do not enable auto-merge."
