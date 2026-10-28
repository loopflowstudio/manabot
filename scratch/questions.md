# bb4160 unresolved decisions — 2026-10-28

The current accepted outcome, implementation and remaining work are in
`scratch/bb4160-evaluate-qwen-versus.md`.

- **World choice ACCEPTED (Jack Heart, 2026-10-28):** run the exact MTG-183
  checkpoint in a separately pinned **historical w13** experiment (w13
  source/native build, w13 decks, policy-input v1). Label all results as w13
  evidence. Do not weaken w14 admission or relabel the checkpoint. Paid
  evaluation within the $10 budget is unblocked by this decision.
- **Independent launch prerequisite:** Etude SDK 0.2.10 rejected the launcher
  database version. No shared database or dependency pin was changed.
- **Memory gap:** `.lf/wave/training/` and its child memories are absent. The
  supplied chapter is the available objective; no other Wave is substituted.
- **Service identity implementation choice, 2026-10-28:** derive the sole vLLM
  served name from the existing behavior hash. Its base argv excludes that
  derived argument to avoid a circular hash; the complete command is retained
  separately. Model-list acknowledgement and completion model checks bind the
  supervised service declaration, not independently measured GPU weight bytes.
  This world-independent work does not settle the matchup decision.

Original checkpoint evidence and the full directive are preserved in
`02fccd93e1983319d355a1866c101d7d0e01565b:scratch/bb4160-evaluate-qwen-versus.md`
and `02fccd93e1983319d355a1866c101d7d0e01565b:scratch/questions.md`.
Obsolete MTG-207 notes remain at
`c5d406e3d2a701525c15c8500a30f7da7896cc40:scratch/questions.md`.
