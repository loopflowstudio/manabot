# Decklist assumptions — 2026-10-28

- Development construction policy reflects existing engine behavior: nonempty
  main deck, positive quantities, eligible non-token definitions in either
  section. No tournament size, copy limit, color or sideboard limit is claimed.
  Existing authored presets remain available. Sideboards are fixed outside-game
  copies; importing one does not provide sideboarding between games.
- Initial pool eligibility is content-owned (892c47); no generic native fallback.
- Inherited MTG-207 notes remain at c5d406e3d:scratch/questions.md.

- Operational import ceiling: 64 KiB source and 1,000 cards per player across
  both sections. This is resource admission, explicitly not a tournament rule.

- Initial implementation (superseded as a mandatory lifecycle by the latest
  concept review): use uniform-distinct-v1, drawing two different
  normalized lists with unbiased ordered seats. Require at least two distinct
  lists; mirrors and adaptive weights are deferred. Manifest paths resolve at
  recipe load into an embedded immutable collection, so recovery never depends
  on a mutable external file. Source records remain private retained evidence.
- Accepted correction: one deck concept; training submits decks to its verified
  world's native validation. Fixed lists can be checked upfront. Computed or
  changing decks must not require a different deck type or registration in a
  frozen roster. Selection/generation and its recovery state belong to training;
  checkpoints know the world, not training. Curriculum/generation machinery is
  future work, not approved implementation scope in this concept pass.
- Coordination: PR383 d0f48ca96 retains the existing uniform roster sampler and
  adds adaptive root reassignment separately. Extend that uniform authority;
  preserve its Elemental sequence and leave reassignment/campaign code alone.
  PR389 c85470a59 requires no simulation API change for this extension.

- 2026-10-28 implementation: imported-setup consumers (sim, Academy) still
  reject checkpoint opponents; pool-bound checkpoints would now be admissible
  there, but enabling it is consumer scope beyond this training correction.
- Checkpoint `bc` extra still carries run_id/stage_id/regime_digest producer
  links that predate this branch; producer admission depends on them. Moving
  them fully to run records is a separate change; no training-deck data remains.
- Sampling policy is the code-named `uniform-distinct-v1`, not a stored field.
