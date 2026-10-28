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

- Training extension (2026-10-28): use uniform-distinct-v1, drawing two different
  normalized lists with unbiased ordered seats. Require at least two distinct
  lists; mirrors and adaptive weights are deferred. Manifest paths resolve at
  recipe load into an embedded immutable collection, so recovery never depends
  on a mutable external file. Source records remain private retained evidence.
- Coordination: PR383 d0f48ca96 retains the existing uniform roster sampler and
  adds adaptive root reassignment separately. Extend that uniform authority;
  preserve its Elemental sequence and leave reassignment/campaign code alone.
  PR389 c85470a59 requires no simulation API change for this extension.
