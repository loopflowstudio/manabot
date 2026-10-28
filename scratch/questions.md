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
