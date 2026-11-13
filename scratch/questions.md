# Decisions and assumptions

- Accepted scope, exact steers and the ordered-zone fallback are in
  `scratch/mtg-320-plan.md`. This step simplifies that implementation; it adds
  no physical-card store, new zone identity or new object kind.
- Spell copies are internally `SpellCopy`; observation `is_token` remains true.
  Predicate token filters mean `ObjectKind::Token`. No defined card filters
  spells by token status (earlier MTG-320 inspection).
- Preserve the tucked-token draw behavior and hidden-world RNG consumption.
  Generic selection transport remains object-based; exact `ObjectRef`s reject
  stale selections. Card-only consumers narrow once.
- Sync decision of 2026-11-12: main's scratch deletion was treated as applying
  to inherited MTG-298 QA; the MTG-320 sections and screenshots were retained.
- Inherited definition-design notes are historical MTG-298 context, not changes
  to frozen evidence in this task. Full notes: `df58f2bff:scratch/questions.md`.
  Pre-compression uncommitted notes were moved to
  `/tmp/mtg320-context-before-compress/` before curation.
