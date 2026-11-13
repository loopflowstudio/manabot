# Decisions and assumptions

- Accepted scope and exact steers are in `scratch/mtg-320-plan.md`.
  The physical-card store and new zone identity remain excluded.
- Reversible implementation choice (2026-11-12): library/hand use one ordered
  `CardZoneEntry` sequence, with `Card(CardId)` and explicit
  `Transient(ObjectId)` occupants. This stores checked cards and preserves
  visitors without a second ordering authority. It replaces the unaccepted
  all-ObjectId fallback; compatibility is checked at the wire and RNG boundaries.
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
- Compression choice (2026-11-12): keep `ZoneObjects` as a borrowed collection
  with explicit `iter()` and `cards()` scans. No separate cursor or
  card-zone-only reader; stored entry kinds, bytes and order remain authoritative.
