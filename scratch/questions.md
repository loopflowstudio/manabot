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
- Compression choice (2026-11-12): keep `ZoneObjects` as a borrowed collection
  with explicit `iter()` and `cards()` scans. No separate cursor or
  card-zone-only reader; stored entry kinds, bytes and order remain authoritative.
- Concept review (2026-11-12, headless): graveyard/exile stay `Vec<ObjectId>`
  per steer f81d37a0's zone list, although the Task brief's quotes name
  graveyards. Assumed the steer governs; flagged in the PR body for the reviewer.

## Historical source facts from main

# Assumptions and source facts

2026-11-12: Use `token.plain_spirit` for Release to Memory's unrestricted
Spirit (concept review replaced `token.colorless_spirit`: TLA's Spirit is
colorless too), keeping TLA's `token.spirit` unchanged. Use `token.zeppelin` for Lita's
unprinted Vehicle. Both are reversible internal naming choices.

Scryfall API verified the first-printing card ids from the pinned inventory:
- https://api.scryfall.com/cards/b45c4bbf-2c61-4d49-96c8-34a0fb0d6d02
- https://api.scryfall.com/cards/f3a5a044-d474-43f4-95a9-e67808908e6c
The earliest 1/1 colorless Spirit token from the ordered printings search is
P04 (2004-01-01), id 46b60d95-b9bc-40f8-b986-bfa8e3eb74f3, empty Oracle text.
Lita's printed rules say Zeppelin is a 5/5 colorless artifact Vehicle with
flying and crew 3; it becomes a creature only when animated.

2026-11-12: The concept-review plan called existing flicker text "Return it
to the battlefield…". Inspection of main's `move_description` shows "Put it
onto the battlefield under its owner's control" for `Moved`. Preserve that
actual wording in the regression assertion, consistent with the instruction
not to reword existing cards; do not change the move verb.

2026-11-12: Use the descriptive `w17-token-cards` fixture suffix for the new
Academy catalog binding. Preserve the existing `w17` files unchanged; replay
only existing Commands and require their offer ids/labels to match.

2026-11-13: Main now includes MTG-297 (Heartbeat of Spring). Use the complete
61/61 bonus sheet and 1,000-deck admission as the combined acceptance outcome;
do not keep a known obsolete "only Heartbeat missing" expectation. Seed new
`w17-token-cards` fixtures from main's current w17 Commands before refreshing
the combined catalog, because main's mana spending redesign changed Commands.
The managed `lf sync` plan selected merge_target for this branch; preserve
history through that supported integration rather than raw git rebase.

## MTG-320 sync resolution (2026-11-13)

Preserve MTG-320's QA page at `scratch/qa.html`; the incoming MTG-313 page
is recoverable from target `22ed4a733484d27db2fd51274d1b4e23245a6741`.
Keep incoming source facts above as historical context. Combine the runtime
object rename with main's explicit mana-spending fixtures, receipts and
triggered production. Foretell completion retains checked `CardId`.

## Scratch deletion during sync (2026-11-13)

Pinned main `1d7b5fde9ef411b6d52cb286c74f94ed776b231f` deletes
`scratch/qa.html` and `scratch/questions.md` as part of its completed backend
work. Retain this branch's versions: they hold MTG-320's current QA evidence
and scoped decisions. This reversible resolution follows the earlier scratch
preservation choice; incoming backend code remains as merged.
