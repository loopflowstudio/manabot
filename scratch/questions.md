# MTG-242 open questions and assumptions — 2026-11-12

Earlier assumptions (2026-11-04) are at
`e5b71f269e5ba4a7b4f75ba641b7ca59b95a470b:scratch/questions.md`.

## For review

1. **Policy input for restrictions outside the columns.** Main's player columns
   name Lesson spells, Shrine spells, Ally spells or Ally sources, and (v10)
   Powerstone mana. No current card makes other restricted mana. The port makes
   every encoder refuse it (`unrepresentable_state`, typed `Unrepresentable`
   error) instead of adding rows, because a full encoding needs a class join
   design. Proposed rows when wanted: a `FactKind::RestrictedMana` per
   (player, class, predicate), `id` the subtype's name token, `detail` the
   predicate form, values (class amount / 10, class ordinal), plus per-class
   color rows. Adding a kind widens `FactEncoder.kind`; checkpoints are
   already refused when the world digest moves, so this needs no world freeze,
   but the join is a model decision.
2. **Exclusions.** `ManaRestriction::ExceptNonartifactSpells` is the only
   exclusion, matching Powerstone and keeping `RestrictedManaData` and the
   published protocol schemas unchanged. A general "except" predicate set is
   unbuilt until a card needs it.
3. **Activation context.** `SpendingContext::Activate` keeps the source's zone
   and characteristics snapshot, not its object identity: no predicate reads
   identity yet.

## Assumptions taken

- Post-cost variable output stays a documented limit: main fixes dynamic
  outputs when offered, and no current card's output depends on its own costs.
- `ManaCost.purpose` serializes as the typed context (omitted when resolving).
  No retained corpus or digest reads priced purposes; definitions keep
  `Resolving` and serialize unchanged apart from restricted-mana outputs, whose
  learning tokens now name typed predicates and subtypes instead of bit masks.
