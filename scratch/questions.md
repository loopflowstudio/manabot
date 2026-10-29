# TLA direct-Rust decisions

Reversible operator choices made headless (2026-10-28 unless dated); each is
noted in the PR. Jack Heart's 2026-10-29 direction superseded the qualification
and pool-membership choices that were here (history: `be2b7586b:scratch/questions.md`).

- Old worlds run from their recorded git revision; no in-process rules shims.
- No w17: main is w15 and w16 is this PR's unreleased world, so renames, basics
  moving to `lea.*`, and support-by-definition all stay in w16.
- IR keeps the retired compiler's JSON shape, built in Rust on `serde_json::Value`
  (made byte-identity provable); lowering straight to `Step` is a later option.
- Learning tokens come from `learning.py` over that IR; schema 8 unchanged.
- 2026-10-29: the 14 setup-used incomplete TLA cards keep their definitions and
  are listed in one `tla::INCOMPLETE` const (key, missing behavior) beside
  `cards()`, rather than a field on all 269 `Card` literals. Native cards sit in
  set files as native `CardDefinition`s and are unsupported (no IR, so not in
  `pool_content`). Crossroads of Destiny moved to `cards/fixtures.rs` but stays
  in the default pack, because Rust integration tests and `manabot/verify` build
  games from it. D1 keeps the `tla-pool-v1` key; `set_cards` is the set filter.
  Undefined TLA cards are no longer listed as unsupported pool rows.
- Not rewritten (dated evidence): the Academy curated pack `local_identity` values
  keep pre-rename keys.
- Full TLA completion remains the eight-PR plan; no merge or new Task authorized.

Main's MTG-193 notes at the sync live at `6d069d6e5:scratch/questions.md`.

## T2 (87e012) — 2026-10-29 operator choices

- Printed flashback plus Iroh's grant: the grant's cost is used (dominant for
  every TLA card; a non-dominated alternative would need a cost choice).
- No schema 9 yet: schema 8 appended subtypes/fields/predicates per slice.
  Airbend appended opcodes 66–68 (and field `except`, predicate `attacking`)
  to schema 8 too: main is schema 7, so 8 is unreleased like w16, and
  `max_opcode[8]` moved 65 → 68 rather than opening schema 9.
  Landcycling reuses `search_basic_land` (any card of the subtype) and
  restricted mana reuses opcode 56 with `spend_only`; a distinct
  `search_library` or restricted `add_mana` opcode would be schema 9.
- Card abilities get their own action variant (`ActivateCardAbility`) sharing
  `PriorityActivateAbility`, rather than widening `ActivateAbility`.
- A discard-cost ability refers to its card's graveyard object after costs
  (Team Avatar's damage source), standing in for the hand object's LKI.
- World stays w16 while T2 only adds cards; bump when existing rules change.
- Observation `lesson_mana`, `ManaOutput.lesson` and the continuation `lesson`
  flag keep names and widths but carry all restricted mana. The flag is set
  when the payment's purpose names a Lesson/Shrine/Ally spell or ability
  source, so in w16 it now also fires for Ally spells and Ally abilities
  (before: Lesson spells only); acceptable while w16 is unreleased. The policy sees restricted totals, not which
  restriction; Academy reads "Restricted mana"; `manabot-text-4` still says
  "Lesson-only mana" (relabel needs a representation version bump).
- `PlayerVitals` bound 32 → 40 bytes (boxed restricted classes); `UndoEntry`
  ≤ 72. Non-tap mana abilities are admitted when `once_each_turn` bounds them.
- Avatar's Wrath's hand-only restriction ends at the start of its caster's
  next untap step and blocks flashback as well as exile grants. Mass airbend
  exiles creatures one at a time (no simultaneous-batch leave event).
- "Other target" (`another`) is enforced for triggered abilities; activated
  abilities already never offer their source. Spells need no source check.
