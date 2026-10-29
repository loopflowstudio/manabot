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

## T3 (6875f0) assumptions — 2026-10-29

- Support is by definition since the T1 sync (86ebe9ccf). Only this batch's tests
  went to `managym/tests/cards/`; moving T1's per-card tests is left to PR #394.
- Conditional entry reads the card owner's battlefield. Schema 8 is unreleased, so
  vocabulary is appended in place. `semantic_conformance_tests` is left to T1's
  world release.
- Zookeeper's creature-type count excludes a fixed noncreature-subtype list.
- Non-+1/+1 counters live in `Permanent.counters` (serialized only when present);
  observations don't show stun or fire counters, nor attachment (later shape change).
- Heartless Act removes counters by a repeated resolution-time choice. Exhaust is
  per incarnation; 702.177b "haven't activated" effects are not modeled.
- `set_base_pt` lowers only for the source until end of turn. Kyoshi Battle Fan's
  "create, then attach" is a `create_token` field.
- Auras act on the object attached at activation (captured before costs). An Aura
  put onto the battlefield uncast goes to the graveyard (303.4g not modeled).
- Swampsnare Trap castability assumes the reduction if any legal target earns it;
  targets that would leave the cost unpayable are not offered.
- Slice 7 (Vehicles): crew offers only other untapped creatures with positive
  power, one at a time, and stops once covered (tapping extra or 0-power creatures
  is legal but omitted). A completed crew then follows the ordinary {0} payment,
  which offers mana abilities when the player has any (as for Earth Kingdom
  Protectors). The observed `is_animated` now means "a creature
  only through an effect" (unchanged for existing content).
- Slice 8 (mana): state-dependent mana outputs are read when the ability is
  offered, not after its costs are paid; no current cost changes the reading.
  A single fixed output is still offered as a one-option `ChooseMode`, like other
  activated mana abilities. White Lotus Tile with X = 0 taps for nothing. Great
  Divide Guide's earlier tests stay in `tests/rules/temur_cards.rs` (T1's move).
- Slice 9 (effective characteristics): Toph's granted land type is read by
  predicates, "land you control" targets and land-counter CDAs, but not by the
  observation's `is_land` (printed types; a later shape change). Granted types
  are not retained in last known information. Summoning sickness is set at entry for
  creatures and Vehicles only (marking every permanent would change existing
  state hashes and observations), so a Clue The Mechanist animates the turn it
  was created can attack. Flopsie's
  limit refuses a second blocker during the one-at-a-time declaration rather
  than repairing it afterwards as menace does.
- Slice 10 (mana remainder): gained basic land types add one activated mana
  ability choosing among their colors, not one ability per type (CR 305.6);
  outputs are the same, ability indices differ. Energybending's affected set is
  fixed at resolution (CR 611.2c), so a land played later that turn gains
  nothing. Ozai's replacement is applied directly as pools empty, not through
  the proposed-event pipeline (no other mana-loss replacements exist); it
  applies while Ozai is on the battlefield as the step ends. Secret Tunnel is
  deferred (two-target activation).
- Slice 11 (X): the X choice reuses the Modal space (`ChooseMode`, mode =
  X, labelled "X = n"); observations don't show the announced X (later shape
  change). Offered X stops at the first unpayable value, capped at 20. Fated
  Firepower's increase applies after prevention and doubling (the affected
  player's best order; no ordering choice is offered). Ability loss ignores
  timestamps (all keywords go, counters' included); CDAs still apply. Crashing
  Wave is deferred (spell waterbend and stun distribution).
