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
- Slice 7 (Vehicles): crew offers other untapped positive-power creatures one at
  a time until covered (extra or 0-power taps omitted), then the ordinary {0}
  payment. Observed `is_animated` means "a creature only through an effect".
- Slice 8 (mana): state-dependent outputs are read when offered (no current
  cost changes them); one fixed output is still a one-option `ChooseMode`.
  White Lotus Tile with X = 0 taps for nothing.
- Slice 9: Toph's granted land type is read by predicates, targets and CDAs,
  not by observed `is_land` or LKI. Only creatures and Vehicles enter summoning
  sick (hash stability), so a Clue The Mechanist animates can attack at once.
  Flopsie's limit refuses a second blocker during declaration.
- Slice 10: gained basic land types add one mana ability choosing among their
  colors (same outputs as CR 305.6, different indices). Energybending's set is
  fixed at resolution. Ozai's replacement applies directly as pools empty while
  Ozai is on the battlefield, outside the proposed-event pipeline.
- Slice 11 (X): the X choice reuses the Modal space (`ChooseMode`, mode =
  X, labelled "X = n"); observations don't show the announced X (later shape
  change). Offered X stops at the first unpayable value, capped at 20. Fated
  Firepower's increase applies after prevention and doubling (the affected
  player's best order; no ordering choice is offered). Ability loss ignores
  timestamps (all keywords go, counters' included); CDAs still apply. Crashing
  Wave is deferred (spell waterbend and stun distribution).
- Slice 12: excess is measured from damage marked after replacements. Lesson
  lifelink applies to stack spells; observations don't show spell grants. The
  tutor offers every match plus failing to find. Hakoda's top card is shown only
  to its controller; CR 401.2's no-peeking-mid-cast limit is not modeled.
- Slice 13: granted abilities follow the grantee's own in its ability list,
  so their `ability_index` depends on what is attached; stale offers are
  rejected by the existing index checks. Activated-ability targets still
  exclude their source (pre-existing), so a Boomerang creature cannot target
  itself. Glider Staff waits for T2's airbend rather than a second one here.
- Slice 14 (Honest Work): no timestamps, so it also removes later grants and
  its "{T}: Add {C}" survives an until-EOT loss; until-EOT subtypes still add.
  Observations don't show the new name or types.
- Tap authority: a static haste grant changes mana output only through moves
  that already invalidate the mana cache; a grant whose condition flips without
  a move leaves the cache stale until the next invalidation, as other
  condition-dependent statics already do.
- Secret Tunnel (item 16): a departed target's LKI is found by the battlefield
  slot it occupied (`ObjectLki.permanent`) instead of capturing target
  `ObjectRef`s on the stack, as the concept review suggested; slots are never
  reused, so both name the same object and the stack shape stays unchanged.
  With more than two targets, "share a creature type" means one type common to
  all, and the first-pick partner check is pairwise (exact for two).
