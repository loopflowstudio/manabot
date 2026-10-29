# TLA direct-Rust decisions

Reversible operator choices made headless (2026-10-28 unless dated); each is
noted in the PR. Jack Heart's 2026-10-29 direction superseded the qualification
and pool-membership choices that were here (history: `be2b7586b:scratch/questions.md`).

- Old worlds run from their recorded git revision; no in-process rules shims.
- T1: no w17 (main is w15; renames, basics moving to `lea.*` and support by
  definition stay in w16). T3 later takes w17.
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
  change). Offered X stops at the first unpayable value (no cap since the T2-sync compress; spell and activation X share `payable_x_values`). Fated
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

## T3 sync onto T2 (PR #398 head `16dfd3dbb`) — 2026-10-29

T2's design and operator choices are at `16dfd3dbb:scratch/87e012-tla-t2-casting.md`
and `16dfd3dbb:scratch/questions.md`; the ones this branch now builds on:
`CastChoices` X, `ActivationSource`, `ActivatedAbilityDefinition.effects`,
`Permanent::{activation_used, record_activation}`, `ManaUse`-keyed restricted
mana, `Game::zone_cast_cost`/`may_play_from_zone`, `ZoneEntry`.

- Schema 8: T2's numbers stay; this branch's entries follow T2's maxima in
  their old order. Same-named entries are one entry (`x`, `mana_x`, `exhaust`,
  `each:creature_you_control`). `earthbend_value` is gone (T2's `earthbend`
  takes a `Count`), so opcodes after it moved down one.
- `with_counter` keeps T2's +1/+1 meaning (Hog-Monkey's text); `no_counters`
  reads every kind. Card-only matching rejects `with_counter`/`attacking`.
- Activation X rides `Payment::Activation.x`; `ActivationX` and `Crew` keep a
  `PermanentId` (battlefield only). `ActivatedAbilityDefinition::target_requirement`
  replaced `requires_target`/`target_is_optional`.
- Library-top casting is a `zone_cast_cost` case; `in_casting_zone` is gone.
- An Aura entering attached invalidates both players' mana caches.
- Card tests: T2's funded `setup` stays; this branch's tests use
  `setup_unfunded`. T2's one-argument `can_activate` is `can_activate_any`.
- World: w17 (own `WORLDS.md` entry); conformance receipts stay w15.
- Crashing Wave (item 18): the stun counters are placed one at a time among
  the tapped creatures an opponent controls at each placement, and all three
  must be placed while any is eligible ("any number" of recipients, CR 601.2d
  applied at resolution). The required waterbend {X} adds to the total cost,
  not the mana value; a free cast still pays it with X = 0. Kicker and a
  required additional cost on one spell are rejected at compile time.
- Item 19 (observation change, with Fire Nation Palace): the new fields are
  observation, protocol-view and Academy-type data only, like T2's `face_down`;
  no policy tensor column and no new policy-input version, so frozen widths and
  checkpoints are untouched. Encoding them for policies is a later choice.
  Name/creature types appear only while an effect sets them; spell keyword
  grants reach `Observation`, not the Academy stack view. Academy UI rendering
  of the new view fields is not added. `.pyi` now also lists `Permanent`'s
  existing `exile_if_dies_this_turn` and `temp_subtypes`.
- `manabot/sim/registration.py` admits worlds only up to w15 (w16 and w17 are
  rejected); left to the world release with the receipts.
