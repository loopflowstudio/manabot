# TLA direct-Rust decisions

Reversible operator choices made headless (2026-10-28 unless dated); each is
noted in the PR. Jack Heart's 2026-10-29 direction superseded the qualification
and pool-membership choices that were here (history: `be2b7586b:scratch/questions.md`).

- Old worlds run from their recorded git revision; no in-process rules shims.
- Superseded 2026-10-29 by Jack Heart's one-representation decision (w17, no
  JSON, tokens from `Effect`): the earlier choices to stay in w16, keep the IR's
  JSON shape and build tokens in `learning.py`.
- 2026-10-29 (sync onto T1 `f9732c12f`): T1 and T3 each filed card tests by set.
  This branch keeps T3's flat layout (`cards/support.rs` drivers, `rules/helpers.rs`
  `Scenario`, `rules/tla_mechanics.rs`, `cards/{fin,mkm,tokens}.rs`) because its
  tests carry T3's engine changes; T1's `tests/support/` families, `mod stage2 { … }`
  grouping, `soi.rs`, `zen.rs` and `rules/engine_events.rs` hold only tests already
  present here. Strixhaven-card tests moved to `cards/stx.rs` and coverage
  evidence identifiers were flattened to match. Revisit if T1's grouped layout
  should win when T1 lands.
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

- Support is by definition since the T1 sync (86ebe9ccf). Superseded after item 20:
  per-card tests already in `tests/rules/` move to `tests/cards/<set>.rs` here
  (Jack Heart's steer; items 21–22), since PR #394 has not moved them; on sync the
  first landed layout wins.
- Conditional entry reads the card owner's battlefield. Schema 8 is unreleased, so
  vocabulary is appended in place. `semantic_conformance_tests` is left to T1's
  world release.
- Zookeeper's creature-type count excludes a fixed noncreature-subtype list.
- Non-+1/+1 counters live in `Permanent.counters` (serialized only when present).
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
  X, labelled "X = n"). Offered X stops at the first unpayable value (no cap since the T2-sync compress; spell and activation X share `payable_x_values`). Fated
  Firepower's increase applies after prevention and doubling (the affected
  player's best order; no ordering choice is offered). Ability loss ignores
  timestamps (all keywords go, counters' included); CDAs still apply. Crashing
  Wave is deferred (spell waterbend and stun distribution).
- Slice 12: excess is measured from damage marked after replacements. Lesson
  lifelink applies to stack spells. The
  tutor offers every match plus failing to find. Hakoda's top card is shown only
  to its controller; CR 401.2's no-peeking-mid-cast limit is not modeled.
- Slice 13: granted abilities follow the grantee's own in its ability list,
  so their `ability_index` depends on what is attached; stale offers are
  rejected by the existing index checks. Activated-ability targets still
  exclude their source (pre-existing), so a Boomerang creature cannot target
  itself. Glider Staff waits for T2's airbend rather than a second one here.
- Slice 14 (Honest Work): no timestamps, so it also removes later grants and
  its "{T}: Add {C}" survives an until-EOT loss; until-EOT subtypes still add.
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
- Item 19 (with Fire Nation Palace): Name/creature types appear only while an effect sets them; spell keyword
  grants reach `Observation`, not the Academy stack view. `.pyi` now also lists `Permanent`'s
  existing `exile_if_dies_this_turn` and `temp_subtypes`.
- `manabot/sim/registration.py` admits worlds only up to w15 (w16 and w17 are
  rejected); left to the world release with the receipts.
- Item 22 moved all of `rules/tla_content.rs` into `cards/tla.rs` and merged its
  drivers into `cards/support.rs`. Multi-set tests went to the first card named
  (`landfall_*` → `fin.rs`). TLA's fixed tokens are TLA cards, so their tests
  stay in `tla.rs` (assumed reading of "tokens go to tokens.rs").
- `rules::structured_offers::structured_offer_rejects_fabricated_and_stale_ids_without_mutation`
  failed once at `structured_offers.rs:329` in a full debug `rules_tests` run,
  then passed alone and on a full rerun. It is untouched here and may compare a
  nondeterministic `Debug` rendering. Gate should watch for it.

## Item 23 (policy input v5) — 2026-10-29

- Effect-set name and creature types encode as one "characteristics set"
  flag. Their values follow from the setter (Honest Work names "Humble
  Merchant", a Citizen); the attachment join carries its definition to the
  host. No definition carries the name, so a definition reference is impossible
  without a name vocabulary in the native encoder.
- Temporary subtypes: flags for the five basic land types and Construct (every
  current grant besides v4's Avatar), then a count of any other, so none is
  dropped silently.
- `sideboard_counts` / `remaining_sideboard_counts`: the agent's remaining
  copies are already outside rows; the opponent's sideboard looks like hidden
  information that the `Observation` (and the LLM text's "Initial sideboard")
  exposes. Not encoded in the tensor. Question for Jack Heart: is a
  sideboard public in this format, or should the `Observation` hide it?
- Event `from_zone`/`to_zone`/incarnations stay out of the v0 event tensor;
  public history v1 (`policy_history_version` 1) encodes them.
- The tensor and text halves land before T1's sync. The learning-token guard
  and card conversion wait for T1's unpublished authoring-layer removal, as
  Jack Heart's steer directs ("sync onto it before reworking the encoding").
- Item 24 adds the state the `Observation` lacked (combat assignments, lost
  abilities, spent activation limits, kept red mana, no maximum hand size).
  Spent limits encode as flags for ability indices 0–3 plus a count of later
  indices (no current card has more). `deathtouch_damage` and `was_cast` stay
  unobserved: SBAs consume the first before any decision, and the second is
  read only while its own entry resolves.
- Scryfall ids (Jack Heart's steer) attach to card definitions, so they wait
  for the same T1 sync as the card conversion.
