# TLA direct-Rust decisions

Reversible operator choices made headless (2026-10-28 unless dated); each is
noted in the PR. Jack Heart's 2026-10-29 direction superseded the qualification
and pool-membership choices that were here (history: `be2b7586b:scratch/questions.md`).

- Old worlds run from their recorded git revision; no in-process rules shims.
- Superseded 2026-10-29 by Jack Heart's one-representation decision (w17, no
  JSON, tokens from `Effect`): the earlier choices to stay in w16, keep the IR's
  JSON shape and build tokens in `learning.py`.
- 2026-10-29 (sync onto T1 `01e8f9975`): T1's test layout wins; v1–v4 stay
  28 wide here and `attacking` is a v5 column (plan, item 26).
- 2026-10-29: the 14 setup-used incomplete TLA cards keep their definitions and
  are listed in one `tla::INCOMPLETE` const (key, missing behavior) beside
  `cards()`, rather than a field on every definition. Since the one-representation
  rewrite every non-token definition outside `INCOMPLETE` is eligible. Crossroads of Destiny moved to `cards/fixtures.rs` but stays
  in the default pack, because Rust integration tests and `manabot/verify` build
  games from it. `set_cards` is the set filter (pool key since renamed `"world"`).
  Undefined TLA cards are no longer listed as unsupported pool rows.
- Not rewritten (dated evidence): the Academy curated pack `local_identity` values
  keep pre-rename keys.
- Full TLA completion remains the eight-PR plan; no merge or new Task authorized.
- 2026-10-29 (merge of T2 `d218e1dc2`): **assumption for Jack Heart to
  overrule at review.** T3 extends T2's unfrozen v5 and w18 in place rather
  than taking v6/w19; no checkpoint exists for either, so this is reversible.
  T2's rows are the base (v1–v4 keep 40 / 41 / 29, `attacking` stays v4
  column 27) and T3's columns follow T2's before validity: player 63, card 55,
  permanent 52. T3 keeps its relation columns (frame X, granted spell
  keywords; width 48), so the relation width alone tells v5 again.
- Same merge: T2's `granted_firebending` replaces T3's `firebending`
  observation field; T2's foretold column stands for `face_down` (T3's card
  column is gone). T2's exhausted / once-each-turn counts and T3's
  `used_abilities` indices both stay: the counts are T2's, the indices say
  which ability is spent. Dropping one is a review question.
- Same merge: T2's card conversion and tests win where both converted a T2
  card; T3-only cards and tests are added. T3's refreshing `can_activate`
  helper is `offers_ability`; T2's `can_activate[_card]` stand. T2's own notes
  are at `d218e1dc2:scratch/87e012-tla-t2-casting.md` and
  `d218e1dc2:scratch/questions.md`.

Main's notes at the syncs live at `6d069d6e5:scratch/questions.md` and `origin/main@633dca159:scratch/questions.md`.

- 2026-10-29 (one-representation rewrite): five cards were filed under a later
  printing and moved to their first printing's set (see plan). Three are TLA
  reprints (Corrupt Court Official, Epic Downfall, Lightning Strike), so
  `set_cards(["tla"])` no longer lists them; a TLA-legal filter would need
  printings, not first-printing key prefixes. Followed Jack Heart's
  first-printing rule; flag for f871a8 deck validation.
- 2026-10-29: `key` and `scryfall_id` are `#[serde(skip)]` — identity metadata,
  not rules — so content and state digests stay equal to w16's and digest-bound
  fixtures keep validating.
- 2026-10-29: learning token values are stable blake3 hashes of Rust/printed
  names (no vocabulary file); the policy builds its vocabulary from observed
  (kind, value) pairs as before.
- 2026-10-29: D1/f871a8 interface change: `ContentPool.compiled_semantics`
  (IR/source hashes) is gone; manifests carry `pack_key`; rows add
  `scryfall_id`; `IrError` is `ContentError`. `content_pool`/`pool_content`/
  `Eligibility` otherwise unchanged.
- 2026-10-29: the Python offline compiler (`manabot/semantic/compiler.py`) and its
  frozen-source tests stay as the reader/reproducer of frozen v1–w11 IR; nothing
  current reads it.
- 2026-10-29 (support reorg): tests that exercise several cards stay where
  they were (Corrupt Court Official with Ozai's Cruelty in `cards/tla.rs`, Epic
  Downfall with Power Word Kill in `cards/afr.rs`); Lightning Strike has no test
  of its own, so no `ptk.rs`/`ths.rs` test files exist yet.
- 2026-10-29 (concept review): world pool key becomes `"world"` (not per-world
  `"w17"`), describing the default pack; row `semantic_key` becomes `key`.
  Coordinate with f871a8 in one comment. The v1 coverage gate is left for a later
  Task rather than PR394.

- 2026-10-29 (assumption): superseded corpora now without consumers
  (`semantic-kernel-w15-v1`, `authored-match-w11-v1`, `authored-match-parity-w11-v1`,
  `public-commitment-parity-v1`) stay in the tree like w12–w14; archiving them
  to S3 is a separate publish step, not done here.
- 2026-10-29 (public-commitment parity, decision): bound to w17 by prefix. Only
  viewer 0's tracker fails (Command 68, South Pole Voyager revealed from library
  into hand); viewer 1 consumes the hero's Island discard (30) and
  `decline_learn` (102) over all 142 Commands. `TRACKED_COMMANDS = {0: 68,
  1: 142}`; identity-stream parity still covers the whole tape. Task 3be183
  (engine wave) owns the transport and removing the bound. No xfail.

## T3 (6875f0) assumptions — 2026-10-29

- Conditional entry reads the card owner's battlefield.
- Zookeeper's creature-type count excludes a fixed noncreature-subtype list.
- Non-+1/+1 counters live in `Permanent.counters` (serialized only when present).
- Heartless Act removes counters by a repeated resolution-time choice. Exhaust is
  per incarnation; 702.177b "haven't activated" effects are not modeled.
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

- `with_counter` keeps T2's +1/+1 meaning (Hog-Monkey's text); `no_counters`
  reads every kind. Card-only matching rejects `with_counter`/`attacking`.
- Library-top casting is a `zone_cast_cost` case; `in_casting_zone` is gone.
- An Aura entering attached invalidates both players' mana caches.
- Crashing Wave (item 18): the stun counters are placed one at a time among
  the tapped creatures an opponent controls at each placement, and all three
  must be placed while any is eligible ("any number" of recipients, CR 601.2d
  applied at resolution). The required waterbend {X} adds to the total cost,
  not the mana value; a free cast still pays it with X = 0.
- Item 19 (with Fire Nation Palace): Name/creature types appear only while an effect sets them; spell keyword
  grants reach `Observation`, not the Academy stack view. `.pyi` now also lists `Permanent`'s
  existing `exile_if_dies_this_turn` and `temp_subtypes`.
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
- Item 24 adds the state the `Observation` lacked (combat assignments, lost
  abilities, spent activation limits, kept red mana, no maximum hand size).
  Spent limits encode as flags for ability indices 0–3 plus a count of later
  indices (no current card has more). `deathtouch_damage` and `was_cast` stay
  unobserved: SBAs consume the first before any decision, and the second is
  read only while its own entry resolves.

## T2 notes at the second merge (head `89e320c48`) — 2026-10-29

T2's operator choices are at `89e320c48:scratch/questions.md`. New since
`d218e1dc2`: possible worlds v3 resample foretold cards (ordered slots,
`P(m, F)` fill factor); belief priors are not yet foretold-aware.
