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
- Settled (Jack Heart, 2026-10-30, "1a, 2a"): T3 extends the open w17 and
  policy input v5 in place; never w19/v6. Widths: player 63, card 55,
  permanent 37 plus the facts block, relation 48 (v1–v4 keep 40 / 41 / 29).
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
- Superseded by main `10eeb1f55` (2026-10-30): the Python compiler is deleted
  and pre-w17 corpora are archived under `experiments/data/archive/`.
- 2026-10-29 (support reorg): tests that exercise several cards stay where
  they were (Corrupt Court Official with Ozai's Cruelty in `cards/tla.rs`, Epic
  Downfall with Power Word Kill in `cards/afr.rs`); Lightning Strike has no test
  of its own, so no `ptk.rs`/`ths.rs` test files exist yet.
- 2026-10-29 (concept review): world pool key becomes `"world"` (not per-world
  `"w17"`), describing the default pack; row `semantic_key` becomes `key`.
  Coordinate with f871a8 in one comment. The v1 coverage gate is left for a later
  Task rather than PR394.

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
- T3 slices 7–14 and item 16 card-level choices (Vehicles, mana, X, Lessons,
  grants, Honest Work, tap authority, Secret Tunnel LKI): `636f90bf8:scratch/questions.md`.

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

## T2 notes at the second and third merges (heads `89e320c48`, `a5587df5a`) — 2026-10-29

T2's operator choices are at `a5587df5a:scratch/questions.md`. New since
`d218e1dc2`: possible worlds v3 resample foretold cards (ordered slots,
`P(m, F)` fill factor); belief priors are not yet foretold-aware.

- 2026-10-30 (continuous effects as records), implement-time refinements of the
  accepted model, reversible:
  - A set fixed at resolution (CR 611.2c) is one record per object sharing a
    timestamp, rather than a `Locked(Vec<ObjectRef>)` record, so every read
    indexes by affected object.
  - Durations are `EndOfTurn` and `Indefinite` only; `EndOfCombat`,
    `UntilYourNextTurn` and `WhileSourceOnBattlefield` join when a card needs
    one (no T3 card does). "Grant ability" stays the Equipment/Aura static.
  - The timestamp clock is the object-id generator (a permanent's `id` is its
    entering timestamp; attaching renews `Permanent.timestamp`).
  - Rules effects (can't be blocked, exile if it would die, kept mana, hand
    size) are not abilities, so losing all abilities does not end them.
  - The Observation keeps `plus1_counters` (pre-T3 consumers) and `counters`
    holds the other kinds, so they partition rather than duplicate.
  - The model hashes fact ids into 4096 embedding buckets (collisions
    possible); reusing the semantic-card token table for fact ids is a
    follow-up if it matters.
  - Ability descriptions of continuous effects are generated uniformly
    (`continuous_description`), so some offer texts change wording
    ("can't be blocked until end of turn").
## T4 (488a40) assumptions — 2026-10-30

- T4 extends the open world in place, like T3 (no checkpoint uses it). T3's
  sync with main renamed that world w17 (one open world on main), so T4's
  w18 evidence became w17 at the T3 sync of 2026-10-30. New cards change the
  content digest, so w17 evidence is re-recorded once at gate.
- T3 sync (2026-10-30): T4's temporary-modifier effects became continuous-
  effect records. `ModifyTargetPlayerCreatures` is
  `ApplyContinuous { affected: AffectedSpec::EachTargetPlayerControls(..) }`
  (the affected set fixed as it resolves, CR 611.2c); its target spec is
  `TargetSpec::Player`, so the variant holds only its predicate (clippy's
  enum-size bound). `dealt_damage_this_turn` is permanent v5 column 36, after
  T3's timestamp rank (rows 38 wide); T4's used-ability flag columns are gone
  because T3 made used abilities fact rows.
- `EffectValue::SourcePower` reads the source's current power and is 0 once
  it has left the battlefield (LKI power is not captured); firebending X
  resolves while its attacker is normally still present.
- Count fields of `MillCards`, `SearchBasicLand`, `PutCountersOnEachMatching`
  and (design review) `DrawCards` are `EffectValue`; no `*Value` siblings.
- T4 card tests are appended at the end of `tests/cards/tla.rs`, after the
  cross-card section, rather than interleaved in `cards()` order.
- Slice 4: raid's intervening-if is not rechecked at resolution: attacking
  this turn cannot be undone within the turn, and both raid triggers resolve
  in the turn they fire.
- Slice 4: `cards_drawn` joined the observed turn history beside the three
  new facts (it was unobserved before); its column is v5 player 66.
- Slice 4: `PlayerSelector::Target` holds a boxed spec (clippy's enum-size
  bound); no card uses it yet. Amounts of `LoseLife`/`DealDamageToPlayer` are
  `EffectValue`, like `DrawCards.count`.
- Slice 4: a land entering is read from its printed card type as it enters.
- Slice 3: The Earth King counts attacking power-4+ creatures as its search resolves ("that many"), not as declared; one removed in response lowers the count.
