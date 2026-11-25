# MTG-430 — Planeswalkers, loyalty, empowerment and emblems (FDN + FRA)

Base `cf73d9778f`; synced main through MTG-420 (#593, `45ddf3bca7`) and
later main (`964699a20b`). MTG-420 supplies exact object identity for stacked
abilities. Live layers and triggers also resolve exact command-zone sources.

Done:
- Slice 8's ten FRA Commons/Annex lands (`b17ed61f62`, ledger `55430f5c91`);
  pending-authority ledger notes (`cc77598c9e`).
- Fixed loyalty, quota, `LoyaltyAtInstantSpeed`, printed loyalty and The
  Theorist across play, v11, text and Academy (`a5534ee7bb`). Three directed
  native cases; the cp313 wheel passed shared offer, fork and observation
  checks and replayed all five Academy opening fixtures.
- Variable loyalty and the shared activation continuation (`aa35e1a2b6`).
  All eight FRA cases and Katara’s mana-X case passed in the debug build.
  The rebuilt cp313 wheel passed Academy’s loyalty offer/fork test and
  reopened the saved combat Command at the next Ogre’s declaration.

- Slice 2 (`0674982740`) gives every declared or entering attacker a defender. Tomik grants
  each controlled planeswalker its own declaration limit. The v11 rows,
  combat preview, compound offers, Academy draft and text observation carry
  the choice. All nine FRA cases and six directed combat cases passed. The
  cp313 wheel cold-restored all nine positions in the five Academy opening
  fixtures. Across their 444 prefix decisions, all 109 player-only attack
  frames matched their previous bytes. The player-only object-bound Command
  at revision 12 retained its original fingerprint and cold-restored. A
  separate Command naming Jace and its attacker also cold-restored.
- Slice 4, empower and `token.jace` with its 22 FRA cards (`4966204033`,
  fixtures `2327a41b94`).

- Slice 3, command-zone emblems (`EmblemDefinition`, `emblem_catalog()`,
  `Effect::CreateEmblem`) and Vivien Reid, Kaito, Ajani Resolute, Chandra,
  Torch of Defiance (`833f67e3ed`, fixtures `52b05b2fa8`, coverage guard
  `eb842d8d57`). 24 FRA cases; the cp313 wheel cold-restored Ajani's emblem
  and the five fixtures (444 choices kept labels/IDs, 109 attack frames kept
  bytes).

- Slices 5 and 6: Way grants through `StaticAbility::Continuous` +
  `GrantActivated` (program abilities sharing the holder's quota),
  `StaticAbility::StateBasedActionDoesntApply { ZeroLoyalty }`,
  `GameEvent::LoyaltyAbilityActivated` (from the stacked ability's
  `loyalty_change`), `TriggerCondition::ActivatesLoyaltyAbility { who,
  min_counters_removed }`, `EffectValue::LoyaltyActivationsThisTurn` and the
  v11 `player_turn_state.loyalty_abilities_activated` column (appended). Activation
  targeting records each target's clause (`target_req_indices` on
  `ActivatedTargeting` and `Operation::Activation`); the positional
  `target_clause_indices` and `ActivatedAbilityDefinition::target_spec` are
  deleted. Cards: the eight remaining Ways, Avatar of Burgeoning Echoes,
  Kiora of Salt and Sand, Sanctum Lurker, Gideon the Oathless; tokens
  `token.beast_4_4_g_trample`, `token.leviathan_8_8_u_hexproof`.
  Commits `6429f0f54a`, `553ce2b149`, fixtures `d83da6e53b`, tour
  `370c804c4d`; six directed FRA tests (quota and leaving, Warlord, granted
  resolution, Lurker, activation triggers with Paradox lands, Kiora/Gideon).
  The pass recorded no test-run results; gate reruns them. Not built: the
  stress state (4 different Ways + 6 planeswalkers, granted abilities ×
  holders in activation enumeration and layer rebuild), which moves to the
  remaining pass. `AttachedHas` stays (see questions.md).
  Lesson: a declined up-to clause made the next clause's target count as
  clause 0, so Warlord re-asked for its player. Positions never name clauses.
- Slice 7 (`580ff7d36e`, cards/tests/stress `a256b00666`): divided damage
  (`TargetRequirement.division`, sequential `PendingChoice::DivideAmount`
  offers >= 1 per target, CR 601.2d/602.2b/608.2b, `target_divisions` on stack
  objects), keep-one-of-each-type (`CardType::PERMANENT_TYPES`,
  `Effect::Remember`, `Except::Remembered`), controller-scoped exception
  (`Except::Matching`), until-your-next-turn delayed attack triggers
  (`DelayedEvent::Attacks`, `DelayedTriggerEvent::Attacks`,
  `DelayedExpiry::UntilYourNextTurn`), choose-from-exiled-top
  (`ExileTopCardsToPlay.choose`, `Effect::GrantPlay`), opponent death-exile
  replacement (`ReplacementEffect::ExileIfWouldDie`, `EventBinding::SacrificedBy`),
  and `ObjectSelector::LibraryExceptBottom`. Appended v11 relation columns
  `until_next_turn` and `division` (`tests/public_api.json` recorded).
  Cards: `Ajani, Caller of the Pride` (`m13`, + `token.cat_2_2_w`), `Chandra,
  Flameshaper` (`fdn`), `Liliana, Dreadhorde General` (`war`), `Ajani
  Unrelenting`, `Garruk, Curse Breaker`, `Garruk, Veiled Butcher`, and `Jace,
  Reality Sculptor` (`fra`). Verified: all 7 directed loyalty tests + 3
  observability/subtype guards (`cards` debug), `conformance_tests` (2/2),
  `scripts/public_api.py check`, `cargo fmt --check`, and `bench_states` stress
  states (`way-grants`: 25 actions / 1 board; `divided-damage`: 10 target
  offers -> 7 division offers; `keep-one-each-type`: 15 linear artifact picks
  on 30 mixed/multi-type permanents).
  Lesson: sequential `DivideAmount` auto-assigns 0/1 targets and `total <= N`
  targets, keeping offers strictly linear (at most `remaining - tail_targets`
  per step); keep-one-of-each-type composes via per-type `Remember` +
  `Except::Remembered` with linear `N` offers and matches CR multi-type rulings.

Already red on CI at `d1ef91d814` (run 45568225341), before slices 5–6, and
left for gate: `observability::` legacy card-row parity (four tests,
card column 8 vs permanent column 6),
`observation_layout::named_layout_matches_every_encoder_width`
(agent_permanents), `content_pack_contract_covers_env_roots_siblings_and_rollout_slots`,
`definitions::every_authored_subtype_has_a_category` and the frozen
semantic-conformance cases.

## Passes remaining

1. Slice 8's remaining ~30 adjacent cards plus the integrated interaction
   checks, then close the ledger. At gate run debug nextest, Engine Budget and
   the pinned pilot, recording a reason if its output moves. Keep main’s
   MTG-420 receipts at their original revision.

Cohort: the 5 FDN planeswalkers, FRA's 40 `05_loyalty_tokens_emblems` rows, and
adjacent FRA rows that depend on 05 and compose once it exists. Oracle text
fetched from Scryfall 2026-11-24 (`set:fra (t:planeswalker or o:empower or
o:loyalty or o:emblem)`, `set:fra o:planeswalker`, `set:fdn t:planeswalker`).

## Engine starting points

- `CardType::Planeswalker`, `CounterKind::Loyalty`, damage removing loyalty
  (`src/flow/damage.rs:172`), zero-loyalty SBA (`src/flow/sba.rs:171`),
  `TargetSpec::PlayerOrPlaneswalker`, observation type flags and counter facts.
- `ActivatedAbilityDefinition` (`state/src/card.rs:861`) has
  `remove_counters`/`add_counters` cost components, `sorcery_timing`,
  `once_each_turn`, `exhaust`, `x_min`; ledgers `Permanent.activated_this_turn`
  / `exhausted`; legality in `src/flow/play.rs` (`activate_ability_action`,
  `check_activation_terms`, `record_activation`).
- `CounterProvenance { cause: Effect | Cost | TurnAction }`; Doubling Season's
  `ModifyCounters { effects_only: true }` already skips cost counters and
  counts enters-with counters (`proposed_event.rs:1393`, test at `:1740`).
- `Modification::GrantActivated` / `GrantTriggered` / `GrantStatic` and
  `AbilityGrant { grantor, index }`; `Duration::UntilYourNextTurn`;
  `EffectValue::ExcessDamage`, `ExileTopCardsToPlay`, `CreateTokenCopy`,
  `AdditionalLandPlay`, `CantBeCountered`, `EntersTappedUnlessControls`.
- Emblems are created directly in their controller’s command zone.

The Theorist and Tomik are defined. Remaining primitives: divided
damage, loyalty activation events, a zero-loyalty SBA exception (proliferate
and behold are pending elsewhere).

## Classes and Rooms against loyalty (decided before fixing the representation)

| | Loyalty (CR 306, 606) | Class (CR 716) | Room (CR 709) |
|---|---|---|---|
| State | `Loyalty` counters: Doubling Season, proliferate, damage and Mabel touch them | level designation, not counters | per-door unlocked designation |
| Action | ordinary activated ability, uses the stack (mana output too) | activated level-up ability, sorcery timing | special action (CR 116.2m), no stack |
| Limit | one loyalty ability per permanent per turn, across all its loyalty abilities | gated by current level = N−1, no turn quota | each door once |
| Cost | signed loyalty cost component | mana | the door's mana cost |

Shared and reused: `check_activation_terms` timing, payment lowering, the
per-permanent ledger shape. Not shared: the state (counter vs `u8` level vs door
set), the action and the limit. So loyalty gets its own cost component and its
own quota ledger; nothing is named "level", and no "activation counter"
abstraction is introduced. A later Class would be `Permanent.class_level` plus
an activation condition; a Room would be a door set plus `Action::UnlockDoor`.
This Task ships neither; record the table in `docs/engine.md` with slice 1.

## Sorting the batch

**Pending on another Task's authority** (no shortcuts; ledger notes added):

- Countersculpt, Theorist's Sanctum — behold (cast/payment authority).
- Tam, the Possibility — proliferate X times with resumable per-iteration
  choices (resumable results/choices).
- Gideon's Memorial — spend-only-on-planeswalker-spell mana needs `ManaUse` by
  card type (cast/payment); its other abilities compose. Build it if `ManaUse`
  gained card-type restrictions by then, otherwise pending.
- Chandra, Chill of Compliance — "spend only to cast a noncreature spell" mana:
  same `ManaUse` dependency. Its emblem and −X stun compose; card waits as a
  whole rather than shipping a partial definition.

Everything else in the cohort is built here, on the slices below.

## Slices (build order, riskiest first)

Slices 1 (loyalty abilities, `a5534ee7bb`, `aa35e1a2b6`) and 2 (attack
targets, `0674982740`, fixtures `5ecb50ceee`) are built; their design is in
`docs/engine.md` and the commits. The Classes/Rooms table above went to
`docs/engine.md` with slice 1. Deferred to gate: the pinned pilot (record with
`--reason` if the attack row moved) and Engine Budget (`bench_steps` digest
expected unchanged; curated decks attack no planeswalker).

Open defect (slice 4): no replacement can divert a battlefield-to-exile move,
so Overwrite's diverted-exile case stays untested until one exists.

Slices 1–7 are built (see `Done:` above). Remaining work is Slice 8 below.

### Slice 8 — Adjacent planeswalker-aware composition

Commons/Annex lands are done (above); their native tests use a constructed
planeswalker until a printed one joins the catalog.

Still to build are Teyo ×2 ("put a loyalty counter on it if it's a
planeswalker" — `If` over the target's type), Your Fate Ends Here, Terminal
Criticism, Refute Destiny, Extended Absence, Fulminous Forte, Essence Burn,
Wrath of the Bloodmane, Silence the Echo, Break Under Pressure, Awaken the
Inferno, Craftwork Crusher, Compel Brutality (damage equal to loyalty — add a
counter-count value if `Property` can't read it), Edgar, Liliana the Faultless,
Liliana the Repentant, Massacre Girl, Winter, Mabel, Living Library, Identity
Echo, Vindictive Triumph, Entrust the Spark, Loyal Tutor, Return to the Light
Realms, Mind Meanderer, Lich's Relic, Loot (if its other parts compose), Face
Yourself. Build those that compose; any that reveal a missing authority go to
the pending list with that authority named, not shortcuts.

Checks: one test per card in `cards/tests/cards/fra.rs`, grouped by shape
(removal unions, Commons entry, Teyo hybrid target).

## Integrated interaction checks (author's synergies)

- Doubling Season + planeswalker entry + `+1` cost + empower: entry doubled,
  cost not, empower doubled (the acceptance case).
- Jace's Machinations + empowered Jace token: instant-speed activation on the
  opponent's turn, once.
- Sanctum Lurker + Jace token −3 to 0 survives; Way of the Mentor/Necromancer
  then grow it.
- Ajani Unrelenting + Way of the Paradox + Ajani Resolute: one activation
  triggers Cadet, life, extra land, and Ajani Resolute's lifegain loyalty.
- Vivien −8 emblem, then Vivien dies to an attack: emblem still pumps; opponent
  viewer sees it.
- Tomik + two attackers at one planeswalker: second offer absent; both can
  still attack the player.
- Kaito unblockable creature attacks the opponent's planeswalker: no Kaito
  trigger (damage to a planeswalker is not to a player).
- Mabel removes loyalty → SBA; Teyo on a planeswalker creature (both counters).

## Consumer invariants (every slice)

Offers complete under `skip_trivial`; replay, forks and search clones carry
loyalty ledgers, attack targets and emblems; viewer privacy unchanged (all new
state is public); v11 columns appended, never reordered; frozen v9/v10 inputs
untouched; `protocol/*.schema.json` patched with each wire type change;
`tests/model/test_pinned_pilot.py` — declare the break with
`scripts/pinned_pilot.py record --reason` if the attack action row moves.
Engine Budget within 10%; rebuild the extension and run debug nextest
(`CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2`) before
publishing.

## Ledger

Update `fdn.jsonl` (`implementation.status`) and `fra.jsonl` rows as each
slice lands; pending rows carry the missing authority now.
