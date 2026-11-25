# MTG-430 — Planeswalkers, loyalty, empowerment and emblems (FDN + FRA)

Base `cf73d9778f`. MTG-420 (PR #593, head `c133c0308f` on 2026-11-24) is still
open; `lf sync main` once it merges, before slice 3 (emblem sources) and before
any slice touches `activated_this_turn`/`activation_used`, which #593 extends
("per-object activation limits").

The first completed cohort covers the ten FRA Commons and Annex lands in slice 8.
Resume slice 1 after the MTG-420 sync. Slice 2's attack declarations can
also be exercised with the existing constructed planeswalker if #593 is open.

Cohort: the 5 FDN planeswalkers, FRA's 40 `05_loyalty_tokens_emblems` rows, and
adjacent FRA rows that depend on 05 and compose once it exists. Oracle text
fetched from Scryfall 2026-11-24 (`set:fra (t:planeswalker or o:empower or
o:loyalty or o:emblem)`, `set:fra o:planeswalker`, `set:fdn t:planeswalker`).

## What exists today (checked against code, not the ledger)

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
- `ZoneType::Command` storage exists; nothing ever moves there.

Missing: printed loyalty, loyalty costs, the CR 606.3 quota and timing
overrides, attack targets other than the defending player, emblems, divided
damage, a loyalty-activation trigger/history, a zero-loyalty SBA exception,
proliferate, behold. No planeswalker card is defined.

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

### Slice 1 — Loyalty abilities and starting loyalty (primitive)

Types:
- `PrintedCharacteristics.loyalty: Option<i32>` (CR 306.5b printed starting
  loyalty; tokens may print `Some(0)` or `None`). Entry applies it as an
  intrinsic enters-with counter in the same place as Saga lore
  (`src/flow/zones.rs:676`): CR 306.5b makes it an intrinsic "enters with"
  replacement (CR 614.1c), so Doubling Season doubles it.
- `ActivatedAbilityDefinition.loyalty: Option<LoyaltyCost>` with
  `LoyaltyCost::{Fixed(i16), MinusX}` (signed; `+1`, `0`, `−3`, `−X`). Lowered
  to one `CostComponent::Loyalty(i32)` in `payment.rs` whose counter change is
  `EventCause::Cost` — never doubled (Doubling Season acceptance). A negative
  cost needs at least that many loyalty counters (CR 606.6); −X chooses X in
  0..=loyalty through the existing activation-X path.
- Quota: `Permanent.loyalty_activated_this_turn: bool` (CR 606.3), checked for
  every ability with `loyalty.is_some()`, cleared at untap/turn start beside
  `activated_this_turn`. Default timing: main phase, own turn, empty stack.
- Timing override: `PlayerModification::LoyaltyAtInstantSpeed { predicate }`
  as an `ApplyContinuous` record on the player (Jace's Machinations: Jace
  planeswalkers you control, until end of turn). It relaxes timing only; the
  quota still holds.
- A loyalty ability with mana output stays in `activated_abilities` and uses
  the stack (CR 605.1a); never a `ManaAbility`.

Consumers: v11 — loyalty is already a counter fact; add the ability's loyalty
cost to the activated-ability token/fact rows (`cards/tokens.rs`,
`agent/record_vocabulary.rs`), the quota flag beside `used_abilities`
(`observation.rs:893`) and the printed loyalty to card/permanent rows as an
appended named column (`ValueLayout` count kind). LLM/ability text renders
`[+1]: …`, `[−X]: …`, `[0]: …` (`ability_description.rs`). Academy shows the
ability label with its cost. Observability guard rows for the new field and
cost component.

Cards (proof cards; definitions land in slice 7): use a fixture-free real card
— The Theorist, Jace Beleren (+1 token, −2 per-opponent bounce, −6) — as the
directed test subject.

Checks: `cargo test -p managym-cards loyalty`; directed tests in
`cards/tests/cards/fra.rs`: second loyalty activation same turn rejected;
−3 with 2 loyalty not offered; −X offers 0..=loyalty; enters with printed
loyalty, doubled under Doubling Season; +1 under Doubling Season adds exactly
1; activation on opponent's turn not offered, offered under Jace's
Machinations, quota still enforced; zero-loyalty SBA after a −N.
Stress state: 6 planeswalkers × 4 abilities each with Doubling Season
(enumeration of loyalty offers is linear in abilities).

Deletes: none (new capability).

### Slice 2 — Attacking planeswalkers (primitive; changes the action space)

- `AttackTarget::{Player(PlayerId), Permanent(ObjectId)}` (planeswalker or
  battle the defending player controls/protects). `CombatState` holds
  `attack_targets: Vec<(PermanentId, AttackTarget)>` replacing the implicit
  `non_active_player()` defender; `Action::DeclareAttacker` carries
  `attack: Option<AttackTarget>` instead of `attack: bool`. Offers per
  attacker: no-attack, the player, each attackable planeswalker — complete and
  linear (attackers × targets).
- Combat damage to an attacked planeswalker uses the existing permanent damage
  path; trample excess goes to the attacked planeswalker, not its controller
  (CR 702.19c); an attacker whose planeswalker left combat deals no combat
  damage unless blocked (CR 506.4c). Defending player = controller of the
  attacked planeswalker (CR 506.2).
- No damage redirection: CR 306.7 was removed (Dominaria, 2018); noncombat
  damage reaches a planeswalker only by targeting it.
- `TriggerCondition::Attacks` gains an `attacked: AttackedSubject` field
  (`Any`, `Player(Actor)`, `PlaneswalkerControlledBy(Actor)`,
  `YouOrYourPlaneswalker`) — a field of the event, not a new variant
  (Jace, Reality Sculptor −3; Garruk, Curse Breaker −4 "attack one of your
  opponents").
- Attack restriction per target: `StaticRestriction::AttackersPerTarget(1)`
  granted to planeswalkers (Tomik). Declare-attackers legality must check the
  whole declaration, not attacker-by-attacker: a second attacker at that
  planeswalker is not offered.

Consumers: action rows gain an attack-target reference (one-based object join,
0 = player); LLM text "attack Vivien Reid"; Academy declare-attackers UI picks
a target per attacker through the shared offers (no browser-only semantics);
replay of old recordings: `attack: bool` is serialized — keep the wire key and
map `true` → `Player(defender)` on read (serialized keys keep their spelling).
Combat history (`AttackedThisTurn`, attack-batch triggers) records the target.

Checks: `cargo test combat` + new `managym/tests/rules/attack_targets.rs`:
attack planeswalker, damage removes loyalty, SBA; trample over planeswalker;
planeswalker removed mid-combat; Tomik limit; Jace −3 fires for attacks on you
and your planeswalker. Python env/Academy tests for declare-attackers offers;
Academy parity fixtures refreshed. Engine Budget: `bench_steps` digest moves
only if curated decks attack planeswalkers (they don't today) — expect no
digest change; record if it does.
Stress state: 10 untapped attackers into 6 opposing planeswalkers + 1 battle
(offers 10×8), plus Tomik on each planeswalker.

Deletes: `declare_attacker(permanent, attack: bool)` signature and the
`non_active_player()` defender in `combat_actions.rs:173/277`; the "to
defending player" prompt text.

### Slice 3 — Emblems in the command zone (primitive; after MTG-420 sync)

- `EmblemDefinition { key: "emblem.<name>", printed: PrintedCharacteristics }`
  in `managym/cards/src/emblems.rs`, reached through
  `managym_cards::emblem_catalog()`, keyed like tokens (CR 114: abilities only,
  no other characteristics; owned/controlled by its player). Registered as an
  `ObjectDefinition` row with status `Emblem` so learning tokens and the
  observation join one row (shifts definition ids → refresh Academy fixtures).
- `Effect::CreateEmblem { key }` places an emblem object in the player's
  `ZoneType::Command`. Emblem statics join the layer board and emblem triggers
  join `TriggerIndex` as ability sources outside the battlefield, using
  MTG-420's persistent ability-source representation (do not add a parallel
  source kind). They never leave; no SBA, no zone change.
- Observation: a command-zone row per emblem (public to both viewers), its
  abilities as fact rows; LLM text "Emblem — Vivien Reid: Creatures you control
  get +2/+2 …"; Academy shows emblems beside the player. Search clones and
  forks copy them; recording/replay round-trips them.

Cards: Vivien Reid −8 and Kaito −9 (FDN), Ajani Resolute −10, Chandra Torch of
Defiance −7 (FRA). Chandra, Chill's emblem waits with that card (pending).

Checks: directed: emblem buff persists after its planeswalker dies; emblem
trigger (Kaito: any player's spell) fires for both players' casts; two copies
stack; viewer observation shows the opponent's emblem; replay/fork identity.
Observability guard row for `CreateEmblem` and the command-zone object.
Stress state: 4 emblems of each kind per player with a full board (layer
board scan includes command-zone sources every rebuild).

Deletes: none.

### Slice 4 — Empower and the Jace token (composition; 22 cards)

Empower N (FRA keyword action): compose, no new verb —
`Seq[If(Not(Controls(planeswalker token with subtype Jace)),
CreateTokens("token.jace", 1)), PutCounters { kind: Loyalty, amount: N,
on: Chosen { predicate: Jace planeswalker token you control } }]`, written
through one constructor `Effect::empower_jace(amount: EffectValue)` in the
state crate. Not targeted; a chosen existing Jace (including a copied one)
is fine; creation and counters are one instruction, so SBA does not see the
0-loyalty token (release notes). Under Doubling Season the creation doubles
tokens and the counters double (effect cause); counters go on one chosen
Jace, not both. `token.jace`: blue Jace planeswalker token, `loyalty:
Some(0)`, `[−1]: Surveil 1`, `[−3]: Draw a card`.

If the ability text should read "Empower Jace N" rather than the composition,
the description template recognises the constructor's shape; decide at
implementation (questions.md).

Cards: Academic Ascent, Arcane Amphisbaena, Campus Crier, Hexhaven Battalion,
Keeper of the Quiet Hour, Mindseeker Oculus, No Admittance, Protege's
Awakening, Repurposed Enforcer, Rewrite Regrets, Solve for Disappointment,
Tam's Resistance, Vraska's Final Mercy, Violent Echoes (`ExcessDamage`,
planeswalker-aware excess), Overwrite the Multiverse (needs
`EffectValue::MovedThisWay` — the count of objects the frame's last `Move`
actually moved; small value, added here), Fatehold Charm (verify `Move` of a
spell from the stack exists; pending on cast/payment if not), Theorist's
Proxy (one-shot "next spell you cast this turn can't be countered" — verify a
delayed cast watcher exists, else add `Duration::UntilNextSpellCast`), Plan
for All Outcomes, Inspired Tethermage (`PutsCounters` already), Jace's
Machinations (slice 1 override), plus Way of the Mentor / Necromancer bodies
(their loyalty triggers compose: "put a loyalty counter on each planeswalker
you control").

Checks: `cargo test -p managym-cards empower`; Jace created when none; chosen
when one exists (two Jaces → a choice); Doubling Season: two tokens, counters
on one; illegal sole target of Academic Ascent → no empower; No Admittance
target dies → empower still happens; Violent Echoes excess into a planeswalker
target; Overwrite with a replacement (exile prevented) counts actual exiles.
Stress state: Doubling Season ×2 + repeated empower (token count growth).

Deletes: none.

### Slice 5 — Granted loyalty abilities and zero-loyalty exception (primitive + 13 cards)

- "Planeswalkers you control have [−N]: …": `StaticAbility::Continuous` over
  `Each { planeswalkers you control }` with `Modification::GrantActivated`
  of a loyalty ability; the stacked ability keeps `AbilityGrant` grantor
  identity; "this planeswalker" in granted text is the holder (`This`).
  Verify the layer path accepts a controlled-set scope; if only
  `AttachedHas` exists, generalise it to a selector scope (reshape, do not
  add a `PlaneswalkersYouControlHave` variant).
- Activation targeting needs independent target clauses (Way of the Warlord:
  up to one creature/planeswalker AND a player). Extend the activated
  ability's targeting to the spell's `targeting: Vec<TargetRequirement>`
  shape; delete the single-requirement accessor.
- Zero-loyalty exception: `StaticAbility::StateBasedActionDoesntApply
  { action: StateBasedAction::ZeroLoyalty, affected: CardPredicate }`
  (Sanctum Lurker), read only by the 704.5i check — destruction, damage and
  the legend rule still apply.

Cards: Way of the Cryomancer (−3 delayed "next instant/sorcery you cast this
turn" copy — reuse copy/retarget), Way of the Deathbringer (`if_you_do` Beast),
Way of the Healer, Way of the Pyromancer (+1 Add {R} on the stack), Way of the
Warlord, Way of the Wildspeaker, Avatar of Burgeoning Echoes (−10), Kiora of
Salt and Sand (−8 Leviathan; attack trigger in slice 6), Sanctum Lurker,
Tomik (static grant from slice 2), Way of the Mentor, Way of the Necromancer,
Way of the Mind Sculptor/Paradox bodies (triggers in slice 6).

Checks: granted ability shares the holder's one-per-turn quota with its
printed ones; granted ability disappears when the Way leaves; Jace token with
Sanctum Lurker survives at 0, dies to damage-destroy/legend rule as normal;
Warlord with no creature target still hits the player.
Stress state: 4 different Ways + 6 planeswalkers (granted abilities × holders
in the activation enumeration and layer rebuild).

Deletes: the single-target activated-ability accessor (replaced by clauses);
`AttachedHas` if fully subsumed by the selector-scoped grant.

### Slice 6 — Loyalty activation events (primitive + 7 cards)

- `GameEvent::LoyaltyAbilityActivated { player, source, counters_removed }`
  emitted by activation (after costs, CR 602.2i), with the cost receipt.
- `TriggerCondition::ActivatesLoyaltyAbility { who: Actor }` (one template;
  "whenever you/an opponent activates a loyalty ability"). "If you removed two
  or more loyalty counters to activate it" reads the event's receipt through an
  `ActiveIf` condition on a triggering value, not a new variant.
- History: loyalty activations this turn per player in the existing turn
  history (`Condition::Compare` over a new `EffectValue::LoyaltyActivationsThisTurn`).

Cards: Ajani Unrelenting (static trigger), Gideon the Oathless, Way of the
Mind Sculptor, Way of the Paradox (cumulative `AdditionalLandPlay`), Kiora of
Salt and Sand (attack condition), Ajani Resolute's life-gain loyalty trigger
(existing `GainsLife`), Garruk, Curse Breaker's ETB draw.

Checks: granted and printed loyalty activations both fire; Mind Sculptor's
−X with X=1 doesn't draw, X=2 does; Paradox allows two extra lands after two
activations in one turn (each planeswalker once); Kiora condition false
before any activation.

Deletes: none.

### Slice 7 — Planeswalker cards (13 cards; two small primitives)

FDN: Ajani, Caller of the Pride (−8 X = life total via `LifeTotal`), Chandra,
Flameshaper, Kaito, Cunning Infiltrator, Liliana, Dreadhorde General, Vivien
Reid. FRA: Ajani Resolute, Ajani Unrelenting, Chandra, Torch of Defiance,
Garruk, Curse Breaker, Garruk, Veiled Butcher, Jace, Reality Sculptor, The
Theorist, Jace Beleren.

Small primitives this slice owns:
- Divided damage (Chandra, Flameshaper −4: 8 among any number of target
  creatures/planeswalkers): `TargetRequirement.division: Option<EffectValue>`
  announced during activation/casting (CR 601.2d, 602.2b), each target ≥ 1.
  Offer it sequentially (choose targets, then assign amounts one target at a
  time with remaining-total bounds) so offers stay linear, never an
  enumeration of partitions. Stress state: 8 damage across 8 legal targets.
- Keep one of each permanent type (Liliana −9, CR 205.2a types): each
  opponent chooses, per type in order, up to one permanent of that type into a
  frame binding (`Chosen` into a named binding), then `Sacrifice` over `Each`
  of their permanents `except` that binding. A multi-type permanent chosen for
  one type also satisfies the others (it is kept once). Stress state: 30
  permanents of mixed and multi-type permanents (artifact creature lands).

Composing pieces to verify per card: Kaito's combat-damage loyalty trigger
(`DealsCombatDamage`), Vivien +1 (look at top four, `SelectCards`), Chandra
Flameshaper +2 (`ExileTopCardsToPlay` choose one of three) and +1
(`CreateTokenCopy` with haste and end-step sacrifice), Chandra Torch +1
(exile top, may cast, else 2 to each opponent — `WhenYouDo`/`If` over the cast
result), Garruk Veiled Butcher's opponent death-exile replacement and
discard-receipt draw, Jace Reality Sculptor 0 (aggregate Jace loyalty
condition via `BattlefieldMatching` sum — add `EffectValue::CountersAmong`
if absent), The Theorist's opponent draw-step trigger and per-opponent
target clause.

Checks: one directed test per planeswalker covering each loyalty ability, in
`cards/tests/cards/{fdn,fra,kld}.rs` (Chandra, Torch of Defiance's first
printing is KLD — her definition goes in `kld.rs`, not `fra.rs`).

### Slice 8 — Adjacent planeswalker-aware composition

Commons/Annex lands ×10 are implemented in `cards/src/fra.rs`. The native
land-play and payment cases use a constructed planeswalker until the first
printed planeswalker joins the catalog. Academy admitted a four-of-each
deck at seed 7, accepted “Keep hand” and “Play Fatehold Annex”, and restored
the same projection on reconnect.

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
