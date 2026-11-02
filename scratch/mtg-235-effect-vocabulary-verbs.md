# MTG-235: Effect verbs over an ObjectSelector

Directive: Task MTG-235, from Jack Heart's concept review of 2026-11-02. Base
is main `4c8324a59`; main had not moved at the 2026-11-02 realign. MTG-225
(T6) has not published, so there is no `TriggerCondition::Chapter` here to
rename (see questions.md, "MTG-235").

Status: implemented and compressed; awaiting gate and Jack Heart's review of
the one cohesive PR (the directive stops there).

## Outcome (implemented)

`Effect` verbs name what they act on with one `ObjectSelector`, resolved by one
`Game::select` (`flow/resolution.rs`) and acted on by one arm per verb. The
object-fused variants are gone.

```rust
pub enum ObjectSelector {
    This,                 // the source object; after it leaves, the card it
                          // became in the graveyard or exile (CR 400.7)
    Target(TargetSpec),   // the clause's chosen target (declares the clause)
    Triggering,           // the permanent whose event fired the trigger
    Attached,             // what the source Aura/Equipment is attached to
    CreatedThisWay,       // the token this resolution created last
    Grantor,              // the exact object that granted the ability
    Moved,                // the objects the latest Move/Sacrifice moved
    Chosen { predicate, another },  // a permanent you control, chosen now
    Each { of: AffectedSpec, another },  // every permanent the spec names
}
```

Verbs: `PutCounters { on, kind, amount }`, `PutCountersFrom { from, on }`,
`Sacrifice { what, optional }`, `Move { what, to, tapped, under, with_counter,
linked }`, `Tap { what }`, `Untap { what }`, `RemoveCounters { from, up_to,
chosen }`, `RemoveAllCounters { from }`.

"If you do" is `Effect::If { condition: Condition::Moved, .. }`
(`Effect::if_you_do(then)`): `EffectFrame.moved` records what the latest
`Move`/`Sacrifice` moved; `ObjectSelector::Moved` reads it ("return it").
`frame.this_way` stays the latest discard/mill ("milled this way"), because
Avatar Destiny mills, moves itself to hand, then reads the milled cards.

`Move` to `Battlefield` enters under the resolving player (CR 110.2b) unless
`under` names one (`PlayerSelector::Owner` is new); to `Library` shuffles into
the owner's library; `linked` keeps the exile link rules (`WithSource`,
`UntilSourceLeaves`: optional implicit target, source must be present).
`Sacrifice { optional: true }` is read only with a `Chosen` selector; the arm
clears `frame.moved` before offering the choice, so declining needs no special
case.

Compositions (fused verbs that depended on which object they hit):
- ReturnToHandDrawIfControlled → `[Move{Target,Hand}, If{Moved, [If{TargetControlledByYou, [Draw 1]}]}]`
- ExileAndReturn / ExileSourceAndReturn → `[Move{X,Exile}, Move{Moved,Battlefield,under}]`
- ExileWithControllerToken → `[Move{Target,Exile}, ForPlayer{TargetController,[CreateToken]}]`
- MoveSourceFromGraveyard{then} / SacrificeSource{then} / MaySacrifice{then} → verb + `if_you_do(then)`

Departures from the directive's sketch, each recorded in questions.md: the
`Linked` selector is `Grantor`; `Target` carries the `TargetSpec`, not a
clause index; `Move` has `under`, `with_counter` and `linked` instead of
`transformed` and `until` (no card needs a transforming or timed move; the
exile link is `linked`); `Chosen` and `Each` are added for "sacrifice a
creature" and "each other creature you control".

Descriptions for the Academy and the LLM text observation come from the
native `ability_description.rs`, which renders every selector; Python has no
second renderer. Learning tokens derive from the definitions through serde
(`cards/tokens.rs`), so the new variants and selectors reached the catalog
with no vocabulary edit; `tests/cards/observability.rs` witnesses every
`ObjectSelector` and `PlayerSelector` and each new verb in the tokens.

## Removed

`Effect::{PutCountersOnTriggeringPermanent, PutCountersOnCreatedToken,
PutCountersOnSource, PutCountersValue, PutCountersOnEachMatching,
PutTriggeringCountersOnSource, PutSourceCountersOnTarget, MaySacrifice,
SacrificeSource, SacrificeTriggeringPermanent, TapSource, UntapSource,
TapAttached, TapTarget, UntapTarget, ReturnGraveyardTarget,
ReturnToHandDrawIfControlled, ReturnToHand, ReturnGrantorToHand,
ReturnSourceToHand, ReturnSourceToBattlefield, ReturnSourceToBattlefieldTapped,
Exile, ExileSourceAndReturn, ExileAndReturn, ExileAttached, ExileWithSource,
ExileUntilSourceLeaves, ExileSourceSpell, ExileWithControllerToken,
MoveSourceFromGraveyard, ShuffleAttachedIntoLibrary,
RemoveAllCountersFromAttached}` (33), `GraveyardReturn`,
`OptionalSacrifice.then`, with their resolution arms, description arms,
observability rows and heuristic names. `Effect` has 79 variants (was 107 in
the directive's count at 657 cards). No source, test, doc or fixture
references a removed name except WORLDS.md's record of the change and the
dated analysis in `docs/research/semantic-kernel.md`, which stays historical.

Fused object-in-name verbs that remain (17; each a different choice or rule,
not an object for an existing verb): ReturnMilledThisWay,
DistributeCountersToTappedOpponentCreatures, AirbendAll,
DamageOpponentCreatures, DestroyCreaturesLosingAbilities,
EachPlayerKeepsCreatures, ExileTopCardPlayThisTurn, ExileUntilNonlandCastFree,
ExileLinked, CastExiledWithSource, AttachSourceTo, OwnerPutsIntoLibrary,
PlaceTargetInLibrary, PutTopCardsInHand, CopyTriggeringObject,
TargetCreaturesDealPowerDamageToLastTarget, FightLastTarget. The fused set
shrank by two thirds, past the directive's halving threshold.

## Acceptance (all met on this branch)

- Debug `cargo test` in `managym/`: every directed card test unchanged (485).
- `tests/cards/observability.rs`: every new variant and every `ObjectSelector`
  named and witnessed in the tokens.
- Learning tokens and content digest changed; w17 evidence re-recorded in
  place (semantic-kernel conformance, authored-match receipt, both parity
  corpora, the five `academy/fixtures/*-w17.json`, belief v3 receipt pins,
  rules-audit citations); WORLDS.md and AGENTS.md record it; no version bump.
- `manabot/sim/heuristic.py` reads the new shapes (`Move` of a battlefield
  target off the battlefield is `MoveTargetAway`); `tests/sim/test_heuristic.py`
  polarity table unchanged.

## Remaining

- Gate and Jack Heart's review of the PR. The extension `.so` is untracked;
  CI rebuilds it.
- `TriggerCondition::Chapter` → `CountersReach { kind, at_least }` when
  MTG-225 (T6) publishes; T6 may write its new effects in this shape directly.
- Not in scope here (the directive names five families and stops at one
  PR): the follow-on candidates under "Concept review" below belong to a
  separate Task after Jack Heart's review.

## Concept review (2026-11-02, headless, against the accepted design)

Accepted experience: a card author writes the verb and names its object with
one selector; the policy sees one verb token with selector and zone
sub-tokens, so `Move { This, Exile }` and `Move { Target, Hand }` share a
word; the Academy and LLM text read as the printed sentence (`Return this
card to the battlefield tapped`, `Exile it`), never as the mechanical form.
The implemented change delivers that for the five directed families; its
departures from the sketch (`Grantor`, `Target(spec)`, `Move { under,
with_counter, linked }`, `Chosen`, `Each`) each follow from a real card and
are recorded in questions.md. The nested `if_you_do([If
{TargetControlledByYou}])` on True Ancestry reads as the card; a
`Condition::AllOf` would flatten it but no second card asks.

What the accepted decision already implies, outside this Task's directive
(record only; separate Task after Jack Heart's review):

- Verbs that still bind their object as `target: TargetSpec` (`Destroy`,
  `Airbend`, `GainControl`, `DealDamage`, `DealDamageValue`, `CounterSpell`,
  `CounterAbility`, `Earthbend`, `DiscardTargetPlayer`) are the second way
  to say `ObjectSelector::Target(spec)`. Four of the 17 "remaining" verbs are
  those verbs over `Each`, not distinct rules: `MassDamage` and
  `DamageOpponentCreatures` are `DealDamage` over `Each`, `AirbendAll` is
  `Airbend` over `Each { another }` (excluding the clause's target rather than
  the source, which `Each` would need to express), and
  `DestroyCreaturesLosingAbilities` is `Destroy` over `Each` plus the
  ability-loss rule. Converting them means those verbs take an
  `ObjectSelector` first.
- Value twins (`DealDamage`/`DealDamageValue`, `GainLife`/`GainLifeValue`,
  `CreateToken`/`CreateTokensValue`, `AddMana`/`AddManaValue`) are the same
  verb split by whether the amount is fixed; this PR merged
  `PutCountersValue` into `PutCounters { amount: EffectValue }` and
  `From<i32> for EffectValue` already exists, so the same merge applies.
- `AffectedSpec` and `ObjectSelector` overlap (`Source`/`This`,
  `Target`/`Target`, `TriggeringPermanent`/`Triggering`); the accepted sketch
  wrote `Each(AffectedSpec)`, so one "which objects" vocabulary is a later
  simplification, not a defect of this change.

Partial selectors are deliberate and documented: `Chosen` is offered only by
`Sacrifice`, `optional` only reads with `Chosen`, `PlayerSelector::Owner`
only `Move` reads. A general resolution-time choice ("tap a creature you
control") waits for a card; no product choice is open there.

Open product choices: none. Next proof: the gate on this branch and Jack
Heart's review of the PR.

## Check-result

2026-11-02 realign — the extension rebuilt from this head (`uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`) differed from the one the implement pass tested; against the rebuilt `.so`, `uv run --extra dev pytest tests/sim/test_heuristic.py tests/semantic/test_learning_projection.py tests/semantic/test_learning_projection_env.py tests/semantic/test_current_content.py tests/semantic/test_text_observation.py tests/semantic/test_static_ability_tokens.py tests/verify/test_observation_parity.py tests/belief/test_state.py tests/academy/test_choice_navigation.py tests/academy/test_trigger_order.py tests/academy/test_learn_setup.py -q`: 152 passed. Rust: debug `cargo test` (cards 485, lib 132, conformance/replay/authored), clippy and fmt clean at the compress pass; no Rust source changed since.

## Concept review with Jack Heart (2026-11-02, in the Task conversation)

Page: `scratch/concept-review.html`. Jack Heart decided 2026-11-02 (answers
"1a 2a 3a 4b" in the Task conversation; the second "3a" read as 2a) and then
said "after that, ship it. this looks good". Accepted direction, amending
PR #412 before review:

- **1a.** Every verb that acts on objects takes `what: ObjectSelector`.
  `Destroy`, `Airbend`, `GainControl`, `DealDamage`, `CounterSpell`,
  `CounterAbility`, `DiscardTargetPlayer`, `Earthbend` lose `target:
  TargetSpec`; `MassDamage`, `DamageOpponentCreatures`, `AirbendAll` and
  `DestroyCreaturesLosingAbilities` become the base verb over `Each` (with
  the ability-loss rule kept for the last). `AirbendAll` needs `Each` to
  exclude the clause's target (`except: Target` or equivalent); keep it a
  one-rule verb only if that cannot be expressed.
- **2a.** `AffectedSpec` folds into `ObjectSelector`: `ApplyContinuous.affected`
  is an `ObjectSelector`, `Each { of }` holds the predicate scope directly
  (`You`, `EachYouControl`, `EachOpponentControls`, `EachMatching`,
  `EachTargetPlayerControls` become selector or scope forms), and the layer
  computation in `flow/statics.rs` keeps its own private resolution of the
  same selector.
- **3a.** Value twins merge on `EffectValue` in the same change:
  `DealDamage`/`DealDamageValue`, `GainLife`/`GainLifeValue`,
  `CreateToken`/`CreateTokensValue`, `AddMana`/`AddManaValue`.
- **4b.** `Chosen` is a general selector: `Game::select` offers the choice for
  any verb ("tap a creature you control"), not only `Sacrifice`.

Same acceptance as the first pass: every directed card test unchanged in
debug `cargo test`, observability witnesses every verb and selector, tokens
and content digest re-recorded in place, w17 stays, no version bump;
WORLDS.md and AGENTS.md updated. One cohesive PR (#412 amended), then ship.

Original mismatch and questions, kept for the record:

The one consequential mismatch: a card author still learns three ways to say
which objects a verb acts on, and the policy sees three token shapes for one
idea. `Move { what: ObjectSelector::Target(Creature) }` tokens as
`Effect::Move · what · ObjectSelector::Target · TargetSpec::Creature`;
`Destroy { target: TargetSpec::Creature }` tokens as `Effect::Destroy ·
target · TargetSpec::Creature`; `ApplyContinuous { affected:
AffectedSpec::Target(Creature) }` tokens as `… affected · AffectedSpec::Target
…`. In `cardsets/tla.rs` the three appear 106, 71 and 82 times.

Questions as put:

1. Verbs still binding `target: TargetSpec` (`Destroy`, `Airbend`,
   `GainControl`, `DealDamage`, `CounterSpell`, `CounterAbility`,
   `DiscardTargetPlayer`, `Earthbend`): take `what: ObjectSelector`?
   (a) amend #412, (b) second PR in this Task after #412 merges,
   (c) separate Task, (d) leave as is.
2. `AffectedSpec`: fold into `ObjectSelector` (`Each { of }` becomes the one
   "all of them" and `ApplyContinuous.affected` is an `ObjectSelector`), or
   keep it as the payload of `Each` only?
3. Value twins (`DealDamage`/`DealDamageValue`, `GainLife`/`GainLifeValue`,
   `CreateToken`/`CreateTokensValue`, `AddMana`/`AddManaValue`): merge on
   `EffectValue` in the same change as 1, separately, or not?
4. `Chosen` beyond `Sacrifice` ("tap a creature you control"): wait for a
   card that needs it (current), or generalize `select` now?

Counterexample kept: `AirbendAll` excludes the clause's chosen target, not
the source; `Each { another }` cannot say that, so folding it needs
`Each { except: Target }` or it stays a one-rule verb.
