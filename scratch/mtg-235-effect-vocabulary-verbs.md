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
- Not in scope here: the 17 remaining one-rule verbs are the next candidates
  if a later card shares their shape.

## Check-result

2026-11-02 realign — the extension rebuilt from this head (`uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`) differed from the one the implement pass tested; against the rebuilt `.so`, `uv run --extra dev pytest tests/sim/test_heuristic.py tests/semantic/test_learning_projection.py tests/semantic/test_learning_projection_env.py tests/semantic/test_current_content.py tests/semantic/test_text_observation.py tests/semantic/test_static_ability_tokens.py tests/verify/test_observation_parity.py tests/belief/test_state.py tests/academy/test_choice_navigation.py tests/academy/test_trigger_order.py tests/academy/test_learn_setup.py -q`: 152 passed. Rust: debug `cargo test` (cards 485, lib 132, conformance/replay/authored), clippy and fmt clean at the compress pass; no Rust source changed since.
