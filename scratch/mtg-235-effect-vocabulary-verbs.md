# MTG-235: Effect verbs over an ObjectSelector

Accepted direction: Jack Heart, 2026-11-02, including the concept-review
answers “1a 2a 3a 4b” and “after that, ship it. this looks good”. One cohesive
amendment to PR #412. The original pass and review rationale are preserved at
`d1bda612103909ead5321163d889818911e8a36e:scratch/mtg-235-effect-vocabulary-verbs.md`;
`scratch/concept-review.html` is the historical review page.

Status: recovery implementation complete. Existing edits from the interrupted
Claude run are preserved. Publication and subsequent delivery remain the selected
Graph's steps; green CI on the old PR head does not validate this amendment.

## Outcome and authority

A card author writes one verb and names its objects with `ObjectSelector`.
Learning tokens derive from those same definitions; the Academy and LLM read
native descriptions of the verbs and selectors. No authoring IR or extra
vocabulary store is introduced.

`ObjectSelector` has `This`, `Target(TargetSpec)`, `Triggering`, `Attached`,
`CreatedThisWay`, `Grantor`, `Moved`, `You`, `Chosen { predicate, except }`,
and `Each { of: Scope, predicate, except, max_mana_value }`. `Scope` names
`YouControl`, `OpponentsControl`, `Battlefield`, or `TargetPlayerControls`;
`Except::{This, Target}` distinguishes “another” from Avatar's Wrath's
“all other”. `max_mana_value` is evaluated at resolution (Day of Black Sun).

`Game::select` is the object resolver, with last known information for a
source that leaves. Player-affecting verbs use the same selector through
`resolve_affected`; a stack ability has no card `ObjectRef`, so CounterAbility
reads the clause's retained target directly. Static layer computation privately
resolves the same selector without recursively reading characteristics.

`Chosen` suspends any supported object verb on `Decision::ChooseObject`.
The frame binds the chosen permanent for that instruction, then clears it,
including when the player stops a nested counter-removal choice.
Optional sacrifice can decline. A move/sacrifice with no candidate clears its
move result, so a later “if you do” cannot read an earlier instruction's success.
`Earthbend` applies to every selected land, including `Each`.

The original counter, sacrifice, move, tap and untap families share selectors.
The amendment also migrates Destroy, Airbend, GainControl, DealDamage,
CounterSpell, CounterAbility, Discard and Earthbend. `ApplyContinuous.affected`
is an `ObjectSelector`; `AffectedSpec` is removed. Discard uses `what`, including
`You` and a targeted player, as accepted in decision 1a.

Amounts use `EffectValue`: fixed and computed damage, life, token counts and
mana production share one variant each. Card definitions use the existing
constructors for ordinary tokens, fixed damage, destruction, selector defaults
and mana modes; only non-default terms need literals. `Effect::mana_choice`
owns modal mana construction, including the restricted-color helpers in TLA.
The heuristic shares target/spec parsing and tests continuous effects through
the actual engine shape.

“If you do” is `Condition::Moved`, reading `EffectFrame.moved`, the latest
Move/Sacrifice result. `this_way` remains the latest discard/mill (Avatar
Destiny mills, moves itself, then reads the milled cards). Flicker is two Moves,
the second over `Moved`. Moves to battlefield default to the resolving player's
control; `PlayerSelector::Owner` is explicit for returns. Exile links use `linked`;
library moves shuffle, with positional placement remaining a distinct rule.
Day of Black Sun composes ability loss and simultaneous Destroy; trigger sources
are captured before the destruction batch.

## Delete — do not maintain

Removed in the original cut: `Effect::{PutCountersOnTriggeringPermanent,
PutCountersOnCreatedToken, PutCountersOnSource, PutCountersValue,
PutCountersOnEachMatching, PutTriggeringCountersOnSource,
PutSourceCountersOnTarget, MaySacrifice, SacrificeSource,
SacrificeTriggeringPermanent, TapSource, UntapSource, TapAttached, TapTarget,
UntapTarget, ReturnGraveyardTarget, ReturnToHandDrawIfControlled, ReturnToHand,
ReturnGrantorToHand, ReturnSourceToHand, ReturnSourceToBattlefield,
ReturnSourceToBattlefieldTapped, Exile, ExileSourceAndReturn, ExileAndReturn,
ExileAttached, ExileWithSource, ExileUntilSourceLeaves, ExileSourceSpell,
ExileWithControllerToken, MoveSourceFromGraveyard, ShuffleAttachedIntoLibrary,
RemoveAllCountersFromAttached}`, `GraveyardReturn`, `OptionalSacrifice.then`,
and their exclusive resolution/description/observability/heuristic paths.

Amendment removes `AffectedSpec`, `Decision::Sacrifice`,
`Effect::{MassDamage, DamageOpponentCreatures, AirbendAll,
DestroyCreaturesLosingAbilities, DealDamageValue, GainLifeValue,
CreateTokensValue, AddManaValue, DiscardTargetPlayer}` and the bare Discard
shape, plus `target: TargetSpec` on the accepted migrated verbs. Preserve
printed card behavior, target legality, simultaneous destruction, resolution
choices, descriptions, tokens and directed card tests on the surviving paths.
No legacy schema adapter or parallel effect dispatcher remains. The duplicate
`cardsets::tla::add_one_of` helper is removed; its callers use
`Effect::mana_choice`. No additional deletion targets remain in this cut.

Effect now has 70 variants, down from 107. Remaining object-in-name rules (14):
ReturnMilledThisWay, DistributeCountersToTappedOpponentCreatures,
EachPlayerKeepsCreatures, ExileTopCardPlayThisTurn, ExileUntilNonlandCastFree,
ExileLinked, CastExiledWithSource, AttachSourceTo, OwnerPutsIntoLibrary,
PlaceTargetInLibrary, PutTopCardsInHand, CopyTriggeringObject,
TargetCreaturesDealPowerDamageToLastTarget, FightLastTarget. The fused set
shrinks by more than half; these remaining rules are outside the accepted cut.

## Acceptance and remaining work

- Directed card tests pass unchanged in debug; observability guards cover
  verbs/selectors and learning tokens. New choice behavior has directed rules
  tests. Native/Python observation parity and text consumers remain covered.
- Refreshed semantic conformance, authored-match receipt, both parity corpora,
  five Academy fixtures and current belief receipt pins against the rebuilt
  extension. Historical receipts are unchanged. w17 and policy input v5 stay
  open in place; WORLDS.md records changed content/tokens.
- Gate owns the affected suites and final automated acceptance. Complete the
  selected Graph through amended publication of #412 and its delivery boundary.
- Explicit integration dependency: MTG-225 (T6) has no published PR on this
  base. `TriggerCondition::Chapter` does not exist here; integration must replace
  it with `CountersReach { kind, at_least }` when T6 publishes, and migrate any
  new T6 effects. This amendment alone does not complete that dependency.

## Check-result

2026-11-02 — `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --test cards
--test rules_tests selector_`: 5 passed; targeted court-official/Ozai discard:
1 passed; `uv run --extra dev pytest tests/sim/test_heuristic.py -q`: 27 passed;
all-target/all-feature clippy, fmt, Ruff and diff checks clean. Release cp312
wheel rebuilt and installed atomically; every serialized definition and the
complete learning catalog match the saved pre-compression bytes exactly.
Authored live/headless/replay parity is regenerated against the simplified
source (142 commands, no divergence). Context remains within both budgets.
Gate owns the full affected suites, observation/world admission and final
conformance reproduction; publication remains the Graph's delivery step.

Retained recovery evidence: semantic conformance recording covered 14 cases /
3396 commands with unchanged counts/outcomes; public parity covered 69
commitments with zero gaps; 90 focused Python and 2 decision-label Vitest tests
passed. The interrupted run's 485-card debug pass predates the recovery fixes.
The complete recovery log is Codex thread
`01a19e71-fbe0-7220-a678-788b730adb09` (Loopflow run
`90147cf1e73340099716c86f29261341`); it preserves rebuild hashes and commands.
