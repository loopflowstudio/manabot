# MTG-235: Effect verbs over an ObjectSelector

Accepted direction: Jack Heart, 2026-11-02, including the concept-review
answers “1a 2a 3a 4b” and “after that, ship it. this looks good”. One cohesive
amendment to PR #412. The original pass and review rationale are preserved at
`d1bda612103909ead5321163d889818911e8a36e:scratch/mtg-235-effect-vocabulary-verbs.md`;
The historical review page is retained at that commit; `scratch/concept-review.html`
shows the current headless review.

Status: the combined T5 source is verified for republication. Main `a8abf01b1`
is integrated by `e5560fb22`; publication validation corrected one test assertion
to call `Characteristics::is_creature()`. The latest results are in “Combined
source publication verification” below; earlier counts describe earlier heads.
Subsequent delivery and MTG-225 integration remain pending.

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

The amendment reduced Effect from 107 to 70 variants; T5's Surveil brings the
combined source to 71. Remaining object-in-name rules (14):
ReturnMilledThisWay, DistributeCountersToTappedOpponentCreatures,
EachPlayerKeepsCreatures, ExileTopCardPlayThisTurn, ExileUntilNonlandCastFree,
ExileLinked, CastExiledWithSource, AttachSourceTo, OwnerPutsIntoLibrary,
PlaceTargetInLibrary, PutTopCardsInHand, CopyTriggeringObject,
TargetCreaturesDealPowerDamageToLastTarget, FightLastTarget. The fused set
shrinks by more than half; these remaining rules are outside the accepted cut.

## Acceptance and remaining work

- Final-source acceptance passed: debug Rust suite 862 passed / 2 ignored,
  including all 485 directed card tests unchanged, selector/token guards,
  choice-lifetime tests and semantic conformance reproduction. Python consumers,
  native/Python observation parity, LLM rendering, world admission, monitoring,
  model and disclosure suites: 411 passed / 1 skipped; the sync resolver
  separately verified the two regenerated parity corpora with 7 passing tests.
- Refreshed semantic conformance, authored-match receipt, both parity corpora,
  five Academy fixtures and current belief receipt pins against the rebuilt
  extension. Historical receipts are unchanged. w17 and policy input v5 stay
  open in place; WORLDS.md records changed content/tokens.
- Academy type checking and build passed; decision-label/decklist Vitest 8 passed;
  headless choice-navigation, Learn and trigger-order recovery 10 passed.
  Final-source clippy (all targets/features) and fmt pass; retained Ruff passes.
  The installed cp312 extension matches the rebuilt wheel byte for byte.
  Publication logs remain local under `.runs/publish-mtg-235/`. Delivery remains
  separate from publication; old-head CI is not used as amendment proof.
- Explicit integration dependency: MTG-225 (T6) has no published PR on this
  base. `TriggerCondition::Chapter` does not exist here; integration must replace
  it with `CountersReach { kind, at_least }` when T6 publishes, and migrate any
  new T6 effects. This amendment alone does not complete that dependency.

## Reconciliation and retained evidence — 2026-11-02

The stopped implementation log ends after rebuilding, before evidence refresh.
Recovery completed the missed Discard selector migration, Each Earthbend and
choice-lifetime fixes. Compression removed repeated constructors and modal-mana
building without changing serialized definitions or the complete learning
catalog. The release cp312 extension was rebuilt and installed atomically.
Focused debug selector tests (5), targeted discard (1), heuristic tests (27),
all-target/all-feature clippy, fmt and Ruff passed in those steps.

The only newly integrated upstream commit, `29660d00`, names Engine, Training
and Eval in the architecture documentation; it changes no selector behavior.
Current MTG-225 status still reports no PR. Its accepted catalog simplification
retains stable definition rows and content identity; integration must preserve
those contracts alongside the T6 effect migration. No new product decision is
needed for this amendment. Engine has no child memory directories; its stale
fused-verb count and incomplete amendment guidance are reconciled here.

Retained recovery evidence: semantic conformance recording covered 14 cases /
3396 commands with unchanged counts/outcomes; public parity covered 69
commitments with zero gaps; 90 focused Python and 2 decision-label Vitest tests
passed. Compression regenerated authored live/headless/replay parity against
its final source (142 commands, no divergence). The complete recovery log is Codex thread
`01a19e71-fbe0-7220-a678-788b730adb09` (Loopflow run
`90147cf1e73340099716c86f29261341`); it preserves rebuild hashes and commands.
Compression evidence is retained in session `session_068a02853c1f42b599b854fb7f585e0a`.

Check-result: 2026-11-02 — final-source acceptance at `059d37b01`: Rust 862 passed / 2 ignored; Python 411 passed / 1 skipped plus 7 parity tests; browser 10 passed; UI checks/build and 8 label/decklist tests passed; `git diff --check` clean. MTG-225 integration remains pending.

## Concept review after recovery — 2026-11-02

Outcome: no design change and no open product choice. Jack Heart's accepted
1a / 2a / 3a / 4b decisions remain the contract for this amendment. The current
types and resolution path express one object vocabulary, one amount vocabulary
and a shared resolution-time choice. Existing authoring guidance in AGENTS.md
already describes that experience; no further usage or skill changes are needed.

Concrete interaction: an author writes `Effect::tap(ObjectSelector::chosen(
CardPredicate::creature()))`. The engine retains the instruction and frame in
`Decision::ChooseObject`; the player selects a legal permanent through the
ordinary action surface, and resolution applies the verb to that exact object.
An illegal selection restores the pending decision. A completed instruction
clears its chosen binding, including after stopping nested counter removal.
An impossible chosen Move/Sacrifice clears `moved` so a following `if_you_do`
does not reuse earlier success. These are execution/recovery obligations,
not reasons to introduce a second selection concept in Academy.

Retain the counterexamples that constrain simplification: Avatar's Wrath needs
`Except::Target`, not `Except::This`; Avatar Destiny needs distinct `this_way`
and `moved` results; Day of Black Sun must preserve ability loss followed by
simultaneous destruction with the correct trigger sources. Player and stack
ability selection retain their typed execution adapters because neither is a
card ObjectRef. A shared authoring vocabulary does not erase those identities.

Definitions remain the sole owner of rules and learning tokens. Academy and LLM
labels consume the native resolution choice; no consumer gains its own rules or
choice state. Changed content identity requires current w17 evidence while
historical receipts and frozen observation widths stay intact. At this review boundary, final-head acceptance was still pending; the publication
checks recorded above now establish it. This concept review itself added no
test-pass or delivery claim; MTG-225 integration remains pending.

## Publication sync — 2026-11-02

Main advanced to `230b2f563` (decklist disclosure) during publication. Loopflow
sync resolved six evidence/note conflicts and merged it as `059d37b01`; selector
code merged cleanly. The rebuilt extension and regenerated receipts retain the
142-command authored tape, all five choice prefixes/offers and 69 commitments
with zero parity gaps. Acceptance was rerun on this combined source as recorded
above. Clippy initially failed to resolve the relative Python interpreter; the
absolute repository interpreter passed. Publication remains separate from merge.

## T5 publication sync — 2026-11-02

The existing sync onto main `a8abf01b1` integrates T5 copy effects, surveil,
layered card characteristics and centralized continuation codes. Retain the
selector amendment: incoming Joo Dee sacrifices through `ObjectSelector::chosen`,
Ran and Shaw uses `ObjectSelector::This`, and `ChooseObject` retains decision
code 4 while Surveil adds code 22. `CreateTokenCopy` replaces the old specialized
copy effect; Surveil raises the combined vocabulary from 70 to 71 variants.

Current w17 fixtures and parity receipts are regenerated against the combined
source. The authored 142-command tape and terminal state are identical; all
navigation/demo prefix offer IDs and labels are retained. Main's layered P/T
projection changes viewer hashes. Native extension rebuilt and replaced
atomically; broader gate results above describe the pre-T5 head, not this sync.
The sync caller owns final verification and publication; this resolver does
not push or land.

## Combined source publication verification — 2026-11-02

`lf sync` confirmed the branch is current with main `a8abf01b1`. Review of
the actual base diff and merge resolution preserves the accepted selector,
EffectValue and Chosen model alongside T5 token copies and Surveil. The full
debug build found an integration error missed by the focused sync checks:
the Each Earthbend test read `is_creature` as a field after main made it a
method. Corrected only that assertion; no native implementation changed.
WORLDS.md now distinguishes the 70-variant amendment from the combined 71.

Final combined-source checks (logs under `.runs/republish-mtg-235/`):

- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml`:
  881 passed, 2 ignored, including all 502 card tests, selector/observability
  guards, continuation contracts and semantic conformance reproduction.
- Python consumers, native/Python observation parity, LLM text, world admission,
  arena/monitoring, all model tests, disclosure, Academy fixture and both parity
  corpora, and belief receipts: 404 passed, 1 skipped. Authored parity retains
  142 commands; public parity retains 69 commitments and zero provider gaps.
- Academy type check and production build passed; decision-label and permanent
  notes Vitest: 9 passed. Headless Playwright choice-navigation, Learn and
  trigger-order recovery: 10 passed; its servers stopped with the test command.
- All-target/all-feature clippy, rustfmt, focused Ruff and diff whitespace checks
  passed. Installed cp312 extension matches the sync's rebuilt wheel byte for
  byte (SHA256 `81ad7d04358488793c555d565c7a39c6894ac5536bdf6aac8a3f3770fc2f86c8`).

Fresh `lf task status MTG-225 --json` still reports no PR (implementation is
running). T6 integration and delivery remain explicit dependencies. Publication
does not merge, arm auto-merge or complete this Task. These results validate the
combined local source; previous-head CI is not evidence for this amendment.
