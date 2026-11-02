# MTG-235: Effect verbs over an ObjectSelector (draft, 2026-11-02)

Directive: Task MTG-235 (Jack Heart's concept review of 2026-11-02). Base is
main `4c8324a59`; MTG-225 (T6) has not published, so there is no
`TriggerCondition::Chapter` here to rename (see questions.md).

## Outcome

`Effect` verbs name what they act on with one `ObjectSelector`, resolved by one
`Game::select` and acted on by one arm per verb. The object-fused variants go.

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

Compositions (fused verbs that depended on which object they hit):
- ReturnToHandDrawIfControlled → `[Move{Target,Hand}, If{Moved, [If{TargetControlledByYou, [Draw 1]}]}]`
- ExileAndReturn / ExileSourceAndReturn → `[Move{X,Exile}, Move{Moved,Battlefield,under}]`
- ExileWithControllerToken → `[Move{Target,Exile}, ForPlayer{TargetController,[CreateToken]}]`
- MoveSourceFromGraveyard{then} / SacrificeSource{then} / MaySacrifice{then} → verb + `if_you_do(then)`

## Delete — do not maintain

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
`OptionalSacrifice.then`. Their resolution arms, description arms,
observability rows/witnesses and heuristic names go with them; behavior is
preserved by the verbs above and proven by the unchanged card tests.

## Acceptance

- debug `cargo test` in `managym/`: every directed card test unchanged.
- `tests/cards/observability.rs`: every new variant and every `ObjectSelector`
  named and witnessed in the tokens.
- Learning tokens and content digest change; w17 evidence re-recorded in place
  (conformance/README.md); WORLDS.md records it; no version bump.
- `manabot/sim/heuristic.py` reads the new shapes; `tests/sim/test_heuristic.py`
  polarity table unchanged.

## Check-result

2026-11-02 — debug `cargo test --tests` (every managym target): all pass, including
every directed card test unchanged (485) and the re-recorded semantic
conformance evidence; `cargo clippy --all-targets -D warnings` clean; Python
content-bound suites (heuristic, learning projection, current content, text
observation, observation parity, belief receipts, Academy choice navigation,
trigger order, learn setup, semantic cards, training decks): 150 pass. Fused
object-in-name verbs: 33 removed, 17 remain (listed in `questions.md`);
`Effect` has 79 variants. Learning tokens and the content digest moved; w17
evidence re-recorded in place, no version bump.

2026-11-02 compress — `Sacrifice { optional }` is read only with a `Chosen`
selector (the unreachable "you may sacrifice [it]" payment path and its
decline hook are gone); the resolution arm clears `frame.moved` before
offering the choice, so declining needs no special case; `sacrifice_permanent`
returns the object the card became, shared by the arm and the decision.
Debug `cargo test`: `cards` 485, `--lib` 132, conformance/replay/authored
targets all pass; clippy and fmt clean. The extension `.so` is untracked and
its rebuild (behavior unchanged for every card) is gate's.
