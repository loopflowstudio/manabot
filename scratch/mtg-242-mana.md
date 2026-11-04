# MTG-242 — mana spending and resources

Accepted outcome and constraints: Jack Heart's 2026-11-04 Task directive;
the implementation approach below is derived from it, not a separately accepted
design. One reviewable PR, stopping at review; no merge or training launch.
Original base: `4463e57f5a92ebd3b60e2fb50c046b84ac2c0372`.
Reconciled 2026-11-04 against `7e6aa4933`, including main `8e8d702f8`.

## Intended outcome

Offers and execution share typed spending restrictions, announced spell or
activation context, and exact resource transitions. A child payment preserves
completion of its parent. Production appears only after costs commit. Commands,
forks, observations, learning tokens, text and Academy describe the same choices.

## Delete — do not maintain

Remaining integration cut: remove `advance_mana_upfront`'s synchronous sacrifice
followed by in-place `ManaUpfront` replacement. Preserve committed costs, exact
source, post-cost variable output evaluation, selected taps and the original
parent through MTG-241's shared completion owner. This cut depends on its
retained zone/event implementation; a separate payment frame protocol is excluded.

Completed cuts: removed executable `ManaUse`/`SPEND_SUBTYPES` masks,
one-use source availability, greedy class assignment, pre-cost variable-output
flattening, the single-use registry guard, and the abandoned bound's
`Mana::usable_total` helper. No exclusive predecessor fixtures remain.
Frozen projection labels remain intentional. Earlier cut rationale:
`7e6aa4933691d41968940a1ae03a2837a5c6da06:scratch/mtg-242-mana.md`.

PaymentModel owns activation and selected-tap feasibility probes, sharing one
immutable inventory per enumeration. Only probe resources change; game and payer
cannot disagree with the inventory. Initial search state and activation policy
have single constructors/checks. Fixed output choices are borrowed, tap bonuses
are evaluated once per set of choices, and inventories no longer implement Clone.
This is local reuse, not a persistent cache or a performance acceptance claim.

## Scope and checks

Restrictions support subtype predicates for spell casting and activation,
without a finite subtype vocabulary. Context records source/zone separately
from purpose; resolving costs cannot use casting/activation-only mana.
Finite resources include source taps/sacrifice, once-turn/exhaust usage, life,
and strictly consuming mana filters. No arbitrary search iteration cap or
promise to solve infinite mana loops. Additional selected-object costs require
explicit shared continuation support, never a card-specific bypass.

Real-card probes cover a non-tapping once-turn filter (Scarecrow Guide),
a repeatable shared-tap source (Supportive Parents), variable production
(White Lotus Tile), and Prism/Treasure nested costs.
Oracle verification precedes new definitions. Compare final resources, not only
whether an action is offered. Preserve old projection columns; full restriction
classes require versioned native/Python parity and learning input.

## Implemented behavior

The executable mask and one-use source model are removed. Full restriction
classes, separate spending context, class-aware remainders, bounded repeatable
resources, selected tap costs, post-cost variable output, v8 projection and
Commands/fork tests are implemented. Scarecrow Guide avoids introducing the
separate Changeling mechanic just to exercise Three Tree Mascot's mana ability.
Existing White Lotus Tile supplies the variable-production composition.
Supportive Parents pays an announced spell with repeated activations, checks
each intermediate tap/output, and distinguishes cost taps from Badgermole Cub's
mana-tap bonus. Academy's native decision instruction exposes the remaining taps.

Oracle checked via Scryfall on 2026-11-04:
- [Supportive Parents](https://scryfall.com/card/spm/119/supportive-parents),
  printing id `d3fe8a5b-4166-46cc-b910-71cd1a19ae1b`: two shared creature taps.
- [Scarecrow Guide](https://scryfall.com/card/woe/250/scarecrow-guide),
  printing id `3dc08461-bec6-4a18-b782-da4c92abb789`: one mana, once per turn.
Both API records report Standard legal. Their complete shell fixtures are local.

Supported costs are battlefield source tap/sacrifice, life, once-turn/exhaust,
selected distinct untapped objects, and strictly consuming mana filters. Arbitrary
infinite loops and selected sacrifice/discard/crew mana costs remain outside
this resource domain; the registry contract names these limits explicitly.
No persistent inventory cache was introduced. Reuse stays within enumeration.
Conservative resource bounds are separate from exact witness transitions; both
bounds share the supported activation-use calculation. V8 fact kinds and its
tap-count column have named constants on the Python projection/model boundary.

Retained implementation evidence covers 34 focused debug Rust tests and seven
projection/model tests, library Clippy, formatting and changed Python lint.
The w18 conformance corpus retains 14 cases / 3473 Commands and 15 Phase rows;
w17 remains unchanged. These bounded results do not establish full-suite or
post-integration acceptance; their prior record is at
`7e6aa4933:scratch/mtg-242-mana.md`.

## Integration findings and remaining work — 2026-11-04

Main's selected-cost X repair, configurable native buffer axes and explicit
first-legal action policy are integrated. V8 keeps relation-width selection,
caller-supplied capacities and original action indices. Reconciliation found
two existing overflow tests failed because the new legacy-mana guard inspected
player state before the public action-capacity check. The encoder now validates
action capacity first; the fixtures and overflow policy remain unchanged.
The TLA pilot's frozen protocols and training evidence are not v8 evidence.

MTG-241's 2026-11-04 completion counterexample remains unresolved:
`advance_mana_upfront` synchronously sacrifices, then assumes `ManaUpfront`
survived. `handle_leaves_battlefield` can return linked exiled cards. A return
requiring replacement ordering must suspend before payment exposes output.
This checkout has neither `FrameCompletion` nor a replacement-order Decision.
The dependency is implementation, not a missing product judgment.

This pass adds
`rules::cr_106_mana::mana_sacrifice_linked_return_preserves_parent_payment_through_commands`:
a constructed Treasure pays life and sacrifices to fund Prism, returning an
opponent-owned card through a single entry-counter replacement. Both seats and
all five colors check uncredited output, exact life/mana, one sacrifice, stable
returned incarnation and parent completion. Every Command checks fork isolation,
replay agreement and stale rejection through the shared test helper. This is
passing synchronous-path evidence, **not** reproduction of replacement suspension.

Read-only inspection of MTG-241's checkpoint `e6b268956` and active plan confirms
its common `FrameCompletion` and `run_before_decision` interface exists, but
zone/event suspension and replacement ordering remain unimplemented. Task status
still reports an active implement process; its coordination comment explicitly
requests no integration/cherry-pick while live. No work was copied or modified
there. Dependent runtime changes stop here until the shared event cut is available.
Integration must retain MTG-242's post-cost `mana_outputs` evaluation (MTG-241's
current payment caller still uses fixed `ability.mana_choices()`), tap obligations,
v8 and frozen schemas. Then the actual suspended-return Command/fork witness must
prove single cost commitment, no early output and exactly one parent resumption.
Payment resources/search remain MTG-242's; completion/events remain MTG-241's.

Gate acceptance still includes the complete debug Rust suite and all-targets,
all-features Clippy; current-world arena admission/monitoring; broader model and
active-training checks; Academy choice-navigation/Learn setup and browser proof;
post-sync conformance replay; and same-runner Engine Budget, including reference
adoption if the curated decision digest moves. Browser evidence unavailable
headless remains with gate/CI or demo. Review must resolve the MTG-241 integration
boundary before claiming completion; no new product decision is otherwise needed.
Publication and review remain later Graph steps.

Check (2026-11-04): `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2` with engine `--lib flow::mana::tests`, state `--lib mana::tests`, cards `--test cards -- roe:: spm:: woe:: tla::white_lotus tla::waterbend_and_mana_taps_have_distinct_bonus_accounting`, and engine `--test rules_tests rules::cr_106_mana` — 40 passed; `uv run --extra dev pytest tests/verify/test_mana_payment_projection.py tests/model/test_mana_input.py -q` — 7 passed after release wheel build and atomic extension installation; formatting/diff checks passed. Reused validation-order repair evidence: observation/cutoff/capacity/mana-projection selection, 123 passed. Full acceptance belongs to gate after shared completion integration.
