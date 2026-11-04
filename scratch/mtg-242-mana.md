# MTG-242 — mana spending and resources

Implementation plan derived from Jack Heart's 2026-11-04 Task directive.
Base: `4463e57f5a92ebd3b60e2fb50c046b84ac2c0372`. One delivery, review only.

## Intended outcome

Offers and execution share typed spending restrictions, announced spell or
activation context, and exact resource transitions. A child payment preserves
completion of its parent. Production appears only after costs commit. Commands,
forks, observations, learning tokens, text and Academy describe the same choices.

## Delete — do not maintain

Completed cuts (no remaining predecessor path or exclusive fixture):

- `state/mana.rs::ManaUse` and executable `SPEND_SUBTYPES` masks: replace with
  typed restriction clauses and a separate spending context. Retain only the
  three frozen projection column labels in the encoder.
- Pool-payment enumeration's per-color greedy restriction assignment: retain
  deterministic ordinary spending but enumerate exact classes for witnesses.
- `PaymentSearchState.available` as one-use-per-source authority: replace with
  tap, departure, usage and pending-activation resources. Preserve identities,
  waterbend's non-mana taps, life and parent obligations.
- Pre-payment flattening of variable mana effects in characteristics: output
  evaluation belongs at the resource transition, with the public layered view
  supplying characteristics. Preserve the existing frame/decision protocol.
- Registry guard's single-source-use assumption and its exclusive assertion:
  replace with an explicit supported resource-domain contract.
- Cloned restriction-key traversal during pool payments: enumerate stable class
  indices, pruning only after the remainder is complete. Borrow restriction
  keys for lookups and projection; keep ordinary narrowest-first ordering.
- Duplicate activation reservation/obligation setup and cloned source inventories:
  share the transition constructors and immutable inventory; copy only resources.
- Temporary ability definitions and effect/string round trips for variable output:
  evaluate choices directly, with the resolving-frame consumer retaining its
  ordinary effect queue. Build activation spending context once from the final
  characteristics view instead of constructing and then overwriting it.

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

## Current implementation and remaining work

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

Validation (2026-11-04): `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2` with
focused pool/resource/v8 and Prism, Parents, Guide, Tile and Tea Shop filters
passes 34 tests; `cargo clippy --lib -- -D warnings`, Rust format and changed
Python Ruff checks pass. Rebuilt the release wheel and installed atomically;
`uv run --no-sync --with pytest pytest tests/verify/test_mana_payment_projection.py tests/model/test_mana_input.py -q`
passes 7 tests. Gate owns full affected suites and acceptance checks below.
Retained implementation evidence: `semantic_conformance record --root
conformance/semantic-kernel-w18-v1` recorded 14 cases / 3473 Commands and 15 Phase
rows, retaining w17; compression did not regenerate these receipts.

Implementation is complete. Gate owns full affected suites, current-world
admission, broader model checks, browser proof and same-runner Engine Budget
(including reference adoption if the curated decision digest moves). Publication
and review remain later Graph steps; no merge or training experiment was run.
