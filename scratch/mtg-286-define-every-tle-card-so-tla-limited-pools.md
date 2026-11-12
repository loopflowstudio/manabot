# Every TLE card in the development world

Status: proposed rules design; implementation has not begun. Review the
replacement payment model and state/observation contracts for phasing, end-turn
and player counters before resuming. The shipping doc is
`docs/engine.md#tla-decks-with-tle-bonus-sheet-cards`; decisions and evidence are
in `scratch/questions.md`.

## Outcome and scope

Define every non-token card printed in TLE using native rules types. The complete
external inventory is `scratch/tle-inventory.md`: 242 names, with 8 literal
definitions already present and 234 absent at the checkout base. The 41 unknown
names observed in the trophy corpus are an acceptance sample, not the scope.
TLE includes new cards as well as reprints; first-printing set files remain the
one definition owner. Do not duplicate the eight existing definitions.

The Task's retained baseline is 922 admitted events out of the first 1,000
events in `experiments/data/mtg213/corpus-v1/corpus.json`. The manifest pins that
file to SHA256 `22f9b2b9411a69a85ff1270605dd68fba5941601859b752d2a4a0ecf5e4a6043`,
11,034,908 bytes. This run has not independently reproduced the baseline.
Success requires all 1,000 to admit with their real main decks and sideboards,
without changing deck names, definitions, construction rules or split assignments
to make the check pass. Expanding MTG-282's saved corpus is subsequent work.

Retain the current 1v1 development format. Commander-dependent clauses must
behave correctly when no commander exists; adding a Commander format or
multiplayer setup is not part of the proposed scope. That does not permit
substituting a legendary creature for a commander or making Arcane Signet
produce arbitrary colors. Catalog tokens must still describe the printed rule.

## Why a definitions-only implementation is insufficient

The strongest counterexample is The Cabbage Merchant. Its mana ability taps
two untapped Foods, without tapping its source. Four Foods allow two uses of
the same Merchant. Current `flow/mana.rs::PaymentSearchState.available` models
one use per mana source. The existing
`registered_mana_activations_have_finite_source_costs` test requires a source
tap or once-per-turn restriction, and `ActivatedAbilityDefinition.tap_cost`
names only one predicate-matching permanent. Adding an artificial tap or
once-per-turn cost would change the card; disabling the invariant test would
leave false payment witnesses. The documented payment model must be extended
before this card is registered.

Other concrete semantic boundaries found while reading the current owners:

| Card/mechanic | Existing boundary | Required behavior |
|---|---|---|
| Volcanic Torrent / cascade | `ExileUntilNonlandCastFree` stops at the first nonland and can put it into hand | Continue past expensive nonlands, offer the first qualifying spell, and randomly bottom remaining exiled cards after the cast decision |
| Sundial of the Infinite | No end-turn instruction in `Effect` | End the turn through the stack/turn authority, including cleanup and surviving delayed triggers |
| Teferi's Protection | No phased-out state or phasing rule in the inspected rules vocabulary | Phase exact permanents and attachments without zone changes, suppress interactions, and restore them at the proper untap boundary |
| Aang, Airbending Master and the other Masters | `Player` has no counters; `PutCounters` currently targets permanents | Persistent player experience counters, correct event batching, and dynamic values visible to both seats and policy inputs |
| Black Sun's Zenith | No minus-one counter kind | Real -1/-1 counters, layered P/T, and state-based cancellation with +1/+1 counters |
| Scout's Warning / The Blue Spirit | Exact-exile grants are the current timing-permission record | Consume the applicable next/first creature timing permission only on a committed cast, regardless of whether flash was needed |
| Shattering Spree / Storm of Memories | Existing modal/copy effects do not announce repeatable cast payments or count prior casts for storm | Cast-time payment/provenance, separate cast triggers, copies and retargeting with preserved identity |

## Proposed implementation slices

Keep these as internal slices of this Task. Each introduces real card consumers
and directed tests together; no empty setup, feature flag or alternate catalog.

1. **Complete the rule allocation.** Resolve each missing card's first printing
   and printing ID, and map every printed clause to an existing or planned native
   rule. Use the existing TLA allocation fixtures as a structural precedent,
   without importing their old Task ordering. Pin the TLE name inventory in an
   offline completeness test; it only detects missing definitions and never
   controls admission. Add Oracle shells to the existing Scryfall conformance
   snapshot when definitions are added. Resolve token identities separately.

2. **Extend payment resources with The Cabbage Merchant.** Replace the assumption
   that an activation consumes its source with resource transitions for the
   actual typed costs: chosen distinct Food objects become tapped. Extend the
   existing cost/decision and payment owners, rather than a second mana planner.
   A payment witness must retain exact selected objects and prevent their reuse
   by nested payment or waterbending. A repeatable source remains available
   while it has a satisfiable cost. Prove finite progress from consumed resources;
   don't replace the invariant with an arbitrary search-depth cutoff. Test two
   activations with four Foods, only one with three, source-tapped behavior,
   competing costs, and exact clone/Command replay. The same payment authority
   must cover triggered extra production and mana whose later spending changes
   an entrant (Heartbeat of Spring, Animal Attendant).

3. **Add counter and history semantics with their cards.** Reuse sparse counter
   storage for players and permanents; extend the verb's domain rather than add
   a second experience-only effect. Add required counter kinds and cancellation
   in the state-based-action owner. Add only the turn-history facts needed by
   printed clauses, recorded at event commit. Experience abilities that say
   "one or more" consume one event batch, not one trigger per moved object.
   Test opponent actions, simultaneous moves, exact source departure and turn
   reset. Dynamic firebending must read experience at the correct time.

4. **Extend casting and resolution continuations.** Represent next-cast timing
   grants and first-cast history in the casting authority. Add alternate exile
   costs, repeatable replicate payments, storm/cascade triggers, unearth,
   cumulative upkeep, and repeatable spell modes through typed native decisions.
   Each interrupted search/cast owns its remaining exact objects and cleanup.
   Cascade must not change the existing exile-until-nonland card's meaning.
   Test declining, no qualifying card, illegal targets, replacement effects,
   countering the parent spell, and replay/fork at every suspended choice.

5. **Add turn and layered rules.** End-turn processing belongs to the existing
   turn/stack owner. Phasing is object presence, not exile: preserve incarnation,
   attachments and control without leave/enter triggers. Player life locks and
   protection are continuous rules; enforce them at life, damage, targeting and
   attachment authorities. Add remaining static costs, copy exceptions,
   legend-rule suppression, combat requirements and restrictions with the real
   cards that use them. Resolve their dependencies through characteristics and
   existing event machinery. Do not create card-specific mutable fields.

6. **Finish all definitions and integrate the world.** Author the remaining
   set-file functions against those rules. Add each function to its `cards()`
   list and each set to the catalog. Behavior tests live with first-printing
   sets; shared scenario primitives remain in `managym/tests/support`. Add
   missing printed subtype categories and token definitions. Keep the shipping
   guide's existing examples and prove them with the rebuilt extension.

The full allocation in slice 1 is still required. The table above is a verified
set of counterexamples, not a claim that all 234 missing cards have been designed.

## Shared observation and compatibility contract

Every new observable state and pending choice must be designed before its
runtime writer: native scalar/vector encoders, Python encoder, semantic capture,
LLM observation, and learning tokens must agree. Add named vocabulary entries
where `record_vocabulary.rs` owns hashed names. Preserve existing policy-input
widths, omission rules and fresh v6 default; do not silently encode novel state
as zero. Where existing extensible fact/token layouts can express a new rule,
use them and explicitly reject older unrepresentable inputs. Any genuinely new
layout needs an explicit input version and its admission/parity tests.

Continue open world w17 according to `WORLDS.md`; the catalog digest changes.
Keep frozen checkpoints, experiments and old receipts byte-for-byte. Update the
open-world description and current receipts only after the integrated rules
are validated. Do not claim that admitting a former deck makes an old policy
compatible with the changed world.

## Delete — do not maintain

The only planned cut is in `managym/src/flow/mana.rs`: replace
`PaymentSearchState.available`'s one-use-per-source model and
`tests::registered_mana_activations_have_finite_source_costs` together in slice 2.
The replacement must preserve source reservation, mana restrictions, explicit
cost choices, exact payment witnesses and termination. Keep the current guard
until that replacement exists; this documentation-only diff has no runtime
cut to make.

Keep `ExileUntilNonlandCastFree` for its distinct existing card behavior.
No catalogs, importers, frozen receipts, split assignments or card definitions
are deletion targets; there is no predecessor TLE implementation.

## Acceptance and remaining work

All slices remain open. Follow AGENTS.md's debug-test and extension-rebuild
requirements during implementation. At gate:

- Check every pinned TLE name is a defined non-token card, with unique key and
  first-printing Scryfall ID, learning tokens and directed behavior coverage.
  Run the debug Rust suite, conformance, observability, formatting and clippy.
- Run the required world-change Python suites from AGENTS.md and
  `docs/development.md`, including encoder parity/admission and checkpoint
  rejection for incompatible historical worlds. Exercise Academy pending
  choices, stale Commands, reconnect and replay/fork identity for new decisions.
- Fetch the pinned corpus through `manabot.data` with R2 credentials injected
  by Doppler; read the same first 1,000 ordered events and admit every full deck
  and sideboard through the public import path. Record the input digest and
  per-event result, expecting 1,000 passes and no dropped cards. Never regenerate
  the source corpus or alter held-out membership to satisfy this check.
- Run the shipping guide's Scout's Warning example against the rebuilt engine.
  Perform any required artifact publication before tracking an admission receipt
  under `experiments/data/`. Reuse CI's engine budget comparison rather than
  inducing local load or comparing timings across hosts.
