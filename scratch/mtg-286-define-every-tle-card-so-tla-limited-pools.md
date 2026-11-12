# Every TLE card in the development world

Status: implementation in progress. The 242-name/411-paragraph rule allocation
is complete as a planning ledger, not a behavior proof. Merchant tap choices now
have scalar, vector, Academy reconnect and Command/fork proofs. 52 inventory
names are defined (ordinary definitions, -1/-1 counters with CR 704.5q
annihilation for Black Sun's Zenith, and the next-creature flash record for
Scout's Warning); 190 remain absent. The
production/spending replacement is proposed in `scratch/tle-production-design.md`;
its event timing and exact v10 encoding table require authored review before
dependent writers. Phasing, end-turn and player counters also need detailed
state/observation contracts.
The shipping doc is
`docs/engine.md#tla-decks-with-tle-bonus-sheet-cards`; decisions and evidence are
in `scratch/questions.md`.

## Outcome and scope

Jack Heart: "lets file a task for covering all of TLE".

Define every non-token card printed in TLE using native rules types. The complete
external inventory is `managym/cards/tests/fixtures/tle-inventory.jsonl`: 242
names. `scratch/tle-inventory.md` retains source and baseline details (8 defined,
234 absent). The 41 unknown names observed in the trophy corpus are an acceptance sample, not the scope.
TLE includes new cards as well as reprints; first-printing set files remain the
one definition owner. Do not duplicate the eight existing definitions.

The Task's retained baseline is 922 admitted events out of the first 1,000
events in `experiments/data/mtg213/corpus-v1/corpus.json`. The manifest pins that
file to SHA256 `22f9b2b9411a69a85ff1270605dd68fba5941601859b752d2a4a0ecf5e4a6043`,
11,034,908 bytes. This run has not independently reproduced the baseline.
Success requires all 1,000 to admit with their real main decks and sideboards,
without substituting deck names, weakening printed card behavior or construction
rules, or changing split assignments to make the check pass. Expanding MTG-282's
saved corpus is subsequent work.

Retained scope assumption: use the current 1v1 development format.
Commander-dependent clauses must
behave correctly when no commander exists; adding a Commander format or
multiplayer setup remains a demo question, not an implementation prerequisite.
That does not permit
substituting a legendary creature for a commander or making Arcane Signet
produce arbitrary colors. Catalog tokens must still describe the printed rule.

The product choice left for demo is whether correct no-commander behavior in
the existing 1v1 world fulfills this Task, or whether playable Commander setup
and multiplayer are also required. No quoted decision settles that expansion;
the proposal keeps 1v1 while retaining all 242 names and their printed clauses.

## What exists

`managym/cards/src/tle.rs::the_cabbage_merchant` defines the counted tap cost and
both printed triggers. `flow/mana.rs::PaymentResources` separates presence,
untapped resources and limited ability uses; `PaymentStep::Tap` records each
selected Food. `PaymentModel::plan_after_tap` shares reservation and remaining-count
logic between ordinary and nested payments. Count-one `TapCost` definitions retain
their serialization and tokens.

Recorded focused results in `scratch/questions.md` establish these bounded proofs:

- Seven tests in `managym/cards/tests/cards/tle.rs` cover two activations using
  four distinct Foods, three-Food rejection, a tapped/summoning-sick Merchant,
  reserved Food costs, waterbend exclusion, viewer-safe continuation and both
  printed triggers. `support/inspect.rs::assert_choice_contract` compares cloned
  Command execution and rejects a repeated stale Command without mutation.
- `observability.rs::merchant_counted_tap_cost_reaches_learning_tokens` checks
  the count and Food predicate. The registered-mana resource guard and focused
  existing single-tap/nested-payment regressions passed in debug.
- `tests/verify/test_observation_parity.py::test_merchant_tap_count_native_reference_and_text_parity`
  passed for v5/v6/v8/v9 at both tap selections: native allocated and preallocated
  **scalar** buffers agree with Python and LLM text. The extension was rebuilt
  after the final payment edits. This test does not execute `VectorEnv` or Academy.

`tests/academy/test_merchant_payment_recovery.py` executes a real three-Food
position with forced skipping on. It recovers both tap prompts and the color
prompt through WebSocket reconnect; fork execution and replay into a native
VectorEnv agree at every prompt for v5/v6/v8/v9. Duplicate and stale Commands
leave state and trace length unchanged. Two Foods would make the second tap
forced, so that setup cannot prove a second user choice.

The full affected suites, current-world receipts and browser visual checks
remain outstanding. Focused passes do not establish
complete payment correctness or a general large-board performance bound.

`managym/cards/tests/fixtures/tle-inventory.jsonl` pins all 242 Oracle identities,
TLE printing IDs, first-printing IDs/sets and every Oracle paragraph. It retains
existing definition identities, chooses KLD over its simultaneous MPS Gearhulk
printing, and records Three Dreams' earlier PSAL printing. This is source data,
not a support list. `tle-allocation.jsonl` binds an implementation obligation to
every one of its 411 Oracle paragraphs, retaining all 242 definition owners,
including the three vanilla cards. 52 names are now defined; 190 remain
absent (evidence in `scratch/questions.md`, fourth pass). Eight additions use existing rules: Capital Guard, Kyoshi Warrior Guard,
Warship Scout, Explosive Shot, Zuko's Offense, Kindly Customer, Koala-Sheep and
Wolf Cove Villager. Their six `tle_baseline` tests passed in debug; their printing
shells/Oracle text are in the existing conformance snapshot. Sheep is now a
creature subtype in the shared table. No corpus re-admission receipt exists. The shipping guide's
Scout's Warning example correctly remains planned behavior.

## Why a definitions-only implementation is insufficient

The Cabbage Merchant disproved the old one-use-per-source payment model:
four Foods fund two activations without tapping the Merchant. The implemented
`flow/mana.rs::PaymentResources` tracks actual consumed resources, and
`PaymentStep::Tap` retains selected objects. Both optimistic bounds account
for repeated activations. The transition contract is below; production semantics
for Heartbeat and Animal Attendant remain a separate design boundary.

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

1. **Rule allocation (planning ledger complete).** The new
   `managym/cards/tests/fixtures/tle-allocation.jsonl` assigns each pinned Oracle
   paragraph to existing or planned native rules. Indices bind to the unchanged
   source fixture; modal bullets retain their announcement owner. The TLA
   allocation supplied the structural precedent, without its old Task ordering.
   `tests/infra/test_tle_allocation.py` protects pinned source and clause binding.
   Add Oracle shells to the existing Scryfall conformance
   snapshot (`managym/tests/fixtures/scryfall_cards.json`) when definitions are
   added. Resolve token identities separately.
   `uv run python scripts/check_tle_coverage.py --allocation-only` checks
   the pinned source and exact paragraph coverage. Without that option it requires
   all names in the installed catalog, first-printing identities, unique keys
   and learning tokens; it must fail while definitions remain missing.
   Neither check establishes printed behavior. Token closure identities and
   the mechanism families below still require implementation design.

2. **Payment resources and production.** The Merchant tap-resource subset is
   implemented under the contract below, with focused evidence in
   `scratch/questions.md`. The shared vector/Academy proofs now pass. Design
   triggered extra production and spend provenance before registering Heartbeat
   of Spring or Animal Attendant; their counterexample is below.

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
   guide's existing examples and prove them with the rebuilt extension, then
   replace its planned-support wording with the demonstrated behavior.

Slice 1 allocates all clauses but does not certify the planned extensions.
The ledger adds concrete production/spending consumers (Toph, Greatest Earthbender,
Uncle's Musings, Tarnished Citadel and The Great Henge), plus state-trigger,
entry-copy, chosen-color and event-batch obligations that the earlier family list
omitted. Review these before defining cards against incomplete rules.

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

## Merchant payment transition design

Use one payment resource state per battlefield object: whether its untapped
resource and presence remain, plus per-ability limited-use marks. Fixed mana
and waterbending consume the untapped resource; sacrifice consumes presence;
limited activations consume their mark. Counted tap costs consume distinct
matching objects' untapped resources, not the source's identity. Every admitted
mana activation must consume at least one such finite resource before nesting
its upfront mana obligation. No arbitrary recursion limit is permitted.

The witness records Activate, each selected tap-cost card, upfront spending,
then output choice. Pending mana activation retains an unpaid tap count and
continues through ordinary SelectCard Commands before mana payment/output.
Each selected tap commits visibly; the source remains usable if it has no
source-tap/limited-use cost. Candidate offers require a witness for the complete
remaining payment, including saved callers. Ordinary nonmana TapPayment uses
the same counted predicate and distinct untapped candidates. Count-one serialized
definitions must retain their existing shape/tokens; counted definitions add
an explicit count. Both pruning bounds sum optimistic remaining productions
based on finite costs; they must never budget one output per repeatable source.

Projection: unpaid tap count is the continuation's existing mode value, source
and ability use existing reference slots, committed taps use permanent tapped
columns. Native and Python encoders already carry mode; text must name the
remaining count and action labels must say Tap. Counted cost learning tokens
come from the definition's serializer. No new tensor width or fresh default.
The focused native proof covers clone/Command replay at each tap and output,
competing waterbend/activation resources, three versus four Foods, a tapped
Merchant and both printed triggers. The integration proof executes both tap selections through vector buffers and
Academy's actual pending decision, including stale rejection, reconnect and
replay/fork identity. Browser presentation remains with gate/demo.

## Production boundary requiring design

Heartbeat of Spring cannot be represented by the existing triggered-mana
record. `state/card.rs::TriggeredManaAbility` holds a fixed `Mana`;
`flow/zones.rs::tap_bonus_sources`
scans only the tapping player's permanents, and `tap_permanent(for_mana=true)`
fires bonuses before a chosen output exists. `Decision::ManaChoice` retains
choices and a caller but discards the source. Heartbeat applies to either player
and chooses a type that the land actually produced. Treating it as a fixed
bonus, computing its color from printed abilities, or granting it only to its
controller is incorrect. Animal Attendant also needs mana-unit provenance that
survives spending into cast/entry; `Mana` currently retains color/restriction,
not this entry-effect provenance (`state/mana.rs::Mana`). These are production/observation contract
changes, not more definitions over the completed tap-resource slice.

The proposed replacement contract is now in `scratch/tle-production-design.md`.
It specifies production occurrences, actual-output triggers, semantically distinct
mana classes, explicit spending choices and receipts attached to the cast/entry.
It names the existing consumers that move in each deletion cut. Toph's total
spending, Uncle's Musings' distinct colors, Citadel's damage and Henge's life gain
are additional constraints from the clause allocation. Production-event timing
and the concrete v10 fact/action join table remain explicit review questions;
no new writer or input version has been added. Continue only definitions that
use existing rules until the corresponding new contract is settled.

## Delete — do not maintain

Completed: replaced `PaymentSearchState.available`, both one-use pruning
bounds and `registered_mana_activations_have_finite_source_costs`. The surviving
resource model and explicit tap witnesses retain reservations and finite
progress. Single-object tap costs use the counted type while serializing identically.
`PaymentModel::plan_after_tap` also replaced duplicated selected-tap
reservation/count logic. No parallel mana planner remains. The proposed
production/spending cut names its additional deletion targets in
`scratch/tle-production-design.md`; do not extend those predecessors.

Keep `ExileUntilNonlandCastFree` for its distinct existing card behavior.
No catalogs, importers, frozen receipts, split assignments or card definitions
are deletion targets; there is no predecessor TLE implementation.

## Acceptance and remaining work

The Merchant subset of slice 2 and eight ordinary definitions are implemented.
Remaining: review the proposed production/spending contract and settle its exact
observation grammar; implement the remaining slices (player experience counters,
cascade, end-turn, phasing and the rest of the allocation) and all 190 missing
definitions. The
Merchant does not establish Heartbeat/Attendant support or corpus admission.
The all-card outcome is not met. At gate, after implementation and extension
rebuild:

- Check every pinned TLE name is a defined non-token card, with unique key and
  first-printing Scryfall ID, learning tokens and directed behavior coverage.
  Run the debug Rust suite, conformance, observability, formatting and clippy.
- Run the required world-change Python suites from AGENTS.md and
  `docs/development.md`, including encoder parity/admission and checkpoint
  rejection for incompatible historical worlds. Exercise Academy pending
  choices, stale Commands, reconnect and replay/fork identity for new decisions.
  Include the required choice-navigation, Learn setup, model/checkpoint,
  observation, action-cutoff and native-buffer-capacity suites named in
  `docs/development.md#validation-by-change-area`; the four scalar parity cases
  above do not replace them. Browser checks can run headless after frontend
  generation/build, on isolated ports.
- Fetch the pinned corpus through `manabot.data` with R2 credentials injected
  by Doppler; read the same first 1,000 ordered events and admit every full deck
  and sideboard through the public import path. `experiments/tla_corpus.py`
  retains them as `Corpus.trophies[*].decklist`; use all trophies, not
  `Corpus.eligible` or the saved `Trophy.deck`, which reflect old admission.
  Call `managym.admit_decklists` on those counted sources. Record the input digest
  and per-event result, expecting 1,000 passes and no dropped cards. Never regenerate
  the source corpus or alter held-out membership to satisfy this check.
- Run the shipping guide's Scout's Warning example against the rebuilt engine.
  Perform any required artifact publication before tracking an admission receipt
  under `experiments/data/`. Reuse CI's engine budget comparison rather than
  inducing local load or comparing timings across hosts.
