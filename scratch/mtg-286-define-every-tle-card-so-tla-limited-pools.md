# Every TLE card in the development world

Status: implementation in progress. The Merchant payment slice is implemented;
233 definitions and the full clause allocation remain open. State/observation
contracts for phasing, end-turn and player counters still need detailed design.
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

Proposed scope assumption: retain the current 1v1 development format.
Commander-dependent clauses must
behave correctly when no commander exists; adding a Commander format or
multiplayer setup is not part of the proposed scope. That does not permit
substituting a legendary creature for a commander or making Arcane Signet
produce arbitrary colors. Catalog tokens must still describe the printed rule.

The product choice left for demo is whether correct no-commander behavior in
the existing 1v1 world fulfills this Task, or whether playable Commander setup
and multiplayer are also required. No quoted decision settles that expansion;
the proposal keeps 1v1 while retaining all 242 names and their printed clauses.

## What exists

The catalog now includes The Cabbage Merchant with counted tap costs and its
two triggers. Seven directed Merchant tests cover repeatability, reserved costs,
waterbending, observations, Command replay and printed triggers. Learning tokens
carry the counted cost. Existing single-tap definitions preserve serialization.

`managym/cards/tests/fixtures/tle-inventory.jsonl` pins all 242 Oracle identities,
TLE printing IDs, first-printing IDs/sets and every Oracle paragraph. It retains
existing definition identities, chooses KLD over its simultaneous MPS Gearhulk
printing, and records Three Dreams' earlier PSAL printing. This is source data,
not a completed rule allocation or support list. Nine names are now defined;
233 remain absent. No corpus re-admission receipt exists. The shipping guide's
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

1. **Complete the rule allocation.** Use the pinned first-printing identities
   and map every printed clause to an existing or planned native
   rule. Use the existing TLA allocation fixtures as a structural precedent,
   specifically `managym/cards/tests/fixtures/tla-allocation.jsonl`, without
   importing their old Task ordering. Pin the TLE name inventory in an
   offline completeness test; it only detects missing definitions and never
   controls admission. Add Oracle shells to the existing Scryfall conformance
   snapshot (`managym/tests/fixtures/scryfall_cards.json`) when definitions are
   added. Resolve token identities separately. This allocation is the next
   concrete deliverable; the mechanism families below are not a complete design.

2. **Payment resources and production.** The Merchant tap-resource subset is
   implemented under the contract below, with focused evidence in
   `scratch/questions.md`. Finish the shared vector/Academy proofs. Design
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

The full allocation in slice 1 is still required. The table above is a verified
set of counterexamples, not a claim that all 233 remaining cards have been designed.

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
The focused proof must include clone/Command replay at each tap and output,
competing waterbend/activation resources, three versus four Foods, tapped
Merchant, and its other two printed triggers before registration is complete.

## Next production boundary — stop dependent implementation

Heartbeat of Spring cannot be represented by the existing triggered-mana
record. `TriggeredManaAbility` holds a fixed `Mana`; `zones.rs::tap_bonus_sources`
scans only the tapping player's permanents, and `tap_permanent(for_mana=true)`
fires bonuses before a chosen output exists. `Decision::ManaChoice` retains
choices and a caller but discards the source. Heartbeat applies to either player
and chooses a type that the land actually produced. Treating it as a fixed
bonus, computing its color from printed abilities, or granting it only to its
controller is incorrect. Animal Attendant also needs mana-unit provenance that
survives spending into cast/entry; `Mana` currently retains color/restriction,
not this entry-effect provenance. These are production/observation contract
changes, not more definitions over the completed tap-resource slice.

Revise the production design before implementing those consumers: retain exact
production source and actual output through triggered mana choices; gather
applicable triggers from both players; distinguish the producing player from
the trigger's controller; retain spend provenance into cast and entry. Define
viewer-safe joins, typed pool facts and scalar/vector/Python parity before any
writer. No new mutable state or guessed behavior for those cards was added.
The implement rule stops dependent work at this concrete counterexample; the
Merchant slice remains independently testable. All-card scope is unchanged.

## Delete — do not maintain

Completed: replaced `PaymentSearchState.available`, both one-use pruning
bounds and `registered_mana_activations_have_finite_source_costs`. The surviving
resource model and explicit tap witnesses retain reservations and finite
progress. Single-object tap costs use the counted type while serializing identically.
Compression removes duplicated selected-tap reservation/count logic:
`PaymentModel::plan_after_tap` consumes the choice on the same resource state
used by search. Callers pass the full unpaid count. Nested candidate offers
build one model/pool per action list; ordinary tap-cost offers use one complete
affordability search. No parallel mana planner or remaining deletion target.

Keep `ExileUntilNonlandCastFree` for its distinct existing card behavior.
No catalogs, importers, frozen receipts, split assignments or card definitions
are deletion targets; there is no predecessor TLE implementation.

## Acceptance and remaining work

The Merchant subset of slice 2 is implemented. Remaining: finish slice 1's
per-clause rule allocation, design production provenance/triggered choices,
then implement the remaining slices and all 233 missing definitions. The
Merchant does not establish Heartbeat/Attendant support or corpus admission.
Do not advance the Graph on this partial implementation. Follow AGENTS.md's
debug-test and extension-rebuild requirements during implementation. At gate:

- Check every pinned TLE name is a defined non-token card, with unique key and
  first-printing Scryfall ID, learning tokens and directed behavior coverage.
  Run the debug Rust suite, conformance, observability, formatting and clippy.
- Run the required world-change Python suites from AGENTS.md and
  `docs/development.md`, including encoder parity/admission and checkpoint
  rejection for incompatible historical worlds. Exercise Academy pending
  choices, stale Commands, reconnect and replay/fork identity for new decisions.
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
