# MTG-242 — mana spending and resources

Accepted outcome and constraints: Jack Heart's 2026-11-04 Task directive;
the implementation approach below is derived from it, not a separately accepted
design. One reviewable PR, stopping at review; no merge or training launch.
Original base: `4463e57f5a92ebd3b60e2fb50c046b84ac2c0372`.
Reconciled 2026-11-04 against `68a2094dc`, including main `8e8d702f8`.
The outcome remains incomplete: shared event completion and cross-Task schema
integration are still required. This Task has no Wave.

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
V9 excludes mana facts from the independent-row reducer before adding joined
classes; the old reduction remains solely for frozen v8 and earlier inputs.
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

Checkpoint `68a2094dc` adds
`rules::cr_106_mana::mana_sacrifice_linked_return_preserves_parent_payment_through_commands`:
a constructed Treasure pays life and sacrifices to fund Prism, returning an
opponent-owned card through a single entry-counter replacement. Both seats and
all five colors check uncredited output, exact life/mana, one sacrifice, stable
returned incarnation and parent completion. Every Command checks fork isolation,
replay agreement and stale rejection through the shared test helper. This is
passing synchronous-path evidence, **not** reproduction of replacement suspension.

Read-only inspection on 2026-11-04 of MTG-241's checkout at `58535f9f2`,
including its live edits, confirms `FrameCompletion` and `run_before_decision`
exist. Its newer completion-event variant, common retained-continuation walk
and public-result selection do not implement linked-entry suspension or
replacement ordering. Task status reports an active compress process; its
coordination comment still requests no integration/cherry-pick while live.
No work was copied or modified there. The missing event interface remains an
implementation dependency, not a product decision.

The integration must retain MTG-242's post-cost `mana_outputs` evaluation,
selected-tap obligations, exact source and original parent. MTG-241 currently
constructs a fixed `ability.mana_choices()` output before its sacrifice child.
That fixed-output caller cannot simply replace this branch's variable-output
path: the shared completion must retain the output calculation until all cost
work, including a suspended linked return, finishes. Only then can it expose
the output choice. The actual suspended-return Command/fork witness remains
required to prove single cost commitment, no early output and exactly one
parent resumption; the synchronous witness above remains useful regression evidence.
Payment resources/search remain MTG-242's; completion/events remain MTG-241's.

## Cross-Task input integration — 2026-11-04

Jack Heart approved an explicit successor policy input for MTG-244 on
2026-11-04, preserving fresh v6 and frozen saved versions. Its live implementation
at checkout HEAD `9486ff3b4` provisionally selects v8, independently of this
branch. Inspection of its encoder, shared-effects module and model establishes
a concrete collision, not merely an unreserved version number:

| Contract | MTG-242 v8 | MTG-244 draft v8 |
| --- | --- | --- |
| Fact kinds 11 / 12 / 13 | ManaAmount / ManaPredicate / ManaLocation | EffectProgram / EffectiveColor / KeywordCounterTimestamp |
| Fact-kind maximum | 13 | 14, including ContinuationProgram |
| Relation width | 50, including remaining mana taps | 49 |
| Effect-detail capacity | 512 | 4096 |

The draft is active work, not an accepted replacement for MTG-242's schema.
Remaining integration includes a distinct successor contract combining mana,
layer and MTG-241 continuation payloads without reinterpreting existing v8
facts or dimensions. Its concrete version/layout remains implementation work;
neither a silent v8 merge nor a fresh-default change is accepted. Both encoders,
model construction/forward, text, native scalar/vector buffer binding and saved
configuration handling must agree on that contract. Existing rejection of
unrepresentable older-input state and action-capacity validation order remain
required. MTG-244's draft also still names world w17; this branch names w18 and
retains a w18 corpus. World identity and conformance evidence must be reconciled
for the combined semantics without rewriting retained corpora. Neither branch's
isolated projection results prove combined-schema acceptance.

## Remaining validation and review

Gate acceptance still includes the complete debug Rust suite and all-targets,
all-features Clippy; current-world arena admission/monitoring; broader model and
active-training checks; Academy choice-navigation/Learn setup and browser proof;
post-sync conformance replay; and same-runner Engine Budget, including reference
adoption if the curated decision digest moves. Browser evidence unavailable
headless remains with gate/CI or demo. Review must resolve the MTG-241 integration
boundary before claiming completion; no new product decision is otherwise needed.
Publication and review remain later Graph steps.

Retained checks at `68a2094dc:scratch/mtg-242-mana.md`: 40 focused debug Rust
tests and seven projection/model tests passed after the release wheel rebuild
and atomic extension installation; the validation-order repair selection had
123 passes. These results are reused, not rerun for this notes-only reconciliation.

Check (2026-11-04): focused debug mana/resource/card tests passed; rebuilt and atomically installed wheel; 13 focused Python projection/model/checkpoint checks passed, with four model checks rerun after the empty-class bypass; focused library/measurement Clippy, Rust formatting, Python Ruff and diff checks passed. Full gate acceptance remains deferred until completion/schema integration.

## Accepted risk measurement and mitigation — 2026-11-04

Jack Heart requested measurement and active mitigation of performance and learning
risks. This adds independent implementation work now; do not stop again solely
on the existing completion/schema dependencies before doing this work. Preserve
review-only delivery: no merge, training experiment or artificial load. Bounded
actual-workload timings and model forward/backward unit probes are in scope;
they do not establish RL learnability or playing-strength improvement.

1. Use bounded representative fixtures: fixed sources, mixed restriction classes,
   repeated shared taps, Prism/Treasure nesting and variable producers, each with
   payable/unpayable obligations. Instrument explored/deduplicated/pruned states,
   dynamic evaluations/game clones and maximum obligation depth without production
   hot-path overhead. Retain before/after code versions and identical cases.
   Measure offer generation and existing step/search workloads on one runner,
   including distributions, sample counts and tail cases. No stress boards or
   contention generation; honor TESTING.md.
2. Attempt targeted mitigation of measured costs: unconditional pruning bypass
   for dynamic producers, cloned-game output evaluation, duplicate resource paths
   and restriction remainder growth. Keep demonstrably useful changes. Bounds
   must remain optimistic when sacrifices change characteristics. No arbitrary
   search cutoff, removed legal choices, card-specific shortcut or unsound cache.
   Preserve debug witness oracles, exact outcomes and resource remainders.
3. Test the actual model path, beyond raw tensor inequality. Matched observations
   must distinguish spending purpose, amount-to-restriction assignment, combined
   versus separate classes, pending output mode/frame and remaining selected taps.
   Check representation/action separability and gradient paths; distinguish
   intentional zero-initialized adapters from structural information loss. Test
   harmless row/class reordering separately. If aggregation loses joins or relies
   unnecessarily on arbitrary group IDs, repair it within an explicit successor
   input, preserving native/Python, scalar/vector, text and checkpoint contracts.
4. Measure fact occupancy, encoding/model cost and nontrivial payment-decision
   counts on these positions. Inspect whether extra decision steps alter discount
   or return semantics; report before changing learning algorithms. Preserve real
   strategic choices and native trivial skipping. Exercise old-input rejection
   and new-input acceptance; do not silently change frozen schemas or fresh v6.
5. Completion/schema integration remains required: prove suspended costs commit
   once, output waits, parent resumes once, and combined schema identities are
   unambiguous when dependencies are ready. Recheck affected measurements after
   integration. Retain a risk/evidence table with baseline, attempted mitigation,
   measured effect and residual limitation. Broader suites and same-runner Engine
   Budget still apply. Do not infer training benefit from engine timings or
   model separability.

## Risk measurements and mitigation implemented — 2026-11-04

Baseline `e88f9b4062ed307aca6c60104d4e1c7bebb57b86` contains the opt-in
instrumentation and final bounded fixtures, before payment optimization.
`research/mana-risk-2026-11/engine-manifest.json` identifies both binaries and
candidate source bytes. Receipts retain individual samples, counters, concrete
remainders, Commands/fork traces, configurations and initial attempts. The first
fixture accidentally consumed its hand spell while placing a same-named source;
that attempt is retained separately, and final before/after fixtures both use a
fresh library copy. Both dependency Tasks were rechecked and remain live writers.

V9 is an explicit model-input successor, not a combined MTG-244 schema. It keeps
v8 wire bytes, kinds and capacities, but removes mana rows from independent-row
pooling and joins their amounts/predicates/locations within each object/class.
The class identifier is never embedded. Frozen v8 keeps its prior aggregation;
v1–v7 rejection, v8/v9 acceptance, native/Python and scalar/vector parity, text,
learning-token definitions, saved configurations and fresh v6 remain intact.
A future combined input must still reconcile the cross-Task collision above.

| Risk | Baseline / attempt | Retained mitigation and measured effect | Residual limit |
| --- | --- | --- | --- |
| Variable search disables all bounds | Unpayable fixture: 22 states, 29.25 µs median | Resume optimistic bounds after variable sources are exhausted and no dynamic output is pending: 17 states, 13.40 µs | Available/pending variable output still disables finite bounds; sacrifice can increase output. A debug Command witness proves that case. |
| Dynamic output clones invalidate layers | 1 clone payable / 2 unpayable | Keep typed tap/mana writes and avoid unchanged life writes; payable fixture 15.69 → 8.35 µs with the bound change | Clone counts remain 1 / 2. No persistent cache or unproved current-output upper bound. Timings are for the combined candidate. |
| Equivalent shared-tap paths | 68 states / 9 duplicates | Canonical sets plus insufficient-tap rejection: 65 states / 6 duplicates, 120.98 → 119.50 µs | Small search reduction; no convincing isolated timing win. All real first-tap choices remain legal. |
| Restriction remainder growth | Three successful terminal allocations, 7.54 µs | Use ordinary deterministic spending only when no parent remains: one remainder, 2.48 µs | Nested parents retain every concrete class remainder. Directed tests compare the terminal choice with the original enumeration. |
| Model learns arbitrary class labels | V8 class relabeling changes player representation by 0.9501 | V9 object/class join makes relabeling invariant; purpose, amount assignment, combined/separate classes, output mode/frame and remaining taps remain policy-separable across rows/typed_pool/typed_gru | Seeded width-16 forward/backward probes, not RL evidence; finite embeddings do not prove global injectivity. |
| Focus-free policy misses player facts | Initial no-FiLM probe changed value but not output-choice logits | Existing explicit `policy_film=True` supplies that path; probe opens intentional zero fact/FiLM gates and checks normalized action preferences and upstream gradients | Defaults are unchanged. Initialization equality is expected; no optimizer or training experiment ran. |
| Model cost / fact capacity | Alternating width-16, batch-2 CPU probes: v8 fact encoding 261 µs (pool), 286 µs (restricted output) | V9 joins cost 400 / 517 µs; empty-class bypass brings no-mana facts to 299 µs versus v8 293 µs | Full output-mode forward median 16.18 → 24.37 ms in noisy samples (individual replies up to 105 ms across cases). Raw distributions retained; no whole-model speed claim. 55/512 fact occupancy is not a general bound. |
| Additional payment decisions | Fixed payment: one retained cast | Shared taps: four retained decisions / six skipped; Prism/Treasure: three / three; variable: one / three. Before/after traces and final remainders agree | Sequential gamma/lambda count retained same-player decisions. Gamma=1 preserves terminal discount, not sampling or bootstrapped trace length. No learning algorithm changed. |
| Broad speed / tails | Five alternating eight-game samples; nine Search-64 decisions per version | Same digests. Median eight-game total 0.303 → 0.277 s; search total 36.778 → 34.233 s | Worst search reply 7.421 → 8.223 s (+10.8%). Instrumented, ambient activity uncontrolled; full uninstrumented Engine Budget remains with gate. |

Ten samples per payment/offer fixture retain tails and explored/deduplicated/
pruned counts, dynamic evaluations/clones, remainder counts and maximum pending
obligation length (five here). Exact successful remainders and unsuccessful
payments agree before/after. Native trace encoding also records ten samples per
retained decision. Trivial skipping may carry the spell through resolution;
immediate payment remainders are retained by the separate payment probes. Model receipts include ten forward and three forward/backward
samples per pair; alternating v8/v9 probes separately time fact aggregation and
the complete actor. Occupancy reaches 55/512 facts in the restricted-output
fixture (10.7%; 55/128 would be 43%). Each default input is 165,616 bytes,
including 24,576 bytes of fact capacity. Exact-fit and one-row-short buffers
prove explicit admission/overflow, not capacity safety for arbitrary boards.

Remaining acceptance is unchanged: suspended linked-return completion through
MTG-241, the combined MTG-244 successor schema/world identity, post-integration
measurement rechecks and gate suites/Engine Budget. The authorized delivery
boundary remains review; publication and landing are not part of this checkpoint.
