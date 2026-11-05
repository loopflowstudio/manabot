# MTG-242 — mana spending and resources

Accepted outcome and constraints: Jack Heart's 2026-11-04 Task directive;
the implementation approach below is derived from it, not a separately accepted
design. One reviewable PR, stopping at review; no merge or training launch.
Original base: `4463e57f5a92ebd3b60e2fb50c046b84ac2c0372`.
Reconciled 2026-11-04 at `b4ac66e4e`, which integrates main `51b4bd274`.
The main-comparison mitigations are `0c6eef111` and `6519f3f75`; subsequent
payment simplification is retained at `91a533b86`. Original risk implementation
and simplification remain at `5e7b60047` and `f9f55efb1`. The independent
main comparison pins `fe83a6b06` and candidate `6519f3f75`; it is not a timing
measurement of the reconciled checkout.
The bounded risk work is implemented; the outcome remains incomplete pending
shared event completion, cross-Task schema integration and final validation.
This Task has no Wave.

## Intended outcome

Offers and execution share typed spending restrictions, announced spell or
activation context, and exact resource transitions. A child payment preserves
completion of its parent. Production appears only after costs commit. Commands,
forks, observations, learning tokens, text and Academy describe the same choices.

## Delete — do not maintain

Remaining integration cut: remove `advance_mana_upfront`'s synchronous sacrifice
followed by in-place `ManaUpfront` replacement. Preserve committed costs, exact
source, post-cost variable output evaluation, selected taps and the original
parent through MTG-241's shared completion owner. This depends on its retained
zone/event implementation and a stable handoff; no competing frame protocol.

Removed predecessors: executable `ManaUse`/`SPEND_SUBTYPES` masks, one-use source
availability, greedy class assignment, pre-cost variable-output flattening,
single-use registry guard and `Mana::usable_total`. No exclusive fixtures remain.
Frozen projection labels and v8's independent-row reducer remain intentional.
Earlier rationale: `7e6aa4933691d41968940a1ae03a2837a5c6da06:scratch/mtg-242-mana.md`.

The surviving `PaymentModel` owns one immutable inventory per enumeration and
independent changing probe resources. Optional activation history and shared
spending snapshots eliminate eager per-source lists and repeated deep copies;
ability indices remain unbounded and equality/serialization retain full values.
Terminal fixed payments stream remainders and use source order, checked against
the complete general-search witness in debug. Hashing uses changing resources,
pool, life, prefix and obligation count; collisions compare full continuations.
No live-state cache or alternate restriction authority is introduced.

Classification shares the fixed-output matcher; fixed choices are borrowed,
tap bonuses computed once per choice set, and pending-cost probes copy only
costs. Reservation and obligation collection walk continuations iteratively.
Automatic-prefix budgets share one initial state rather than rebuilding its
resources and obligations. The obsolete obligation Hash derive is removed;
visited-state equality still compares every purpose and obligation.

V9 removes padding and independent mana-row projections, joins complete classes
with exact scalar routing, and routes both row and class embeddings into one
object accumulator. Saved parameters, gradients and v8 semantics remain intact.
The class encoder returns targets and embeddings; it no longer allocates a
second object-sized delta. V8/v9 parity uses one configured test path and policy
probes derive capacities and field indices from their existing owners.
These simplifications follow the retained acceleration measurements; they make
no additional throughput claim or change their pinned source receipts.

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

The w18 conformance corpus retains 14 cases / 3473 Commands and 15 Phase rows;
w17 remains unchanged. Earlier implementation and validation details remain at
`7e6aa4933:scratch/mtg-242-mana.md`; these bounded results do not establish
full-suite or post-integration acceptance.

## Integration findings and remaining work — 2026-11-04

Main's selected-cost X repair, configurable native buffer axes and explicit
first-legal action policy are integrated. V8 keeps relation-width selection,
caller-supplied capacities and original action indices. Reconciliation found
two existing overflow tests failed because the new legacy-mana guard inspected
player state before the public action-capacity check. The encoder now validates
action capacity first; the fixtures and overflow policy remain unchanged.
The TLA pilot's frozen protocols and training evidence are not v8 evidence.

The earlier main sync added the published Standard roadmap and
`managym/PRINCIPLES.md`. The guide reinforces shared completion ownership,
consequential policy distinctions and equivalent-work measurements, including
cumulative milestone cost alongside the rolling Engine Budget reference.
The new engine-performance Wave and renamed coverage Wave do not associate
this Task with a Wave; their memory remains read-only here.

The latest sync (`b4ac66e4e`, main `51b4bd274`) also integrates opt-in native
vector observation-capacity recovery and its Python collector/receipt plumbing.
Only typed capacity errors in eligible isolated steps become failed-game
outcomes; malformed payloads, buffer errors and reset failures remain fatal.
Fresh recovery remains disabled. The merged buffer tests retain both main's
41-permanent overflow/recovery witness and this branch's v8 parity cases.
The TLA v10 protocol and experiments Wave reorganization are upstream context,
not mana-input acceptance or training authorization. Required gate coverage now
includes the merged recovery/receipt path as well as observation/cutoff checks.

MTG-241's 2026-11-04 completion counterexample remains unresolved in this checkout:
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

Read-only inspection on 2026-11-04 of MTG-241 at `d83376181` supersedes the
earlier missing-interface finding. Its retained `EventBatch`, `FrameCompletion`
and `run_before_decision` now support linked-entry replacement suspension.
`flow/resolution_tests.rs::retained_batch_linked_entry_suspends_actual_mana_cost_completion`
constructs a tap/life/sacrifice mana source with a linked return and two output
colors. It checks the actual replacement pause, queued tap trigger, uncredited
output, one sacrifice/return and original payment completion through checked
Commands. MTG-241 records passing debug evidence at `301877ea1`, then focused
checks after simplification at `7b381909e`. This evidence is for its fixed-output
engine, not this branch or a combined implementation.

The 2026-11-04 Task status refresh still reports MTG-241 in implementation,
with no PR or superseding integration handoff. Its request against copying live work remains in force;
no work was copied or modified there. The dependency is stable integration and
adaptation of an existing mechanism, not implementation of replacement
suspension from scratch. MTG-244 has separately published PR #427 and resumed
implementation for CI repairs; its pinned schema and current limits are below.

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

MTG-244's current layer work also invalidates characteristics on taps/untaps and
conditionally rebuilds its raw board when a layer reads unspent mana. This
qualifies the local optimization's current assumption that taps preserve layer
caches (`transition_outputs`). Combined code must use the layer owner's typed
accessors and preserve its invalidation, with debug recomputation and renewed
timings. The earlier timing reduction cannot be carried forward as a speed claim
under the expanded layer semantics.

## Cross-Task input integration — 2026-11-04

Jack Heart approved an explicit successor policy input for MTG-244 on
2026-11-04, preserving fresh v6 and frozen saved versions. Read-only inspection
of [published PR #427](https://github.com/loopflowstudio/manabot/pull/427), head
`6165e0fe48e7c4d500e724b030b6dc11fce749f6`, confirms that it still selects v8
independently of this branch. Its stable encoder/configuration sources confirm
the earlier collision, not merely an unreserved version number:

| Contract | MTG-242 v8 | MTG-244 published v8 |
| --- | --- | --- |
| Fact kinds 11 / 12 / 13 | ManaAmount / ManaPredicate / ManaLocation | EffectProgram / EffectiveColor / KeywordCounterTimestamp |
| Fact-kind maximum | 13 | 14, including ContinuationProgram |
| Relation width | 50, including remaining mana taps | 49 |
| Effect-detail capacity | 512 | 4096 |

MTG-242's v9 retains the v8 wire layout in the table and changes only model
aggregation; it does not resolve this collision. Publication supplies a pinned
schema reference, not an accepted replacement for MTG-242's schema.

The 2026-11-04 Task refresh reports PR #427 open with implementation resumed.
Jack Heart's repair directive (comment `2029bc84-f1b5-4c9f-afc9-7006f9ad3c4d`)
records matched CI digests but search total 168.192 → 453.756 seconds,
worst reply 4.909 → 15.349 seconds and steps 12,817 → 5,654.8/s. It also
records a missing-reference failure before layer-risk timing, collector deadline
failures and six Academy UI failures. These are reported dependency results,
not measurements of MTG-242 or the combined engine. The layer Task owns repair
and remeasurement without a baseline reset; no superseding repaired handoff is
established here. Its published schema remains inspectable, while its live
repair work is not an integration source.

Remaining integration includes a distinct successor contract combining mana,
layer and MTG-241 continuation payloads without reinterpreting existing v8
facts or dimensions or replacing saved v9 model semantics. Its concrete
version/layout remains implementation work;
neither a silent v8 merge nor a fresh-default change is accepted. Both encoders,
model construction/forward, text, native scalar/vector buffer binding and saved
configuration handling must agree on that contract. Existing rejection of
unrepresentable older-input state and action-capacity validation order remain
required. MTG-244's published code also still names world w17; this branch names w18 and
retains a w18 corpus. World identity and conformance evidence must be reconciled
for the combined semantics without rewriting retained corpora. Neither branch's
isolated projection results prove combined-schema acceptance.

## Remaining validation and review

Gate acceptance still includes the complete debug Rust suite and all-targets,
all-features Clippy; current-world arena admission/monitoring; broader model and
active-training checks; merged recovery and resource-limit receipt checks
(`tests/training/test_recovery.py`, `test_game_error_recovery.py` and
`test_game_errors.py`); observation/cutoff checks; Academy choice-navigation/Learn setup and browser proof;
post-sync conformance replay; and same-runner Engine Budget, including reference
adoption if the curated decision digest moves. Browser evidence unavailable
headless remains with gate/CI or demo. Review must resolve the MTG-241 integration
boundary before claiming completion; no new product decision is otherwise needed.
Publication and review remain later Graph steps.

Prior focused debug, rebuilt-extension, projection/model, Clippy and formatting
results are retained in the plans at `68a2094dc` and `f9f55efb1`.
`91a533b86:scratch/mtg-242-mana.md` retains the post-simplification check:
42 focused debug Rust tests, rebuilt extension, 13 projection/model cases,
library Clippy and formatting. These precede the latest main sync.
Broader suites and post-integration Engine Budget remain with gate.

## Accepted risk measurement and mitigation — 2026-11-04

### Superseding acceleration direction

After seeing the final main comparison, Jack Heart rejected the remaining
6.7–8.0% median cost as insufficient and requested "make it stupidly fast".
This supersedes the earlier no-major-slowdown stopping target below. The goal is
to erase that regression and pursue a substantial measured speedup, without
promising an unmeasured percentage or weakening behavior. New independent work:

- Reprofile the current optimized engine on bounded ordinary step/search and
  representative complex payments. Attribute remaining CPU, allocation and tail
  costs rather than repeatedly rerunning wall clocks until a favorable sample.
- Investigate a structurally verified ordinary-payment path that avoids general
  search bookkeeping when unnecessary, inside the shared payment authority.
  Compare it against the exact general witness oracle in debug. No card-specific
  shortcut, choice pruning or separate restriction implementation.
- Examine compact resource states, reusable per-enumeration scratch storage,
  incremental/backtracking search state, continuation sharing and repeated source
  queries. Change only demonstrated hot costs; preserve full equality after hash
  collisions, exact identity and nested-parent obligations. If memoization is
  useful, scope/invalidate it for every relevant resource and rules revision,
  including new layer reads; retain debug recomputation. No whole-engine rewrite.
- Address the measured model aggregation overhead as well: profile and attempt
  batched/vectorized class joins and reductions without per-class interpreter
  loops or unnecessary synchronization. Preserve arbitrary-label invariance,
  purpose/amount/mode/parent distinctions, policy gradients, saved semantics and
  native/Python parity. Measure model cost separately from engine timings; do not
  restore the faster representation that lost the required invariants.
- Keep before/after main/base comparisons on the same runner and equivalent
  workload, retaining count/digest checks, distributions, CPU diagnostics and
  raw failures. Track new-mechanic absolute costs separately. Recheck the accepted
  Command/fork/remainder cases and affected debug suites after algorithm changes.
  Stop unproductive micro-tuning once evidence supports that judgment, and report
  the remaining dominant cost candidly; do not claim success from the old 10% bar.

Proceed with this work before returning to the unchanged integration dependency.
Reassess stable handoffs but do not copy live MTG-241 work. No merge, training
experiment or artificial load. The full Engine Budget and post-integration
validation remain required; existing reference evidence must not be rewritten.

Jack Heart requested measurement and active mitigation of performance and learning
risks. This authorized independent work before completion/schema integration;
its implemented results and limits are recorded below. The earlier no-net-speedup
stopping target is superseded by the acceleration direction above. The standing
10% Engine Budget remains a guardrail, not sufficient evidence of that goal.
Delivery remains
review-only: no merge, training experiment or artificial load. Bounded
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

## Acceleration pass — 2026-11-04

The superseding request is implemented as another bounded pass from `b4ac66e4e`;
`research/mana-risk-2026-11/acceleration/final/source.patch` pins the measured
runtime changes over that parent. No dependency code was copied. A fresh shared
status read still reports MTG-241 and MTG-244 in implementation; MTG-241 has no
PR and MTG-244's PR #427 remains open. Their handoff boundaries above persist.

Retained changes stream first-result pool remainders, use one `try_pay` for a
terminal cost, walk independent fixed taps in source order (complete witness
comparison with general search in debug), discharge zero upfront costs directly,
and bound/memoize at payment nodes rather than forced output/tap steps. Search
keys share immutable continuation/resource slices with local `Rc`; mutations
copy only affected resources. Full equality still covers every obligation and
purpose. No live-state cache, arbitrary cap, new restriction authority or
automatic choice of a modal/activated source was added.

The first cut improved branch medians only about 2%; that attempt and the noisy
forced-step cut remain in `first/` and `forced/`. The final three serial rounds
use 128 games (26,277 decisions, zero capped) and nine Search-64 replies per
revision; every count, decision and score digest agrees. Reference binaries
retain the previously pinned original base and main `fe83a6b06`.

| Final median | Original base | Pinned main | Before (`b4ac66e4e`) | After |
| --- | ---: | ---: | ---: | ---: |
| 128-game elapsed seconds | 0.931 | 1.075 | 0.980 | 0.926 |
| Search total seconds | 8.248 | 8.400 | 9.872 | 7.804 |
| Search process CPU seconds | 8.184 | 8.331 | 8.547 | 7.766 |
| Worst reply per round seconds | 1.573 | 1.527 | 1.704 | 1.497 |

The retained candidate reduces branch CPU cost by 5.6% for steps and 9.1% for
search. These samples support recovery of the former regression, **not a
substantial net speedup over the original base**. In the final round, base and
candidate search total are 7.827 / 7.804 seconds and step time 0.931 / 0.926.
Main's inflated samples must not turn near-base performance into a large win.
Candidate search totals span 7.774–11.651 seconds; its first 1.946-second tail
and 11.651-second total fail the raw 10% checks against all three references.
Every failure remains in `summary.json`. Unrelated compilers/test processes were
observed; no contention was induced or other process modified. CPU diagnostics
help attribute delays but do not replace wall-time acceptance. Dedicated Engine
Budget and combined-engine measurement remain unresolved.

Separate three-second profiles retain the sampled trees: DefaultHasher writes
fall from 75 to 33 top-of-stack samples, while allocation/free and layered-view
reads dominate the remaining list. Sampling is attribution, not a causal percent.
No broader storage/layer rewrite or further unmeasured micro-tuning was selected.
The requested substantial speedup remains unproven; the original 10% threshold
is not being used to claim that goal was achieved.

V9 now compacts linked rows before embedding and uses exact scalar class keys
instead of tuple-wise uniqueness. Parameter shapes and pooling mathematics stay
unchanged; v8 and older still run their frozen path. Alternating CPU batch-2,
width-16 probes with identical weights retain full-model output/gradient parity:
amount-assignment fact encoding 382.8 → 242.7 µs, output-choice 505.9 → 277.3 µs,
and remaining-tap 257.6 → 145.7 µs (37–45% lower). Whole-model medians remain
roughly 12.6–14.0 ms, with no convincing whole-model improvement. Raw timings,
scalar-only attempt, profiler tables and source hashes remain in `model.json`
and adjacent files. No GPU, optimizer, RL or playing-strength claim follows.
Ten instrumented samples per payment/offer retain identical concrete remainders
and all five full decision traces. Payable medians: fixed 1.292 → 0.313 µs,
shared taps 34.605 → 13.875 µs, Prism/Treasure 4.166 → 2.333 µs, variable
3.583 → 2.667 µs; variable failure 6.063 → 4.125 µs. Contrary evidence remains:
restricted-pool payable 0.625 → 0.730 µs and unpayable 0.250 → 0.458 µs;
the first variable-success sample reaches 45.5 µs versus 38.5 before. Small
fixtures are absolute costs, not baseline-compatible throughput ratios. The
direct fixed path visits no general search states; zero counters mean no DFS,
not no work. The first instrumented build reused a Cargo target across source
snapshots and returned an identical binary hash; `instrumentation-attempt/`
retains that excluded attempt. The accepted reference was rebuilt in a fresh
independent target, then the candidate root sources were refreshed and rebuilt.
Distinct hashes, complete outcome checks and final timings are in
`payment-audit.json`. The earlier uninstrumented comparisons are unaffected.

The new test covers batch/object isolation, maximum 32-bit class labels,
permutation invariance and gradients; the existing policy tests cover all three
encoder arms, intentional zero gates and checkpoint reloads.

## Risk measurements and mitigation implemented — 2026-11-04

### Main comparison completed and mitigated — 2026-11-04

The original before/after results below compared two modified branch revisions.
The independent comparison now retains original base
`4463e57f5a92ebd3b60e2fb50c046b84ac2c0372`, fetched main
`fe83a6b068dace2187613e1bf124f9677b077ff3`, initial candidate
`78cc5a8ba277ce2f3bc5fa14fe2e3697e229c8a9`, and final candidate
`6519f3f759b7bab997d68781e51b09366d2019ac` in
`research/mana-risk-2026-11/main-comparison/`. Manifests retain engine trees,
identical benchmark hashes, lockfile/compiler/binary identities, commands and
alternating order. Archived source snapshots used separate Cargo build targets;
no managed worktree or live dependency was copied. Counters are disabled for
all ordinary step/Search-64 comparisons. The repository Engine Budget readers,
digest checks and unchanged 10% timing checks produced each summary; the standing
baseline was not changed. No full CI-budget pass is claimed by these bounded runs.

Initial three-round, 32-game samples found a real regression versus main:
median step time 0.395 versus 0.276 s (+43.1%), Search-64 total 11.579 versus
7.877 s (+47.0%), and worst-reply median 2.225 versus 1.532 s (+45.2%). All
7,517-step and search-score digests matched. Main and original base were close;
intervening main changes include the selected-cost X fix and vector/encoder
capacity handling, not changed curated decks. These results establish net cost
rather than attributing a branch-local optimization to main.

A separate three-second CPU sample (retained compressed) showed source-resource
cloning, hashing and allocation among the hot paths. The first attempt skipped
bounds/hash work for payable terminal obligations and cloned tap resources only
for available transitions. It reduced step time but retained a 4.412 s search
outlier. The next attempt (`0c6eef111`) boxed optional activation history and
shared immutable spending-characteristic snapshots; equality/serialization stay
value-based and ability indices have no fixed cap. Its larger 128-game follow-up
still measured +11.1% step / +13.8% search elapsed time versus main, with about
+10.8% CPU cost on both. That contrary evidence remains in `compact-validation/`.
The final change hashes changing resources and continuation depth, while full
visited-state equality still compares every obligation and purpose. No choice,
restriction, class remainder or nested parent was removed.

Final three alternating rounds use 128 games (26,277 steps, zero capped) and nine
Search-64 decisions per round, per revision. All counts and digests agree:

| Final comparison (median per round) | Original base | Measured main `fe83a6b06` | Candidate `6519f3f75` |
| --- | ---: | ---: | ---: |
| 128-game elapsed seconds | 0.953 | 0.946 | 1.009 |
| Nine Search-64 replies, total seconds | 8.409 | 8.012 | 8.649 |
| Worst reply per round, seconds | 1.700 | 1.556 | 1.648 |
| Step process CPU seconds | 0.938 | 0.940 | 0.990 |
| Search process CPU seconds | 8.049 | 7.896 | 8.580 |

The final medians are +6.7% steps / +8.0% search / +5.9% worst reply versus
current main. Cumulative CPU cost versus original base is +5.6% steps / +6.6%
search. This is a bounded reduction of the measured regression, not a speedup
claim or policy/training measurement. That checkpoint selected no further optimization; the superseding pass above replaces that decision.

Tails remain explicit: final candidate search totals span 8.552–8.687 s and
worst replies 1.634–1.661 s; main spans 7.902–15.974 s and 1.521–2.935 s. The
15.974 s main run used only 8.340 s of process CPU, supporting scheduling delay
rather than equivalent added engine work. Candidate's third step sample still
fails the raw 10% check (1.050 versus main 0.946 s); its CPU cost is 0.989 versus
0.940 s. CPU diagnostics do not replace wall-time acceptance. Every failed raw
check and outlier remains retained. A dedicated same-runner full Engine Budget,
and affected remeasurement after dependency integration, remain gate acceptance;
these host-noisy samples cannot certify that pass.

Separately, ten instrumented samples per new-mechanic payment/offer retain
absolute costs at `6519f3f75`: shared-tap payable median 34.81 µs, variable
payable 3.81 µs / unpayable 6.19 µs, and Prism/Treasure payable 4.27 µs.
`payment-audit.json` pins the instrumented binary and the exact outcome check. Unsupported baseline mechanics have no ratio. Ten payment
outcomes/concrete remainders and five decision traces match the earlier evidence,
including legal choices, skipped decisions, fact occupancy and final resources.
Model v8/v9 cost and learnability limits in the retained risk table remain
separate and unchanged; no model or training algorithm changed in this pass.

MTG-244's published schema reference and subsequent CI-repair status are above.
MTG-241 remains live and its no-live-copy boundary remains intact. Completion/schema integration is still
required; the missing main comparison is no longer deferred to that dependency.

Baseline `e88f9b4062ed307aca6c60104d4e1c7bebb57b86` contains the opt-in
instrumentation and final bounded fixtures, before payment optimization.
`research/mana-risk-2026-11/engine-manifest.json` identifies both binaries and
candidate source bytes at `5e7b60047`; later clone/probe simplification makes no
new timing claim. Receipts retain individual samples, counters, concrete
remainders, Commands/fork traces, configurations and initial attempts. The first
fixture accidentally consumed its hand spell while placing a same-named source;
that attempt is retained separately, and final before/after fixtures both use a
fresh library copy. That earlier measurement checked both dependencies while they were live writers;
PR #427 is now the published layer handoff described above.

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

Remaining acceptance is integration of MTG-241's now-existing retained batches
with post-cost variable output and selected taps, the combined MTG-244 successor
schema/world identity, post-integration measurement rechecks and gate suites/
Engine Budget. The main-comparison receipts now quantify cumulative cost and retained
mitigations; their bounded, host-noisy results do not replace dedicated gate
acceptance. No training claim follows, and no product decision is missing. The authorized delivery boundary
remains review; publication and landing are not part of this checkpoint.

Check after simplification (2026-11-04): focused debug Rust mana/state/CR-106/Prism/shared-tap/filter/variable tests — 42 passed (`CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2`, filters `mana::tests`, `cr_106_mana`, `roe::`, `supportive_parents`, `scarecrow_guide`, `white_lotus_tile`); rebuilt cp312 extension installed atomically; `uv run --extra dev pytest tests/verify/test_mana_payment_projection.py tests/model/test_mana_input.py tests/model/test_mana_separability.py -q` — 14 passed (model and rebuilt-extension projection runs); bounded `uv run python` comparison against the pre-edit FactEncoder — 18 paired v8/v9 output/parameter-gradient cases passed, including empty facts; Rust formatting, changed-file Ruff, `git diff --check` and `lf context` pass. Full suites, all-target Clippy, dedicated Engine Budget and post-integration acceptance remain gate-owned.
