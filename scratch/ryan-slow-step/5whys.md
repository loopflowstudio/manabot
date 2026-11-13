# Payment stalls: why the checks missed them

2026-11-12, MTG-334 incident step 3. Analysis only; no engine changes,
publication, release or live training changes in this step.

## Status and evidence

Ryan's PyPI training and flat-search workflow suffered 2–120-second engine
steps. His report says, “It is not an infinite loop: every slow step observed
did return.” The unchanged 426-action replay reproduced on isolated PyPI
0.2.1 and 0.2.2 (33.65 / 31.22 seconds at the final step). Those are the retained
measurements in [the recovery investigation](../ryan-training-recovery.md),
not a new timing run here.

The reported replay is repaired in this checkout by `e8f7bde51` and
`5e8eebbd3`: [debug receipts](debug.md) record final scalar/vector steps of
1.399 / 1.404 ms, matching all 427 decision boundaries, plus a bounded
open-deck search proof. This is **a source repair, not restored service for
Ryan**: no replacement PyPI package or live Jace verification is established.
Exact payment search still has other combinatorial cases. The concurrent
colour-bound work is outside this step; its checkout was read, never modified.

[5whys-evidence.json](5whys-evidence.json) retains the GitHub state read today,
the initial budget failure excerpt, publication dates, the other checkout's
instrumented step-benchmark receipt and an injected timing check. Existing
reproduction logs remain unchanged. No costly replay or benchmark was repeated.

## Causal chain

1. **An ordinary decision asked an expensive impossibility question.**
   The retained native samples locate the stall in `PaymentModel::witness`,
   reached through activation-X legality and waterbend payment, rather than
   establishing board-view construction as the cause. `payable_x_values`
   tests successive values until the first unpayable one: proving that final
   value impossible is part of producing the legal action space. Training,
   scalar play and search share that authority.

2. **The solver confused distinct identities with distinct payment problems.**
   The release memo key retained a bit per permanent. Identical lands therefore
   generated different keys for equivalent subsets. Weak optimistic bounds let
   infeasible obligations reach those states, and automatic-prefix budgets
   repeated searches. The release whole-payment bound counted restricted output
   unavailable to the debt; its spendability check covered only the first debt.
   #469 additionally summed uses competing for the same tap. These bounds were
   conservative for legality but poor at limiting work. Termination and a correct
   first witness did not imply acceptable cost. The controlled comparisons in
   [debug.md](debug.md), such as 255 versus eight visited states on eight identical
   lands, isolate this mechanism without host-load explanations.

   The rewrite had a real semantic requirement: presence, taps and limited uses
   must be independent to allow tapping before sacrifice and repeated activations
   paid by other objects. Its gross-output bound deliberately relaxed competition
   and upfront costs to avoid rejecting legal payments. Keeping that correctness
   margin without checking search effort made the regression possible; reverting
   the richer resource model would lose the new rules semantics.

3. **Tests established rules answers, not the cost of rejecting a payment.**
   The inspected CR 106 and TLA tests cover pool arithmetic, restrictions,
   Herbalist modes, Waterskin and waterbend X. Many card fixtures deliberately
   supply ample floating mana (`five_color_setup`) or small unfunded boards;
   Katara's X test checks three taps plus one mana. #469's counted-tap tests use
   three/four Foods and check exact Commands. These are useful semantic tests,
   but do not establish scaling across interchangeable sources and conflicting
   obligations. Before this incident, `flow/mana.rs`'s module test checked that
   activations consume finite resources, not search effort. A 300-second per-test
   timeout catches a hung test, not a successful multi-second game step.

4. **Performance and consumer proofs sampled the wrong payment problems.**
   Engine Budget exercises one curated matchup: 512 uniformly random games,
   and Search-64 at 40 roots for each of three seeds. It checks aggregate step
   throughput and search worst/total reply time plus behavior digests. Those
   measurements can catch a stall *if the workload reaches it*. They do not
   establish coverage of resource/obligation interactions. The optional step
   tail output samples only the first 128 calls and is not the default gate.

   The other checkout's `e8f7bde51 + counters` measurement recorded 522,421
   payment plans but at most **11 witness calls per plan** in `bench_steps`;
   its report records at most 12 in `bench_search`. The same step receipt has
   `max_sources: 37`: total board size alone is a poor coverage criterion.
   Its constructed impossible-colour case needed 737,287 calls with 15 sources
   before memo canonicalization. These are post-recovery instrumentation results,
   not a measurement of #469's historical CI binary, but demonstrate the missing
   workload dimension. Ryan separately reported, “Uniformly random play did not
   show it in several hours, probably because random play rarely builds such
   boards.” The probability explanation remains his hypothesis.

5. **Successful slow calls did not produce the same feedback as failures.**
   In release-tag training code, the collector times the vector step plus
   encoding; persisted phase timings contain total seconds and call count.
   Its profiler's maximum is over a 100-entry reservoir, not every call.
   A controlled check here injected 999 durations of 0.004 s and then 35 s:
   total time was 38.996 s, but the retained sample maximum was 0.004 s.
   Exception capture retains failure evidence; a slow successful call does not
   trigger it. Collection checks its deadline between calls, so that deadline
   cannot identify or interrupt an in-progress native payment search.

   Evaluation is a useful counterexample: `EvalPerformance` already retains an
   all-call maximum for `native_step_and_encoding`, independent of its bounded
   timeline. The same injected check preserved 35 s there. Per-decision
   `latency_seconds` instead times policy selection, so inspecting only that
   field misses native execution. Whole-game supervision can time out and retain
   pending Commands; a successful slow step below the game deadline can still
   pass. There is no evidence here that a recorded native maximum was reviewed
   and ignored.

## How #469 and the releases passed

The releases precede #469. GitHub publication times were 2026-11-12 01:22 UTC
for v0.2.1 and 10:05 UTC for v0.2.2; #469 merged at 13:35 UTC. Their
`flow/mana.rs` files are identical. #469 aggravated an existing release defect;
it did not introduce that defect into either published version.

The [v0.2.2 release run](https://github.com/loopflowstudio/manabot/actions/runs/43161774626)
succeeded. The release smokes establish installed-package provenance, CLI,
training/checkpoint functionality and dependency compatibility. Both tags run
the local CPU recipe: two updates, two streams, a 16-transition target with
complete-game closure, the curated matchup, and a 180-second outer timeout.
V0.2.2 also plays one seeded random red/white game through the engine-only
wheel. Neither contains a late-board payment-complexity case or per-step latency
acceptance. Isolation was good; the sampled behavior was insufficient.

**The budget did flag #469's ordinary slowdown.** Its
[initial run](https://github.com/loopflowstudio/manabot/actions/runs/43221435403)
failed: 12,880.0 → 11,693.6 steps/s, below the 11,709.1/s threshold; search
worst reply 4.913 → 4.944 s and total 167.486 → 169.186 s passed. A 9.2%
throughput loss is approximately 10.1% more time per step. The
[PR description](https://github.com/loopflowstudio/manabot/pull/469) says,
“It is adopted as an accepted cost from” that CI run. The reference moved to
`7ef6ce588`; final Engine Budget, Rust, integration and package checks passed.
The PR body also says, “No v10 training run has been made yet.” GitHub returned
no review/comment entries; that does not establish absence of review elsewhere.

The supplied comparison of #469 with PyPI 0.2.2 at replay step 242 is
35.4 versus 0.68 s (about 52×); it was not rerun here. The source diff supports
the relaxed-bound explanation. The accepted 9.2% cost was measured on another
workload and cannot justify that tail. The documented reference-move procedure
requires explaining the measured delta, but does not require payment-complexity
coverage when payment state or pruning changes. This is the supported process
gap, not evidence that someone knowingly approved 35-second steps.

## Contradictions and remaining unknowns

- “Training never reached large boards” is false as a general explanation.
  [The TLA pilot report](../../experiments/tla-trophy-pilot.md) records a
  41-permanent capacity failure, later 65-permanent overflow, and thousands of
  completed games on earlier revisions. That proves exposure to large boards,
  not this payment interaction or the later #469 regression. Aggregate collection
  time cannot attribute historical stalls. Do not relabel old timeouts or missed
  training targets as this defect without their traces.
- The initial reporter stack description emphasized `payment_model`; retained
  samples and controlled fixes point into DFS. The final repaired replay opens
  X choices, not the report's described mana colours. Preserve these differences.
- The 11–12-call benchmark results were measured after the first repair; exact
  historical maximum search effort in #469's CI remains unknown. That limits a
  literal claim that its benchmark could never encounter an expensive payment.
- Ryan's live configuration, checkpoints and step telemetry were not inspected.
  Whether prior training/evaluation receipts contain unreviewed large native-step
  maxima remains open. No broad archive retrieval or new training campaign is
  justified merely to fill that gap.

## Prevention: implemented versus proposed

**Already implemented on this branch:** whole-payment spendable/tap-exclusive
bounds; canonical memo keys for interchangeable sources; small-board first-witness
comparisons and large-board state budgets that fail during exploration. These
remove the demonstrated identity-subset growth and make several recurrences
cheap deterministic test failures. They do not promise a universal polynomial
solver. The concurrent colour bound is not counted as delivered by this note.

**Proposals, not approved work:**

1. Keep those search-work budgets in ordinary debug CI and require a bounded
   infeasible-payment case when changing resources, restrictions, modal output
   or pruning. Check full witnesses on small boards as well as work counts on
   larger boards. This is the highest-leverage guard: legality digests alone
   cannot distinguish a fast correct answer from a slow one. Test failure must
   stop exploration; do not restore an exponential solver and wait for minutes.
2. Promote the retained replay proof from scratch into a maintained regression
   and installed-wheel check, including scalar/vector equivalence and a bounded
   search path. Keep the closed reporter fixture separate from open-deck search.
   Before the next release, run it against the built wheel outside the checkout
   and record native provenance and maximum step duration. No CI/release command
   currently invokes `verify_recovery.py`; existing unit guards cover the solver,
   but not that complete consumer path.
3. Extend Engine Budget with a small deterministic payment workload covering
   repeated basics, restricted mana, competing taps and impossible modal colours.
   Retain the curated throughput benchmark; separately report work counts and
   same-runner timings for the added workload. A source/plan coverage summary
   should accompany future payment-cost reference moves. Verify it distinguishes
   a controlled pruning regression while preserving witness/digest checks.
4. Add an all-call native-step maximum and slow-call count to training phase
   receipts, with a bounded replay-linked diagnostic for slow successful calls.
   Reuse evaluation's existing maximum rather than adding duplicate telemetry;
   expose it in routine run review. Validate with injected durations, including
   an outlier after the sample limit, and check that its game/action can be
   replayed. Timing is diagnostic, never a reason to silently drop legal actions,
   reset a valid game or change training samples.

No wider release-policy rewrite, stress test, general solver replacement or
longer random campaign follows from these findings. Focused complexity guards
and a real consumer replay address the demonstrated gap more directly.

Note: `scratch/ryan-slow-step/5whys.md`. Next useful action: carry the existing
complexity tests and the reporter's installed-wheel regression into the repair's
validation/delivery plan; review the concurrent colour-bound result separately.
Publication and live recovery remain later steps, not outcomes of this analysis.
