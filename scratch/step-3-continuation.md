# Step 3 continuation — 2026-11-08

Jack Heart replied "keep going" after this evidence-review recommendation:
"I recommend repairing BF16, lengthening the packed cells, and completing the missing measurements."

Continue that outcome: repair and validate BF16, make packed cells long enough
for the unchanged three-overlapping-post-warmup-updates minimum, and measure
the missing packing crossover and a384l8. Carry forward the direction to use
effective hashed records and investigate similar unnecessary overhead.

The first pass remains retained evidence, not a completed answer: 11 valid solo
cells, no admissible packed cells, BF16 failed before training and all a384l8
cells remained unrun. Step 3 inclusive estimated spend is $1.741, total experiment
$3.781; final billing needs reconciliation before allocating the remainder.
Keep the original $10 step-3 envelope inclusive of that first pass and the $50
experiment hard cap. This continuation does not increase either ceiling.

Make a prospective protocol amendment before execution; retain old source,
regime and result identities. Specify enough concurrent samples, a bounded
duration, required controls after any source change, and uncertainty. Prefer
missing cells and necessary controls over repeating the full grid. Verify the
BF16 CUDA path before renting the full campaign. Successful CPU checks were
insufficient admission last time. Recheck world/input admission after integration;
never relabel old evidence to match a new world.

ed2300 tracks the BF16 startup defect. Fresh status: unstarted, no checkout
Sessions or Graphs, no PR. Do not create another defect or competing repair.
The continuation's software plan must explicitly name where its repair is
delivered and keep ed2300 linked to that delivery. c5fba4 owns its separate
shared-effect catalog repair.

Profile specific avoidable costs from step-3/overhead-audit.md where useful to
the missing measurements. Preserve selected samples, RNG order, observations,
legal actions and output/gradient parity in correctness-preserving repairs.
Do not silently change minibatches, learning semantics or precision defaults.
TF32 learning equivalence remains unestablished.

The completed Graph b9289ddb stopped at evidence review; no Graph remains live.
Continue from experiment-design for the amendment, then the necessary software,
admission, freeze and execution steps under the authority above. Return the
resulting evidence here; routine details within these bounds need no repeated
approval. Material scope or budget changes remain separate decisions.

## C1 design, 2026-11-08

The owning design is the prospective **Step 3 continuation C1** section of
`experiments/protocols/training-throughput-b015d5-v1.md`. This scratch document
keeps the direction and implementation handoff; the protocol owns the schedule
and bounds. No run, rental or provider quote was made in this design step.

Question: locate the packing crossover and measure a384l8 and repaired BF16 on
the same Elemental recipe. These are speed/health results, not strength or
generalization claims. Keep the original w17/cb722dc8c evidence intact and put
new results in `step-3-c1/` with new source, cohort and regime identities.

- Fresh seeds 26401–26404; 26400 for CUDA admission, 26410 for diagnostics.
  Effective hashed input v6 / 512 facts, 256 streams, both seats, one producer,
  unchanged deck roster, capacities, complete-game closure and learning rule.
  Freeze actual current-world/source identities after software integration;
  new-source/new-host controls are mandatory, old results remain context.
- Eight updates per speed run; keep the three-overlapping-post-warmup-update
  minimum. One fresh twelve-update whole-cell rerun is allowed for insufficient
  overlap, with a matching solo endpoint when necessary. Never pool retries.
- Start with one rental and real CUDA regression plus a two-update w160l8 BF16
  worker run before opening the other rentals. Measure a384l8 solo early, then
  w128l4/w160l8/a384l8 packing with the retained 85% memory, 1.10 crossover and
  1.30 K=4 rules. The A40 w160l8 precision block remains four twelve-update
  legs (float32, repeat, TF32, BF16). Revisit w64 only for the missing lower
  bracket/original A40 packing cell after the higher-priority measurements.
- All comparisons name exact included update ordinals, per-seed latency,
  machine decisions/hour and billed cost. Show seed/window sensitivity, not
  false confidence intervals from dependent updates. No evaluation games,
  evaluation draws or checkpoint selection. Capture training draws, games,
  decisions, failures and replacement debt separately.
- New liability is at most `min(10 - prior_step3, 50 - prior_total, 8.00)`.
  Proposed lines: A40 $3.30/5h, A4000 $1.60/6h, L4 $2.40/4h, $.70 cleanup and
  failure reserve within those windows. Reconcile bills and reserve unresolved
  liability first; $1.741 is still an estimate. All work, including retries,
  fits 15 aggregate GPU hours and 1,600 aggregate updates. Decline a cell whose
  watchdog/cleanup liability cannot fit; completeness never overrides a cap.

### Software handoff

Route: **software**. Final source freeze remains open, not the design choices.
`lf task status ed2300 --json` on 2026-11-08 returned `unstarted`, no PR, no Graph
processes and no Sessions. Deliver that repair in b015d5's next software PR,
link ed2300 to it and record its validation there; do not launch a duplicate
repair checkout. Recheck ownership at implementation time.

1. Fix the demonstrated semantic-card dtype join with float32 output/gradient
   parity and CUDA full/cached/selected/empty-row forward/backward coverage.
   Inspect the actual GRU dtype: the observed error said Half under BF16.
   Remove the CPU test's overclaim that its coverage guarantees CUDA behavior.
2. Add separately identified C1 legs, endpoints, seeds and admission to the
   existing launch owner, retaining legacy four-update discovery and IDs.
3. Extend the analysis owner with C1 isolation, ordinal matching, overlap and
   cost fixtures, and complete cell dispositions; keep the validity minimum.
4. Add at most two bounded, fixed-input diagnostics for first-update recording
   and minibatch device/host transfers. Each is at most one instrumented update
   or 120 seconds; no extra rental. No speculative speed repair gates freeze.
   Any included repair needs fixed-input timing plus sample/RNG/output/gradient
   parity before freeze. c5fba4 retains ownership of its catalog optimization.

The protocol names focused tests and real CUDA admission. CPU-only success
cannot admit the repaired numeric path. Software delivery does not complete the
experiment; freeze, bounded execution, analysis and evidence review remain.
