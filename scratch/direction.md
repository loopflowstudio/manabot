
## Coordination with MTG-241 (2026-11-08)

MTG-241 (#433) changes the w17 card digest that the protocol JSON pins (6a818cbc ->
0798b2c7: library searches as one selection, replacement ordering). Answered:
land 241 now; 241's PR moves `runtime_world.card_digest`. Nothing scored ran under
6a818cbc; calibration, the rehearsal and the fact-embedding check did. Recalibrate
on 241's world (under USD1) before the freeze. The fact-embedding verdict stands.

## Continued session, 2026-11-08

Preserved authority from the prior conversation and its saved memory (not new
approval): Jack Heart said "you dont need to ask for these small PRs" and
"any experiment you want to run thats <3 dollars, jsut do it. only ask for
approval on the big runs". Long frontier runs still require execution review.
The $10 continuation allowance was used for the 150-update vocabulary check;
its prior $100 total ceiling may rise by that allowance, not without limit.

Fresh observations:
- Endpoint eval `tla-frontier-c5fba4-vocabulary-v1-8eecfee66463`: succeeded,
  96/96 matches, launcher estimate $0.1106688784, rental ended. This is vocabulary
  evidence, not frontier scoring. Scores are being read from retained blocks.
- #433 remains open. New-world recalibration waits for its merge.
- Timing diagnostic `7446ef290152` launched at source `bf408ec48c20`, on one L4,
  cohort `c5fba4-vocabulary-timing-20261108`, $0.75 and 0.75 h caps, one attempt.
  Quote $0.59 at $0.59/hour. Exact diagnostic is in
  `scratch/vocabulary_timing_probe.py` and immutable Job.command. Same sampled
  observations, ABBA order, cached inference and forward/backward at 64/256,
  profiles separate from timing. Fresh weights, no optimizer steps; this isolates
  model cost, not end-to-end training throughput or strength.
- Task status says "Merged; complete the Task", but that refers to software PR
  #450. Scientific outcome is incomplete; do not complete MTG-267.
- Old spend headline $8.84 disagrees with its own line items. Fresh launcher
  allocation sum before timing is $6.4316537804, partly estimated, with every
  historical rental ended. Several settled allocation readings are materially
  below earlier whole-rental estimates. Do not silently treat that as a billing
  refund or budget credit; retain the conservative $8.84 + $0.11 pending audit.

Timing probe finished successfully (2026-11-08 ~11:43 UTC). Retained job log:

| Batch | Path | Catalog s | Hashed s | Ratio |
|---|---|---:|---:|---:|
| 64 | cached inference | 0.0992876 | 0.0664239 | 1.49476 |
| 64 | forward/backward | 0.214001 | 0.150998 | 1.41725 |
| 256 | cached inference | 0.310062 | 0.214489 | 1.44559 |
| 256 | forward/backward | 0.740119 | 0.492068 | 1.50410 |

This confirms a model-path difference on one GPU and fixed inputs, not its precise
cause or its end-to-end training multiplier. Earlier attribution of the whole gap
to pod variance is unsupported. Profiles and timing JSON are outputs of
`7446ef290152`; local receipt copies live under `.runs/c5fba4-20261108/`.

The original batch-256 catalog inference profile recorded 3,699
`cudaStreamSynchronize` calls, 1,452 `aten::nonzero` calls and 15,041 kernel
launches. `SharedEffectEncoder` resolves catalog names, program tokens and sources
inside its per-record loop. A candidate batches those catalog lookups after the
existing admission checks; original and batched inputs, state dicts, sequence
order, multiplicity and gradient paths are retained. Focused tests including
full independent-vs-batched output/gradient checks: 24 passed. No speedup has been
established yet, and the candidate is uncommitted.

The before/after CUDA quote refused L4 stock (zero new rental). The same bounded
job is queued under `c5fba4-vocabulary-batched-20261108`, $0.75 / 0.75 h / one
attempt, with original source and the exact candidate embedded in Job.command.
It first checks full-model CUDA output/gradient agreement, then compares original
and batched catalog on identical observations, ABBA order. Do not adopt the
candidate without the result. Scripts: `scratch/launch_vocabulary_batched.py`
and `scratch/vocabulary_batched_probe.py`.

MTG-241 integration update received as steer 107f2cdc (2026-11-08):
"amended the prospective runtime digest to
0798b2c7c7f54b7329043b13cb1742d74fe21b35373820580b76f979e6b7eb5f
in #433. Runtime-only execution vocabulary now has explicit catalog rows; old
preflight evidence stays on its recorded world. Fresh model totals are 449154
(w64) and 5697282 (w160l8), compute counts unchanged at 307010 and 5341922."
The steer reports tests and a pending push; it does not establish merge.
Recalibration stays here. Exact source is in the session's steer file.

Direct current-w64 parameter count (before #433): catalog 412482, hashed
1958402; buffer elements 260059 / 257115. Previous reply estimated parameters
from raw checkpoint bytes and was wrong: those files also hold buffers. The
user was told this correction; actual catalog parameter reduction is 4.75x
on this retained configuration, not the previously stated 2.6x.

Before/after diagnostic is queued as `9754bb253795` (L4 unavailable), with no
rental as of 11:50 UTC. The candidate in its immutable command is independent
of subsequent edits to this checkout.

At ~11:52 UTC the second diagnostic's launcher was `renting`, but logs show no
worker start: RunPod POST pods returned 400 with create outcome unknown, then
GET pods returned 429. The earlier progress statement that it had started was
corrected. Do not duplicate this rental request; reconcile `9754bb253795` /
`l4-235b6b5cad35`. No before/after result exists yet.

Candidate validation: `uv run --extra dev pytest tests/model/ -q` completed with
271 passed, 1 skipped; focused model and active-training checks 24 passed.
No production benchmark approval or merge is implied. The existing diagnostic
request is watched with `etude.launch.client.wait(..., every=30)` to avoid tight
provider polling; there is no second rental request.

## Current focus — 2026-11-08

Jack Heart: "win % against init is not very interesting"; "mostly need to
look at the speed issues of catalog it seems"; "focus on that".
Prioritize catalog runtime: correctness-preserving changes measured on identical
inputs on one GPU, then the remaining profile. Initialization comparisons stay
background; do not launch the long frontier training while this work is open.

Before/after request reconciliation, ~12:16 UTC:
- Cancellation of `9754bb253795` is recorded at 12:10:38 UTC. The launcher still
  reports renting; no worker or CUDA result exists.
- A bounded read through RunPodAPI.named("etude-235b6b5cad35") completed and found
  no matching pod. This is provider inventory evidence, not final launcher closure.
- Etude 0.2.15's `_observe` retains a create with no receipt until the reserved
  rental deadline, even after cancellation. The retained deadline is 12:49:37 UTC
  (one-hour rental reservation at $0.59/hour; the job itself is capped at .75 h).
  Do not mutate the launcher database or treat its interim time estimate as an
  incurred bill. The existing 30-second waiter remains alive (PID 37680).
- The next speed step is the exact same-device original/candidate CUDA parity
  and timing test after the canceled request settles. No replacement was rented.
  Keep the candidate unmerged until that result exists. This is a provider /
  launcher dependency, not a pending user decision.
- #433 is now merged at 0aa7dabbdd6df0a1a2bc70ec66c38a6abd2005c1. The current
  checkout and timing diagnostic still use the recorded old source; integrate
  the new world before final delivery, preserving original timing provenance.

## Speed measurement retry — 2026-11-09 UTC

The previous request `9754bb253795` is terminal: lost/create_unconfirmed, $0.00.
No speed result was produced. Fresh supply reads found no L4; A4000 initially
reported available at $0.25/hour, then quote reported no stock before admission.
Prepared one queued retry, `c5fba4-vocabulary-batched-retry2-20261109`, on A4000,
with the same $0.75 / .75 h / one-attempt job limits. Its three arms are original
catalog, batched catalog and hashed on identical inputs, balanced forward/reverse
order on one device. The hashed arm measures the remaining gap on that device;
do not compare its absolute times to the previous L4. CUDA output/gradient
agreement runs before timing. Source stays bf408ec48c20 plus the literal candidate
in Job.command. Scripts: scratch/launch_vocabulary_batched_retry.py and
scratch/vocabulary_batched_retry_probe.py. The candidate has not changed.

Retry admitted as `fd27b31b55f6`. Fresh status: queued, waiting for a4000, $0.00;
controller heartbeat 10 seconds old. No worker started yet. Disposition: capacity.

## Catalog timing result and integration — 2026-11-09

Run fd27b31b55f6 succeeded; attempt 07be30dd64cf, A4000 a249180d7c84.
Rental deletion confirmed. Final allocation $0.008823643767472050602152061084
(estimated=false; this is the job allocation, not a whole-rental invoice).
CUDA outputs and parameter gradients agree (output rtol1e-5/atol1e-6;
gradient rtol1e-3/atol1e-5). Medians in seconds:

| Batch/path | Original catalog | Batched catalog | Hashed |
|---|---:|---:|---:|
| 64 inference | .129467 | .078993 | .086707 |
| 64 forward/backward | .330716 | .228379 | .225751 |
| 256 inference | .473786 | .273782 | .324452 |
| 256 forward/backward | .986963 | .681879 | .682813 |

Batching reduces inference time 39–42%, forward/backward 31%. Batch-256 inference
profile: CUDA synchronizations 3699 -> 1823; nonzero calls 1452 -> 84;
kernel launches 15034 -> 9386. Fresh weights and fixed random-play positions;
no optimizer, transfer or engine timing. This is not whole-training throughput.
Retained outputs: r2://manabot/jobs/fd27b31b55f6/attempts/07be30dd64cf/.
Local verified copies: .runs/c5fba4-20261109/. timing.json SHA256
 d9f8214fbf866bd8f33d41f0f1fbbbba34c9e85e76536badc33016f8aabe9e75.

Sync found main's newer implementation from 3a395dde5 (#421), which batches
catalog lookups as part of full shared-effect admission/projection/reduction
batching. Kept that implementation unchanged; our narrower implementation is
superseded. Our extra catalog output/gradient regression test remains. Do not
attribute the above timing to main's different implementation.
A bounded integrated diagnostic is being admitted (same .75 USD/.75h/one attempt)
with original source bf408ec48c20 and main's shared_effects.py plus records.py
embedded under isolated module names. Both optimized catalog and hashed use the
new shared encoder; original catalog is retained for CUDA parity and baseline.
This isolates the code change on the same old-world inputs without rebinding
frozen evidence. Scripts: launch_vocabulary_integrated.py and
vocabulary_integrated_probe.py in scratch.

Jack Heart: "ok so shoul we start with catalog rather than hashed".
Recommendation: catalog with batched implementation, given parameter savings,
the controlled runtime result and no clear learning winner. This question is
not an instruction to launch the long runs. Integrated measurement, current-world
calibration and execution review still precede those runs.

Integrated current main and rebuilt the native extension atomically. Focused
catalog/shared-effect/record-coverage/active-training suite: 28 passed, 2 skipped
(24.05 s). Ruff and diff whitespace checks pass. Shared-effect production code
matches main; only the extra catalog batch output/gradient regression test is
new relative to main. The managed sync also retained three unrelated Wave files;
do not remove or publish them as part of the speed fix without resolving ownership.

## Launch direction — 2026-11-09

Jack Heart: "lets get it going im eager to see how good the agents are".
Proceed with catalog frontier training at both agreed sizes within the existing
budget, after completing required calibration and fixing/freezing the runner.
This supersedes the prior pending execution-review boundary: do not ask again
for the same launch. Keep regular strength evaluations during training. Preserve
historical preflight evidence and all failure/spend accounting.

Jack Heart: "one thing im interested to do is trying in the 10x10 grid against a
model that only trains in that 10x10 grid". Existing grid is Elemental's ten
decks; capture a separate broad-versus-grid-specialist comparison, preserving
frontier launch scope and budget. Proposed controls: matched architecture and
compute, same 100 cells with paired seats/deals; own budget before execution.

Jack Heart: "and then otherwise interested to do evals agaisnt the
trianed-half-as-much as the main measure of progress". Before scoring, make
same-agent half-training comparison the primary frontier progress measure.
Operator interpretation, stated in conversation: half elapsed training hours,
consistent with this experiment's wall budget, using the latest saved checkpoint
at or before that half-hour mark. Initialization and heuristic scores are
secondary; preserve equal-hours/equal-updates size comparisons as secondary.
Jack Heart: "for that we can look first at against random decks from teh 17lands
set". Apply the main half-training progress comparison first on the existing
seeded random 17Lands deck-pair schedule, swapping decks and seats on paired
deals. Keep the Elemental specialist experiment separate and later.

Current-world calibration submitted at merged source 68812e26e, cohort
`tla-frontier-c5fba4-calibration-batched-20261109`: w64 c157793c38bb,
w160l8 bebed58cfe2b, w64 one-producer d79f1c5042d8. Three five-update jobs,
L4, parallel 2, combined cap $2; source world matches prospective 0798b2c7.
A 30-second Etude waiter supervises these requests. Do not duplicate them.
Elemental specialist idea filed once as MTG-270, unstarted; no new budget.

Integrated encoder probe 65ca9f28d09a / c8c5205d1068 succeeded and its rental
was confirmed ended by Etude after explicit cleanup reconciliation. Verified
outputs are in .runs/c5fba4-20261109-integrated/. Against original catalog,
current shared effects give 5.08–7.53x inference and 2.33–2.43x forward/backward
speedups on one A4000. With BOTH vocabularies using current shared effects,
catalog/hashed ratios are 1.137/1.064 inference and 1.200/1.260 forward/backward
at B64/B256. Told the user this supersedes "essentially tied" for the newer
implementation. CUDA parity passed. Keep catalog; calibrate full training.

Current-world record coverage queued as 6e32551e90aa, A4000, cap $0.50/.5h,
one attempt, merged 68812e26e, 200 games seed0. No local benchmark games.

## MTG-252 design evidence — 2026-11-09

Comment 1d691509-1d1e-435e-87f1-0eaf77c4a4af by Session session_99ffa20715cd497fada9abfce92cbaee for Jack Heart:

MTG-252 final results for MTG-267's training design

The participant requested: "can you also give hte results to mtg267 to affect its training design".

Report with graphs: https://github.com/loopflowstudio/manabot/blob/1631c4c403a62280615df3eab74976aa1125ea46/experiments/value-readout-252-results.md (PR #434). The self-contained graph report is retained as experiments/data/value-readout-252/report.html in that commit's artifact manifest, alongside raw evaluation blocks, score tables and selected training metrics. Local report: /Users/jack/src/manabot.mtg-252-compare-learned-value/experiments/data/value-readout-252/report.html.

Five width-64/four-head arms, three seeds each, completed 300 updates. All 28,200 evaluation games / 7,050 four-game pairings are valid and complete. Update-300 heuristic scores: depth-2 masked mean 58.0%, token 54.3%, attention readout 57.8%; depth-4 masked mean 61.0%, token 61.7%.

Depth-4 token minus masked mean is +0.7 percentage points, 95% interval [-5.5,+6.5]. Their direct match gives masked mean 52.5% [48.3,56.3]. There is no clear readout winner. Extra depth improves token's heuristic score by +7.3 points [1.0,13.2], masked mean's by +3.0 [-3.5,9.0]; neither depth head-to-head excludes a tie. Intervals resample seeds and shared deal blocks; three seeds, repeated schedule and post-hoc extension make this exploratory.

The design-relevant caveat: equal-update advantage is not established per hour. At 4.298 recorded collection+learning hours, interpolated scores are depth-2 mean 57.0%, token 53.2%, attention readout 57.5%; depth-4 mean 56.8%, token 57.6%. At equal 13.46M transitions the depth-4 arms score 59.3%/58.4%. These are interpolated descriptive comparisons, not fresh evaluations or controlled throughput claims. No retained update reports floor binding or an empty actor batch. High critic explained variance (~.94) is against bootstrapped lambda targets, not eventual wins.

Training-design implications to incorporate at the next design/freeze review:
- Keep the same masked-mean critic across the two model-size arms unless another explicit hypothesis requires a readout change. This result does not justify adding a token/readout axis.
- Preserve equal-training-hours heuristic and head-to-head milestones and calibration-based damping horizons. Report transitions and optimizer exposures too; update count alone hides different data/work budgets.
- Treat deeper capacity as a candidate, not proof that eight layers beats four or that the larger TLA arm is more cost-efficient. Keep the current two-arm scope and execution-review gate.
- Preserve MTG-267's own source/world and catalog calibration. Elemental/v6 evidence does not establish transfer to TLA/newer inputs. One seed per model size is a case study, not replicated superiority.

No additional rental, training launch, change to frozen evidence or default promotion is requested by this handoff. Please incorporate the findings into the existing training design rather than expanding the experiment.

<!-- loopflow-requester:"Jack Heart" -->

<!-- loopflow-session:session_99ffa20715cd497fada9abfce92cbaee -->

<!-- loopflow-steer:1d691509-1d1e-435e-87f1-0eaf77c4a4af -->

Incorporation: retain masked mean in both arms, the two-size scope, current-world
calibration, half-training primary and equal-hours secondary comparisons. Retain
transition and optimizer-exposure metrics for the outcome report. No readout
axis or new rentals. This evidence handoff does not revoke the November 9
frontier launch instruction.

Record-coverage run 6e32551e90aa succeeded and rental ended. It visited 21,127
random-play decisions in 200 games; 335/335 eligible cards were dealt and all
observed fact/effect ids mapped. 109 games ended at capacity limits. These are
world-wide synthetic decks, not the frozen 17Lands evaluation population, so
that abort fraction is not a frontier failure-rate estimate.

The elapsed-time reader and primary half-training comparisons are implemented.
Cohort counts now deduplicate match identities shared by multiple chart slots;
each slot retains its own score and failure bounds. Source remains unfrozen
until current-world calibration/rehearsal and fitted horizons are available.
