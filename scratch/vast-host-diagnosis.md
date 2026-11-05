# Vast admission diagnosis — 2026-11-04

## Operator repair and connectivity block — 2026-11-05 UTC

`42709fa1b` adds `scripts/profile_representation.py`: one excluded ordinary
256-stream update, retained TrainingRun/exports, cProfile and one Torch operator
trace each for inference and learning. CUDA declaration uses existing Etude
source binding, progress supervisor, graceful stop and rental deadline; no
synthetic load or alternative training loop. Its remote execution is unverified.

The original CPU batch diagnosis (seed 74200, pooled, width64/depth2, one thread)
completed 256 games, 51,095 learner decisions and 200 optimizer minibatches.
Collection/learning: 107.43/95.42s; encoder calls: 30.36s; index validation:
823,864 calls; reducers: 70,402 calls. Small smoke profiling had understated the
batch cost (about 0.3–0.4s in the encoders), so it was not used for attribution.

`59ec1bd24` replaces per-record device dispatch with host routing admission and
batched live-token projections, flat segmented sums/packed GRUs, and membership
scatter. Numeric features and object/catalog joins still carry gradients. It
retains all validation, canonical order, source/parent joins and duplicate edges.
Unused continuation projections retain None gradients, important for Adam's
momentum behavior. Checkpoint parameters and fixed input contracts are unchanged.

The final pooled CPU diagnosis completed the same game/decision/minibatch counts:
collection/learning 64.31/57.40s (121.71s versus 202.86s, about 1.67x local speed),
encoder calls 2.71s versus 30.36s. Both used cProfile; the final run additionally
captured two operator traces. An interim pass took 117.44s total but preceded the
absent-gradient repair and is not the final comparison. Saved pre-change native
multi-position oracles agree on outputs and parameter/object/definition gradients
for both treatments. This establishes a CPU dispatch improvement, not CUDA
synchronization cost, a GPU speedup, or learning superiority. The final GRU
diagnosis also completed: 256 games, 54,184 decisions, 212 minibatches;
collection/learning 77.51/72.61s. It has no matched full-batch CPU baseline.
Both final diagnoses exported checkpoints and retained inference/learning traces.

Fresh 23:21:22 UTC accounting found all seven Task Vast instances absent and
reported-plus-settled compute $0.8545825939. Existing $3 compute/$6 ancillary
reservations remain. The compatible 3090 quote was $0.2474074074/hour, $0.19
for the bounded diagnostic window, capped at $0.50 (temporarily leaving $15.50).
Launch stopped before any run/rental record: Etude's remote-history Git fetch
stalled and was terminated. A subsequent bounded exact-ref publication also
timed out; GitHub SSH and HTTPS were unreachable, and Doppler fell back after
its API timed out. The second script never reached quote/launch. No provider
allocation was requested, so that diagnostic commitment is released and $16
remains under the existing conservative accounting, not a new allowance.
Remote ref publication is unknown; that cannot create a rental. No final bill
or fresh provider inventory after the outage is claimed.

Evidence remains local (network prevented artifact publication):
`.runs/representation-profile-batch256-before/`,
`.runs/representation-profile-batch256-after/` (interim),
`.runs/representation-profile-final-pool/`,
`.runs/representation-profile-final-gru/`,
`.runs/representation-parity/`, and
`.runs/representation/gpu-operator-profile/` (quote/accounting/failure logs).
No retained experiment receipt was changed. Remaining acceptance is CUDA
forward/backward profiling/parity and matched two-arm GPU throughput, then the
unchanged cohort only if complete costs fit. Gate owns broader suites.

Focused validation: 28 encoder tests passed across semantic routing, batched
forward/backward equivalence, padding/order, absent gradients and checkpoint
reload selections; strict saved native oracles agree for both treatments; Ruff
and diff checks pass. The two final CPU learner diagnoses completed normally.

Continuation, Jack Heart, 23:15 UTC: continue the diagnosis into bounded
operator-level profiling and a justified implementation repair, then matched
throughput admission toward the original GPU plateau study. The current cost
projection does not close that remaining implementation work. Any necessary
diagnostic rental comes from the existing cumulative $25, with fresh accounting,
a bounded quote and confirmed cleanup; no budget increase or scientific scope
change. Task comment `f10b962c-1310-435c-a46d-22c9532820aa` records the direction.

Jack Heart's 22:38 UTC direction: diagnose our launch/training implementation
before assigning host fault, correct fixed-cost extrapolation and add prospective
startup/progress/affordability stops. Original $25 total and $0.50 admission cap
remain; no competing rental, scientific shortening, merge or promotion.
Direction: Task comment `aece955c-e64d-4f51-9336-311898578948`, Session steer
`ab0fe717-1f66-4671-8aec-a802ac927a60`.

## Completed admission, scientific objective still unmet

Run `803bcc26e7de`, source `22c10ca55197ab225546e5880cc53709db985901`,
Etude lock `64e9eaabf016f567cb9567280bcf8730f6f5d524`, succeeded at
22:56:05 UTC. Both excluded seed-74200 learners completed four updates and
1,024 games on the same RTX 3090, CUDA float32, one actual thread, 256 streams.
No scientific seed or score ran. All 46 output files (131,204,931 bytes) match
the worker's SHA256/size inventory. Etude's courier embeds `etude-result.json`
in the attempt record but its fetch path requests an absent standalone object;
that 39-byte file was recovered from transported clocks using the frozen JSON
formatter and matches the original hash. All other files were fetched normally.

Instance `59168556` was terminated through Etude after completed delivery;
provider absence confirmed 22:58:28 UTC. Its reported charge is $0.205 versus
$0.20159 whole-lifetime quote estimate; no final settlement claim. Combined
reported-plus-settled compute is $0.85458 using the prior 21:59 read. Conservative
reserves still leave $16 for science. Evidence is manifest-backed under
`experiments/data/state-representation-gpu-5da5f4/cheaper-3090/`.

## Measured work and limits of attribution

| Host wall seconds / count | Pool | GRU |
|---|---:|---:|
| Executor total | 1,286.32 | 1,419.93 |
| Executor setup | 0.34 | 0.34 |
| Collection | 843.07 | 969.56 |
| Policy inference/action selection, inside collection | 720.10 | 790.30 |
| Environment/encoding, inside collection | 36.51 | 37.40 |
| Batch assembly, inside collection | 38.82 | 97.26 |
| Learning | 440.84 | 448.04 |
| Forward/loss, inside learning | 227.27 | 218.96 |
| Backward/clipping, inside learning | 143.58 | 155.24 |
| Export | 0.61 | 0.58 |
| Learner decisions | 186,007 | 188,735 |
| Optimizer minibatches | 728 | 740 |

Complete admission elapsed is 2,715.66 seconds: 9.41 seconds beyond the two
executor clocks. Setup/native installation and numerical checks precede this
measurement; the whole rental lasts 48.94 minutes. Whole rental less admission
is about 220.63 seconds for setup, numerical checks, delivery and cleanup, not
measured learner idle time. In-stage unclassified bookkeeping is 1.45/1.40s.
The repeated inference/learning phases dominate; fixed overhead does not explain
these rates. GPU sampling during GRU showed 23% utilization, 18,510 MiB memory,
126 W of 250 W, 61 C, 1695 MHz SM; this is one sample, not a utilization trace.
Learner CPU was approximately one core. Actual cgroup quota was 6.4 cores,
versus quoted 13.33; two CPU-pressure samples were zero and cumulative throttling
did not increase between them. Peak learner RSS was 17.55/17.34 GB, below its
24 GiB cap. Neither OOM nor sustained host contention is established.

The frozen source audit identified per-batch/per-frame loops, `.tolist()`, scalar tensor
reads and repeated CUDA validation in `TypedContinuationEncoder` and
`SharedEffectEncoder` as plausible CPU-dispatch/device-synchronization costs.
These operations occur in both inference and learning. No operator-level GPU
profile isolated their contribution at that source. The later CPU repair above
does not establish CUDA attribution. The older 4090 receipt differs in both training and engine source
hashes; it cannot isolate hardware speed even though several recipe/ABI hashes
match. A causal comparison needs identical source/workload and retained profiling.

## Corrected affordability and prospective controls

The former projection multiplied all two-arm elapsed time by the update ratio.
The replacement separately charges recurring collection/learning/bookkeeping
(50% allowance), per-learner initialization/close/export, seven intermediate
exports bounded by each admission export bundle, and one setup/evaluation/teardown
reserve. Unknown outer admission time is charged once per paired seed. Frozen
receipts are unchanged; `phase-projection.json` is a separate derived analysis.

Corrected full-cohort projection is 290.27 hours / **$71.82** at the actual quote.
Recurring training alone, without any allowance or fixed/evaluation costs, is
**$47.58**, versus $16 remaining. Four updates do not guarantee later game length,
but this measured arrangement cannot fund the declared study. Plateau remains
unmet; no replacement science launched.

Future admission binds limits in its job config: 180s to a ready CUDA/one-thread
record, a no-completed-update window derived from four times the affordable mean
update pace (minimum 300s), and after at least two updates a phase-cost rejection
if this arm alone exceeds the whole scientific allowance. These are capacity
policies, not scores or host-defect labels. Durable progress and stop reason
precede a child stop-file request; 120s grace permits evidence flush before hard
process-group reaping. Ordinary nonrecoverable collection now consumes the
existing executor stop/resource callback. This code did not govern the frozen
completed rental and has only fixture validation so far.

Focused validation: 12 projection/watchdog/process tests and two executor
cancellation tests passed; Ruff passed. Native extension rebuilt to match current
source before the executor tests. Gate owns broader suites. Remaining work after
the CPU repair is CUDA profiling/parity and matched bounded admission within the
same allowance; the four-paired-seed cohort,
72 curve endpoints, eight held-out endpoints and plateau criterion remain open.
