# Vast admission diagnosis — 2026-11-04

## Trace-delivery continuation — 2026-11-05 UTC

Fresh 01:55:50 accounting confirms all eight preceding Task rentals absent.
The sum of each reported/lifetime maximum plus settled L4 is $0.93290824;
the existing $3.50 compute and $6 ancillary reserves cover it. Diagnostic
`b153cc2e5a77` reserves another $0.50, leaving $15.00 under the original $25.
Source `2b2061c8d` and locked Etude `64e9eaabf` are unchanged. The live quote
is $0.2474074074/hour and $0.18555556 for 30 minutes work plus 15 setup.
Instance `59191189`, session `c8987cae4da6`, was acquired at 01:56:29.
All four retained CUDA oracle comparisons passed (maximum absolute error
7.6294e-5). One of six routing tests failed: pooled parameter gradients differed
in 1/192 entries by 1.3351e-5, exceeding default float32 atol 1e-5. No learner
started. Both output files (4,854 bytes) matched the delivered inventory. The
rental ended at 01:59:30; fresh inventory confirmed absence, charge $0.013
versus lifetime estimate $0.01240595. These are not final invoices.
Evidence is manifest-published under `trace-delivery-failure/` in the GPU study
data directory; local operations remain `.runs/representation/gpu-trace-delivery/`.

`af601877e` compares both float32 paths against independent float64 outputs,
object/header gradients and parameter gradients, with 2e-5 relative/absolute
bounds, retaining duplicate membership and poisoned-padding cases. CPU pool/GRU
pass; CUDA still requires execution. No runtime encoder, scientific recipe or
frozen receipt changed. Failed routing reports now enter the carried log too.
Fresh 02:01:57 inventory found all nine prior rentals absent; reported/lifetime
maxima total $0.95190824. The $4 compute/$6 ancillary reserve plus a new $0.50
diagnostic commitment leaves $14.50. Run `911423cce90b`, source `af601877e`,
uses the same $0.18555556 whole-rental quote and is starting on instance
`59191723` (RTX3090/13.33 vCPU/257.509 GB).

That run succeeded: all four frozen output/gradient comparisons and six CUDA
routing cases passed. Pool/GRU each completed 256 games, with collection/learning
42.56/40.15s and 42.38/39.95s. All 59 output files (170,244,006 bytes), including
both inference and complete learning traces, match their worker hashes. The
39-byte result envelope was reconstructed from the frozen formatter and transported
update clock after the SDK requested an absent standalone object; its original
SHA256 matches. All other files were fetched normally. Evidence and derived
trace analysis are manifest-published under `cuda-traces/`. The rental ended
02:08:54; 02:10:42 inventory confirmed absence. Reported $0.030 exceeds the
$0.02577188 lifetime estimate; no final invoice is claimed.

The first learning minibatch's effect/continuation host scopes sum to
13.09ms pooled and 15.45ms GRU; CUDA stream waits inside those scopes sum to
0.099/0.093ms. Across the profiled update, cProfile attributes about 9.15/9.59s
to these encoders. Host routing/dispatch remains measurable, but stream waits
do not dominate these captured scopes. First-step kernel loading (133/137ms
in the learning operator tables), profiler work and sampled gradient/parameter
health prevent extrapolating trace totals as recurring costs. GPU annotation
spans include gaps; they are not utilization or sums of kernel execution.
No further runtime repair is selected before ordinary throughput admission.

Admission `37c1523a98ed` uses the identical `af601877e` source, four updates per
arm, $14 scientific allowance and $0.26 hourly supervision ceiling. Fresh
02:11:19 accounting confirmed all ten prior rentals absent; reported/lifetime
maxima plus settled L4 total $0.98690824. Prior compute reserve $4.50, new
commitment $0.50 and ancillary $6 leave $14.00. Whole-rental quote $0.371111
for 75 minutes work plus 15 setup fits the attempt cap. Instance `59192537`,
session `46dcc516f702`, reports the same RTX3090/13.33vCPU/257.509GB hardware.
Both arms completed four updates/1,024 games. Their exact collection/learning
times are 136.4431/122.2212s (pool), 132.4886/111.1156s (GRU), with 516.7714s
overall admission. All 49 files (131,228,103 bytes) verified and published under
`batched-admission/`; the result envelope's frozen bytes matched its hash as
above. Rental ended 02:24:19 and was confirmed absent at 02:25:05; reported
$0.052 versus lifetime estimate $0.05139499.

The complete phase projection is 55.66970 rental hours / $13.77310. Before the
result, accounting was corrected to release unused completed-attempt ceilings:
reserve max($3.50, twice fresh per-rental reported/lifetime maxima plus settled
L4), $6 ancillary, and all unknown/live commitments. At 02:25:26 all eleven
prior rentals were absent; the reconciled total was $1.04590824, leaving
$15.50 for science. No final settlement or new allowance is implied.

The matching live quote admitted scientific Run `e51f61b92931` at the same
`af601877e` source. Instance `59194544`, session `2b7ebc4aa353`, has the same
RTX3090/13.33vCPU/257.509GB hardware. Its rental ceiling is $13.77310 and
deadline 2026-11-07 10:09:07 UTC. The full $15.50 allocation remains reserved.
At 02:34:17 UTC the provider confirmed this as the sole Task rental, running.
The carried log reaches `train: typed_pool seed 74211`, proving preceding
CUDA numerical/hardware gates returned success and the first scientific child
was launched. Its completed update count is not carried in the parent log;
there is no final scientific inventory or plateau result yet.
`batched-plateau/` contains the prospectively frozen allocation/source/quote;
the protocol retains the unchanged seeds, horizon, endpoint schedule and criteria.
Etude's documented `Execution.finish` requested delivery and self-deletion after
the sole job; the worker acknowledged `finish when done` at 02:32:32 UTC.
The independent deadline remains a backstop. The local wait client was stopped
after this handoff; eventual final inventory, analysis and teardown remain open.

## Complete-trace attempt lost — 2026-11-05 UTC

`6c0e33d4b` extends the operator trace from the first grad-enabled forward through
backward, clipping, Adam and numerical health. Inference has its own bounded
trace; failures retain a `.partial` trace. Two focused tests establish actual
operator coverage and exact weights/gradients across two ordinary updates.
`66ec957fe` adds both reducers' directed CUDA routing/absent-gradient checks.
The pre-batching CPU oracles and all four retained CPU profiles are now
manifest-backed under this study's `encoder-reference/` and
`cpu-operator-profiles/` data directories. CPU oracle agreement is unchanged.

Run `a46e80b31212`, source `66ec957fea9fe94d6ed919e2e971fb4f63d3f468`,
used four digest-verified reference inputs. Its Japan RTX3090 quote was
$0.2474074074/hour, $0.371111 for 75 minutes work plus 15 minutes setup,
within the $0.50 cap. Instance `59185580` started at 01:03:07 UTC.
Fresh accounting had confirmed the seven preceding Vast rentals absent.
The runtime installed locked Torch 2.10.0+cu128 and Etude `64e9eaabf`.

The retained stdout reaches both profile children, proving the preceding
four CUDA reference comparisons and six CUDA routing tests returned successfully.
Their detailed JSON/test reports were not delivered. Both one-update profiles
completed 256 games: pool collection/learning 46.67/45.86s; GRU 48.01/43.53s.
The separate uninstrumented pooled child then completed one update/256 games,
41.45/30.29s. These are rounded durable progress clocks, not complete TrainingRun
receipts, matched four-update admission or recoverable checkpoints.

At 01:12:47 UTC Etude recorded `lost / machine_lost` and deletion, with no
final exit status or output inventory. Provider absence was confirmed independently,
including the 01:16:18 reconciliation. No scientific seed or score ran. The
bucket contains only the incomplete attempt record and contiguous job-log chunks;
those bytes were recovered separately from the cached controller log. No traces,
full CUDA parity report, learner receipt or throughput result can be recovered.
The cause is unknown: no evidence establishes OOM, a host defect or code failure.
Intermittent local storage-read timeouts do not establish the cause of deletion.

The new reported charge is $0.038 versus $0.03984744 whole-lifetime estimate;
reported-plus-settled compute is $0.89258259. These are not final invoices.
The $3.50 compute and $6 ancillary reserves leave **$15.50** within the original
$25. Task-local inventory uses recorded Etude session labels (provider labels are
`etude-<session>`, not Task names); all eight Vast instances are absent. No new
rental or science was started after the loss.

The failed combined job exposed an evidence-delivery boundary: this SDK uploads
ordinary output files only when an attempt finishes. The maintained diagnostic
now finishes immediately after the two profiles (one update target, 30-minute
work bound), so its inventory can be delivered before a separate throughput job.
It also prints bounded parity/test and operator summaries into the carried log.
This changes no learner or scientific recipe. Full traces still need a successful
fresh diagnostic; the code change is not proof of delivery on another host.

Attempt evidence is published under
`experiments/data/state-representation-gpu-5da5f4/complete-operator-profile/`;
`.runs/representation/gpu-complete-profile/` retains local operational scripts.
Remaining acceptance: retained complete CUDA operator evidence, complete matched
uninstrumented admission, and the unchanged scientific cohort only if affordable.
The loss does not revoke the authorized bounded diagnosis, replenish the budget,
change the horizon, justify a model decision or satisfy the plateau objective.

Focused check: two operator tests and six CPU routing cases passed; six local
CUDA cases skipped. The remote routing gate returned success before learning,
but its report is unavailable. Ruff and the profile-only declaration check pass.
Gate owns broader suites; no publication/merge or model promotion is performed.

## Earlier CPU repair and connectivity evidence

`42709fa1b` introduced ordinary-executor profiling; `59ec1bd24` replaced per-record
CUDA dispatch with host routing admission, batched projections and reducers.
Matched pooled CPU work (256 games, 51,095 decisions, 200 minibatches) fell from
202.86s collection/learning to 121.71s; encoder call time fell from 30.36s to
2.71s. Both used cProfile, the final additionally captured operators. Native
multi-position outputs and all parameter/object/definition gradients agree with
pre-change oracles. Duplicate memberships, canonical ordering, padding and None
gradients for unused Adam parameters survive. The GRU CPU run completed 256 games,
54,184 decisions and 212 minibatches in 150.12s without a matched CPU baseline.
These are CPU engineering observations, not GPU affordability or learning quality.

The initial profiler captured only grad-enabled forward; `6c0e33d4b` expanded it
through backward, clipping, Adam and health. Earlier GitHub/Doppler outages stopped
preparation before allocation; no new compute charge or scientific seed resulted.
The later complete-trace attempt and successful delivery above supersede that
connectivity block. Full commands, contrary interim timing, references and prior
validation remain at `af601877e:scratch/vast-host-diagnosis.md`; no frozen evidence
was rewritten. Manifest-backed CPU profiles and encoder references remain under
the GPU study's `cpu-operator-profiles/` and `encoder-reference/` directories.

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
