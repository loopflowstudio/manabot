# PPO CUDA updates (MTG-338)

**Inconclusive; candidate reverted after measurement pass 10.**
The frozen comparison at `02c202956` could not start. The single L4 availability
check returned “none available”, “0 of 1 in use”, “not rentable now” at
**2026-11-13 11:43:03 UTC**. No launch was attempted. There are **no completed
paired samples**, no CUDA equivalence result and no candidate rate or range.
Measurement restored the runtime, candidate tests and comparison harness to
`734f2cd5c`. No rental, retry or queued work was created. This is not a
performance rejection.

The last measured baseline remains **0.26437 optimizer steps/s** on one NVIDIA
L4 (five repeats, **0.26366–0.26474**), or **60.521 s/update**
(60.437–60.684). Backward and clipping take 76.0% of update time. These are
prior baseline measurements, not results from the unavailable comparison.
The indexed-gradient hypothesis is neither confirmed nor refuted.

This is exploratory throughput measurement, not playing-strength evidence.
The expectation committed before execution was: “Current PPO sustains roughly
0.26 optimizer steps/s on an NVIDIA L4 at the Elemental w64l2 recipe,
256 streams × 64 transitions, both seats, TF32, bucketed shapes and live tokens;
backward and clipping account for roughly three quarters of update time.”

## Measurement and behavior

Clean run `4a289dc30a82` used revision `2202ad632cb504d97155f2fa9268f898c72ab525`.
Its training implementation is unchanged from `6ce618b8a`; intervening changes
restore and instrument the MTG-325 harness and add this experiment launcher.
The locked build used CPython 3.13.8, Torch 2.15.0+cu132 and CUDA 13.2,
TF32, one Torch CPU thread and deterministic cuDNN, matching the production
recipe settings. Seed 25201 initializes the model and collection; each update
uses a fresh NumPy generator seeded with 3 for selection/shuffling.

The Elemental w64l2 workload collects 256 streams × 64 transitions, both seats,
with `elemental-uniform`, bucketed shapes and live tokens. PPO uses
`Learning(reference="action_type_uniform")` at schedule progress 0.3:
50% retained rows, four epochs and four minibatches. Each update starts from
the same initial weights, a fresh Adam optimizer and a deep copy of the same
collected batch. One warmup precedes the five timed updates. Timing includes
the entire update and final CUDA synchronization; copying, weight checks,
JSON writes and profiling are outside it.

| Repeat | Seconds/update | Optimizer steps/s | Optimizer steps | Errors |
| --- | ---: | ---: | ---: | ---: |
| 1 | 60.552 | 0.264234 | 16 | 0 |
| 2 | 60.437 | 0.264740 | 16 | 0 |
| 3 | 60.684 | 0.263660 | 16 | 0 |
| 4 | 60.515 | 0.264397 | 16 | 0 |
| 5 | 60.521 | 0.264372 | 16 | 0 |

All updates consumed 53,341 learner rows, retained 26,671 and performed
106,684 optimizer exposures: 80 optimizer steps and 533,420 exposures across
the five repeats. Optimizer hooks count work inside the timed region and agree
with update diagnostics. There were no rejected or skipped steps and no reported
nonfinite values. Parameter deltas were nonzero; loss ranged from
0.2423583523 to 0.2423593638.

**CUDA outputs were not bitwise repeatable.** The five final state-dict digests
are distinct despite restoring the same initial weights and batch. Maximum
absolute floating-weight differences from repeat 1 were 0, 2.242e-5, 2.486e-5,
1.898e-5 and 1.877e-5. This bounds the observed drift for this batch only; it
neither establishes an acceptable tolerance nor identifies its cause. CPU
bitwise equivalence remains a requirement for a future implementation change.

The easier real workload (8 streams × 8 transitions, 2,035 learner rows)
completed 16 steps in 1.902 s, **8.413 steps/s**, with zero errors. This is one
measured repeat after warmup, with no measured spread. It demonstrates workload
sensitivity, not a speedup at equal work.

## Limiter and source

The separate profile comes from run `e9e5a6bd4649`, revision `71e27e82f`, with
the same runtime implementation, workload and production settings on another
L4 allocation. It records preparation and the first optimizer step after two
unprofiled updates. It is excluded from all five reported timing samples.

Of 2.964 s summed GPU kernel duration, 989 `indexing_backward_kernel<float,4>`
calls account for **2.322 s (78.3%)**. The largest reductions accumulate
many gathered vectors into the **463 × 64** definition table:

| Definition lookup | Incoming gradient shape | GPU kernel time | Source |
| --- | --- | ---: | --- |
| Fact sources | 6668 × 256 × 64 | 1.126 s | `facts.py:193` → `RecordEmbeddings.sources` → `semantic_cards.py:414` |
| Continuation relation definitions | 6668 × 64 × 2 × 64 | 0.542 s | `backends/torch/policy.py:722–724` → `embed_definitions` |
| Visible cards | 6668 × 2 × 60 × 64 | 0.303 s | `model/semantic_cards.py:424` |
| Permanents | 6668 × 2 × 32 × 64 | 0.181 s | `backends/torch/policy.py:663` |
| Decklists | 6668 × 2 × 32 × 64 | 0.088 s | `model/semantic_cards.py:446` |
| Known hand | 6668 × 2 × 8 × 64 | 0.046 s | `model/semantic_cards.py:434` |

Attribution follows kernel/CPU correlation, autograd sequence numbers and the
recorded tensor shapes matched to source; recovered Python source frames were
empty. Two attention-backward kernels total only 0.044 s in this window. This
makes definition lookup gradients the first optimization target, rather than
Adam or recurrent sequence length.

Combined cProfile/operator tracing heavily perturbed host execution: the
profiled update took 315.750 s versus 61.027 s for its preceding unprofiled
update. Its host time in observation gathering is not a steady-state estimate.
The GPU kernel breakdown is a local diagnostic, not a measured whole-update
speedup ceiling.

Clean update section medians provide the end-to-end arithmetic:

| Section | Seconds/update |
| --- | ---: |
| Batch transfer, targets and behavior checks | 0.952 |
| Selection and health diagnostics | 0.024 |
| Minibatches, including the nested sections below | 59.550 |
| Forward and loss | 11.429 |
| Backward and clipping | 45.975 |
| Optimizer step | 0.029 |
| Minibatch diagnostics | 0.007 |

Collection took 28.257 s once, so collection plus the median update is
88.777 s; the update is 68.2% of that path. Removing *all* backward and clipping
would bound the gain at 4.161× for the update and 2.074× for that combined path.
Those are impossible-zero-cost bounds, not predictions. Removing the optimizer
step alone saves at most 0.048% of update time. Section medians are nested and
must not be summed as independent totals.

## Limits and next measurement

Five repeats cover one batch, initialization, process and L4 allocation. They
do not measure between-seed or between-host variation, full training throughput,
evaluation/checkpoint overhead or strength per dollar. That run had no candidate
arm to alternate. The historical 0.256 steps/s is context, not a comparator:
the restored harness omitted production thread and cuDNN settings, corrected
before this baseline.

Planning decision, 2026-11-13 UTC: **iterate**. Replace advanced indexing inside
`SemanticCardEncoder.embed_definitions` with functional embedding during CUDA
autograd, preserving the original CPU/MPS/inference path. This reaches the fact
source and continuation joins responsible for 56.3% of profiled GPU time. Predict
at least 20% less median update time: **≤48.417 s**, or **≥0.33046 optimizer
steps/s**, with identical work. Refute that prediction if the alternating-arm
comparison misses it; reject unexplained behavioral differences. The comparison
uses the launcher's **$4 cap**. This remains an untested exploratory hypothesis.

The design frozen at `02c202956` specifies the one-method change,
CPU bitwise checks, CUDA summation-order evidence and five alternating samples
per arm. Keep a change only when its gain clears measured spread and behavior
holds; the historical baseline drift does not establish a tolerance.

Implementation, 2026-11-13 UTC: the one-method CUDA-autograd embedding candidate
was prepared, with focused lookup tests for values, repeated-ID and row-zero sums,
empty/noncontiguous/scalar indices and upstream producer gradients. CPU, MPS and
inference retain indexing. CUDA numerical admission and the paired throughput
comparison remained unrun at implementation handoff; the candidate had not met
the acceptance bar and was subsequently reverted after rental admission failed.

## Run record and evidence

Runs on 2026-11-13 UTC used the task-specific `mtg338-l4` template at the
available $0.69/hour offer, under launcher caps no greater than $4 per run.
Costs below are the launcher's displayed approximate figures, recorded once.
All four allocations were terminated and have deletion timestamps.

| Etude run | Source revision | Outcome | Launcher cost |
| --- | --- | --- | ---: |
| `8f56c7ef9fe9` | `5853054ab` | Cancelled; no completed timing sample | ~$0.12 |
| `0fd992b8d6c3` | `57362f610` | Cancelled in warmup; no completed timing sample | ~$0.05 |
| `e9e5a6bd4649` | `71e27e82f` | Diagnostic and profile completed; cancelled during later clean warmup | ~$0.26 |
| `4a289dc30a82` | `2202ad632` | Five clean repeats and easier check succeeded | ~$0.13 |

Missing samples are missing, not zero. The second attempt's internal stack
snapshot caught `RecordedObservations.gather` at 120 s; that untuned process
had 75 OS threads and a 5.1-CPU cgroup quota. It prompted the production thread
setting correction, not a runtime optimization. External py-spy inspection of
the first attempt was refused by missing `SYS_PTRACE`. A replacement launch
was refused at the template machine limit before its command started; the old
allocation was terminated before retrying. The local Etude database requires
SDK 0.2.15; control commands used that version without changing the database.

Published evidence is indexed in `experiments/data.manifest.json`:

- `experiments/data/ppo_cuda/baseline/{baseline,easier}.json`: all samples,
  settings, counts, diagnostics, digests and weight differences.
- `experiments/data/ppo_cuda/profile/`: diagnostic JSON, cProfile data/text,
  compressed operator trace, derived kernel summary/source mapping and the
  untuned attempt's original log.

Retrieve with `uv run --extra artifacts python -m manabot.data fetch experiments/data/ppo_cuda`.
Raw artifacts remain untracked. Ruff, Python compilation, both generated job
commands' shell syntax, a bounded one-step profiler smoke check and
`tests/infra/test_experiment_data_untracked.py` passed. No training, model or
engine implementation changed in this measurement step.

## Paired comparison freeze (2026-11-13 UTC)

Expected shipping claim: “PPO updates are at least 20% faster on one NVIDIA L4
at the Elemental w64l2 workload, with the same 16 optimizer steps per update,
CPU bitwise equivalence and bounded CUDA summation differences.” This comparison
starts a new timing baseline because its harness alternates the two methods
on one captured batch. The kept method is recovered verbatim from `734f2cd5c`;
both Python implementations compile against one locked native build.

Before timing, run focused model/training tests and the recovered MTG-325 CPU
output/gradient/trained-weight harness; its comparator now fails on differences.
Five A/B pairs follow one warmup per arm. Two additional audit updates per arm
retain raw first-step gradients, first-step and final weights and policy probes.
They are excluded from timing. Require exact initial policy outputs and first
loss; require every raw gradient element to satisfy abs(error) <= 1e-6 +
1e-4 * max(abs(reference parameter gradient)), first/final weight absolute
error <= 1e-4 and final probability/value error <= 1e-4. These are explicit
one-update admission bounds, not thresholds inferred from the observed baseline
drift and not a claim about long-run training equivalence. Inspect gradient
support/counts, same-arm drift and float64 repeated-sum errors as well; any
unexplained mismatch rejects the candidate. A separate candidate profile follows
all measurements. The launch cap remains $4, with no automatic retries.

## Paired measurement outcome (2026-11-13 UTC)

**Inconclusive, not a performance rejection.** The measurement source is
`02c202956` and the last kept runtime is `734f2cd5c` (same runtime as
`6ce618b8a`). All admissions below failed before a job command started and
before a run was recorded. Requested IDs therefore identify launch attempts,
not completed Etude runs. No rental or workload was left running.

| Requested run ID | Admission outcome |
| --- | --- |
| `d2c75f320999` | `RUNPOD_API_KEY is not set`; no command started |
| `3136e424a2a9` | Manabot Doppler config supplied storage, but no RunPod key; no command started |
| `09737bd788b4` | Etude Doppler config supplied RunPod, but no `R2_ACCESS_KEY_ID`; no command started |
| `4e50a037bd12` | Both configs injected; provider reported no available machines; no command started |

The last admission receipt (`947aca1850c9`, 10:57:42 UTC) says:
“cannot launch on mtg338-l4: provider reports no available machines; etude queue
would wait”. A separate read at 10:58:25 UTC reported an L4 offer at $0.69/hour,
“none available”, “0 of 1 in use”, and “not rentable now”. No queue was created.
There is no launcher cost to report for a recorded run from these attempts.

The available credential injection is nested Doppler execution: `manabot/prd`
provides R2 storage and `etude/prd` provides RunPod. Keep both environments on
launch, wait, fetch and cleanup commands; no secret values belong in evidence.
This is operational context, not a credential or provider configuration change.

The comparison source compiled, its generated baseline/paired shell commands
passed `bash -n`, and Ruff passed for the paired harness and launcher. Those
checks do not establish numerical behavior or performance. The recovered CPU
harness, remote tests, float64 CUDA checks, five alternating pairs, same-arm
numerical audits and candidate profile are **all unrun**. Previous local lookup
checks remain 8 passed / 8 CUDA skips; they do not admit the CUDA change.

Per the measurement keep rule, the unmeasured runtime candidate, its tests,
paired launcher changes and recovered/adapted comparison scripts were removed
from the active tree. Keep the existing baseline report and artifacts. To settle
this hypothesis, restore the candidate and harness from `02c202956`, obtain one
available L4 under the $4 cap, and complete the frozen CPU/CUDA correctness and
alternating-run checks before deciding whether to keep it. No speedup, memory
reduction, behavior equivalence or strength-per-dollar gain is established.

## Measurement pass 2 (2026-11-13 UTC)

All eight recovered candidate/harness files matched `02c202956` byte for byte;
the launch requested that same source revision and unchanged paired workload.
The expected claim and numerical bounds above stayed frozen. The read-only
availability response at **11:08:05 UTC** was: “NVIDIA L4 with at least 6 vCPUs,
48 GB at $0.69/hour (ceiling $0.75)”, “available”, “0 of 1 in use”, “rentable now”.
Availability did not guarantee admission.

The first command failed CLI validation because a cohort cap requires a cohort
name; it did not reach admission. The corrected command used cohort
`mtg338-paired-pass2`, nested Doppler `manabot/prd` and `etude/prd`, `--paired`,
`--revision 02c202956`, `--rent mtg338-l4` and `--max-usd 4`. Requested run
`4bacd2187ed1`, admission `db4502e2006a`, began source preparation at 11:09:15 UTC
and failed admission at **11:10:12.624491 UTC**:

> cannot launch on mtg338-l4: provider reports no available machines; etude queue would wait

The retained receipt explicitly states “job command not started” and “no run
recorded”; a cohort status read found no run. No capacity retry or queue followed,
and this attempt allocated no rental. There is no launcher cost for a recorded
run to report. CPU equivalence, CUDA audits, alternating timings and the candidate
profile remain unrun. Missing samples, error counts and work counts are
unavailable, not zero.

The measure keep rule again restores the three tracked candidate paths and
removes the five recovered comparison scripts, retaining their frozen source
and this evidence. One successfully admitted L4 allocation under the $4 cap,
followed by the frozen correctness checks and five alternating pairs, would
settle the comparison. Another implementation recovery alone would add no
performance evidence.

## Measurement pass 3 (2026-11-13 UTC)

All eight recovered candidate/harness files again matched `02c202956` byte for
byte. The expected shipping claim, workload and numerical bounds stayed frozen.
At **11:16:14 UTC**, the one permitted availability read returned “available”,
“0 of 1 in use”, “rentable now”, for an L4 with at least 6 vCPUs and 48 GB at
$0.69/hour (ceiling $0.75).

The single launch used nested Doppler `manabot/prd` and `etude/prd`, source
`02c202956`, `--paired`, `--rent mtg338-l4`, `--max-usd 4` and cohort
`mtg338-paired-pass3`. Requested run `3ca9e26b83a0`, admission `797c99576241`,
entered source preparation at 11:16:24 UTC and failed admission at
**11:17:14.552355 UTC**:

> cannot launch on mtg338-l4: provider reports no available machines; etude queue would wait

The retained receipt says “job command not started” and “no run recorded”; a
cohort status read returned no run. No rental was allocated, no queue was created
and no retry followed. There is no recorded-run launcher cost to report.
CPU equivalence, CUDA audits, five alternating samples per arm and the candidate
profile remain unrun. Workload failures and work counts are unavailable, not
zero; the operational outcome is one failed launch admission.

The measure keep rule restored the three tracked candidate paths to `734f2cd5c`
and removed the five comparison scripts absent from that commit. Their exact
bytes remain in `02c202956`. The preceding implementation pass's eight CPU
lookup passes and eight CUDA skips do not establish CUDA correctness. A
successfully admitted L4 allocation followed by the frozen correctness and
paired timing checks would settle the claim; another recovery without execution
cannot. No speedup or behavior equivalence is established.

## Measurement pass 4 (2026-11-13 UTC)

All eight recovered candidate/harness files matched `02c202956` byte for byte.
The expected shipping claim, workload and numerical bounds stayed frozen.
The single capacity read, using nested Doppler `manabot/prd` and `etude/prd`,
returned at **11:22:45 UTC**:

> mtg338-l4: NVIDIA L4 with at least 6 vCPUs, 48 GB at $0.69/hour (ceiling $0.75); none available; 0 of 1 in use; not rentable now.

The capacity condition was not met, so no launch, cohort, admission attempt,
rental or queue was created. No recorded-run launcher cost exists to report.
CPU equivalence, CUDA audits, five alternating pairs and the candidate profile
remain unrun; workload error and work counts are unavailable, not zero.
The control CLI used `etude-sdk[s3,eval]==0.2.15`; an earlier help invocation
mistakenly requested the nonexistent distribution version `etude==0.2.15`
and failed dependency resolution without contacting launch admission.

Under the measure keep rule, the three tracked candidate files were restored
to `734f2cd5c` and the five comparison scripts absent there were removed.
The frozen candidate remains recoverable at `02c202956`. A successfully
admitted L4 allocation and completion of the frozen correctness and alternating
measurements would settle the claim. This pass adds capacity evidence only;
it establishes no speedup or CUDA behavior equivalence.

## Measurement pass 5 (2026-11-13 UTC)

All eight candidate/harness files matched `02c202956` byte for byte. The claim,
workload and numerical bounds stayed frozen. The single permitted availability
read, with nested Doppler `manabot/prd` and `etude/prd`, returned:

> mtg338-l4: NVIDIA L4 with at least 6 vCPUs, 48 GB at $0.69/hour (ceiling $0.75); none available; 0 of 1 in use; not rentable now; asked 2026-11-13 11:26:32 UTC.

No launch was attempted because the capacity condition was not met. No rental,
queue or background work was created, and there is no recorded-run launcher
cost to report. CPU equivalence, CUDA audits, five alternating samples per arm
and the separate candidate profile remain unrun. Candidate rate, range,
workload error count and work count are unavailable, not zero.

The measure keep rule restored the three tracked candidate files to
`734f2cd5c` and removed the five comparison scripts absent there. Their bytes
remain recoverable at `02c202956`. This pass is inconclusive: successful L4
admission followed by the frozen correctness checks and paired measurements
would settle the claim. Capacity evidence does not reject the optimization.


## Measurement pass 6 (2026-11-13 UTC)

All eight candidate/harness files matched `02c202956` byte for byte. The shipping
claim, workload and numerical bounds remained frozen. The single permitted
availability read, with nested Doppler `manabot/prd` and `etude/prd` and
`etude-sdk[s3,eval]==0.2.15`, returned:

> mtg338-l4: NVIDIA L4 with at least 6 vCPUs, 48 GB at $0.69/hour (ceiling $0.75); none available; 0 of 1 in use; not rentable now; asked 2026-11-13 11:29:45 UTC.

No launch was attempted; no rental, queue or background work was created.
There is no recorded-run launcher cost. CPU equivalence, CUDA audits, five
alternating samples per arm and the separate candidate profile remain unrun.
Candidate rate, range, workload error count and work count are unavailable,
not zero. This is capacity evidence only, not a performance rejection.

The measure keep rule restored the three tracked candidate files to
`734f2cd5c` and removed the five comparison scripts absent there. Exact candidate
bytes remain at `02c202956`. Verdict: **inconclusive**. Successful L4 admission
followed by the frozen correctness checks and paired measurements would settle
the claim. No further capacity read, retry or alternate GPU was used.


## Measurement pass 7 (2026-11-13 UTC)

All eight candidate/harness paths matched `02c202956` byte for byte. The
shipping claim, workload and numerical bounds remained frozen. The single
permitted availability read, using nested Doppler `manabot/prd` and `etude/prd`
and `etude-sdk[s3,eval]==0.2.15`, returned:

> mtg338-l4: NVIDIA L4 with at least 6 vCPUs, 48 GB at $0.69/hour (ceiling $0.75); none available; 0 of 1 in use; not rentable now; asked 2026-11-13 11:33:45 UTC.

No launch was attempted and no rental, queue or background work was created.
There is no recorded-run launcher cost. CPU equivalence, CUDA audits, five
alternating samples per arm and the separate profile remain unrun. Candidate
rate, range, workload errors and work counts are unavailable, not zero.

Verdict: **inconclusive**. The measure keep rule restored the three tracked
candidate files to `734f2cd5c` and removed the five comparison scripts absent
there; exact candidate bytes remain at `02c202956`. Successful L4 admission and
the frozen correctness and paired timing checks would settle the claim.
No retry, further capacity read or alternate GPU was used. This capacity
result establishes neither a speedup nor a performance rejection.


## Measurement pass 8 (2026-11-13 UTC)

All eight candidate/harness paths matched `02c202956` byte for byte. The
expected claim remains “CUDA embedding lookup reduces median PPO update time
by at least 20% on the Elemental w64l2 L4 workload, with identical CPU behavior
and bounded CUDA rounding.” The workload and numerical bounds stayed frozen.
The single permitted availability read, with nested Doppler `manabot/prd` and
`etude/prd` and `etude-sdk[s3,eval]==0.2.15`, returned:

> mtg338-l4: NVIDIA L4 with at least 6 vCPUs, 48 GB at $0.69/hour (ceiling $0.75); none available; 0 of 1 in use; not rentable now; asked 2026-11-13 11:36:42 UTC.

Verdict: **inconclusive**. No launch was attempted; no rental, queue or
background work was created. There is no recorded-run launcher cost. CPU
equivalence, CUDA audits, five alternating samples per arm and the separate
candidate profile remain unrun. Candidate rate, range, workload errors and
work counts are unavailable, not zero. The prior baseline remains 0.26437
optimizer steps/s (five repeats; 0.26366–0.26474); it is not a paired comparison.

The measure keep rule restored the three tracked candidate files to
`734f2cd5c` and removed the five comparison scripts absent there. Their exact
bytes remain at `02c202956`. Successful L4 admission followed by the frozen
correctness checks and alternating measurements would settle the claim.
No retry, further capacity read or alternate GPU was used. Capacity evidence
establishes neither a speedup nor a performance rejection.


## Measurement pass 9 (2026-11-13 UTC)

All eight candidate/harness paths matched `02c202956` byte for byte. The
expected claim remains “CUDA embedding lookup reduces median PPO update time
by at least 20% on the Elemental w64l2 L4 workload, with identical CPU behavior
and bounded CUDA rounding.” The workload and numerical bounds stayed frozen.
The single permitted availability read, with nested Doppler `manabot/prd` and
`etude/prd` and `etude-sdk[s3,eval]==0.2.15`, returned:

> mtg338-l4: NVIDIA L4 with at least 6 vCPUs, 48 GB at $0.69/hour (ceiling $0.75); none available; 0 of 1 in use; not rentable now; asked 2026-11-13 11:39:54 UTC.

Verdict: **inconclusive**. No launch was attempted; no rental, queue or
background work was created. There is no recorded-run launcher cost. CPU
equivalence, CUDA audits, five alternating samples per arm and the separate
candidate profile remain unrun. Candidate rate, range, workload errors and
work counts are unavailable, not zero. The prior baseline remains 0.26437
optimizer steps/s (five repeats; 0.26366–0.26474); it is not a paired comparison.

The measure keep rule restored the three tracked candidate files to
`734f2cd5c` and removed the five comparison scripts absent there. Their exact
bytes remain at `02c202956`. Successful L4 admission followed by the frozen
correctness checks and alternating measurements would settle the claim.
No retry, further capacity read or alternate GPU was used. Capacity evidence
establishes neither a speedup nor a performance rejection.


## Measurement pass 10 (2026-11-13 UTC)

All eight candidate/harness paths matched `02c202956` byte for byte. The
expected claim remains “CUDA embedding lookup reduces median PPO update time
by at least 20% on the Elemental w64l2 L4 workload, with identical CPU behavior
and bounded CUDA rounding.” The workload and numerical bounds stayed frozen.
The single permitted availability read, with nested Doppler `manabot/prd` and
`etude/prd` and `etude-sdk[s3,eval]==0.2.15`, returned:

> mtg338-l4: NVIDIA L4 with at least 6 vCPUs, 48 GB at $0.69/hour (ceiling $0.75); none available; 0 of 1 in use; not rentable now; asked 2026-11-13 11:43:03 UTC.

Verdict: **inconclusive**. No launch was attempted; no rental, queue or
background work was created. There is no recorded-run launcher cost. CPU
equivalence, CUDA audits, five alternating samples per arm and the separate
candidate profile remain unrun. Candidate rate, range, workload errors and
work counts are unavailable, not zero. The prior baseline remains 0.26437
optimizer steps/s (five repeats; 0.26366–0.26474); it is not a paired comparison.

The measure keep rule restored the three tracked candidate files to
`734f2cd5c` and removed the five comparison scripts absent there. Their exact
bytes remain at `02c202956`. Successful L4 admission followed by the frozen
correctness checks and alternating measurements would settle the claim.
No retry, further capacity read or alternate GPU was used. Capacity evidence
establishes neither a speedup nor a performance rejection.

- 2026-11-13: L4 and A40 stock were unavailable; the paired comparison runs on an
  RTX A6000 (on-demand `mtg338-a6000`), so it supports a relative speedup only.
  Etude run `bb2863fe318c` (~$0.04) failed before measurement: four
  `test_active_training.py` cases call `git rev-parse` in Etude's non-git source
  copy; the other 40 CUDA-required tests passed. The rerun drops that file from
  the remote test list; candidate, harness and bounds are otherwise `02c202956`.
