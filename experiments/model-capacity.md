# Capacity: early progress and a bounded terminal allocation

ETU-103 software preparation, 2026-10-05. The original three-capacity WDL
study below remains proposed and unexecuted. Jack Heart subsequently authorized
the separate mini depth screen recorded here.
No default-model or promotion-model decision is supported yet. Keep the existing
default unchanged until this comparison supplies evidence. ETU-91 and the frozen
ETU-106 value-token checkout are untouched; their evidence is not capacity data.

## Authorized mini depth screen (2026-10-06)

Jack Heart selected capacity as the first self-contained mini experiment,
superseding the filter-scope selection. ETU-105's worker was interrupted before
mini training. The laptop history campaign in ETU-106 stays untouched. This
allocation is eight hours on `jack@100.96.227.95`, checkout `/Users/jack/src/etude`;
no laptop training, paid compute, retry, evidence replacement or Task completion.

`depth-screen` reuses the greedy-only screen supervisor, TrainingRegime,
TrainingRun, EvaluationProtocol and replayed arena. It compares `depth-1` and
`depth-2`: width 64, heads 4, scalar value-token, no history. Everything except
attention depth and recipe ID is identical, including authored Allies/Lessons,
semantic input, float32 CPU/one thread, current self-play, Adam, actor/critic
quantile filtering, four streams × 64 transitions and the move-learning recipe.
This deliberately departs from the older WDL three-capacity proposal. Extra depth
changes both policy and value representations. Paired seeds do not imply shared
weight bytes, identical training trajectories, or equal inference cost.

The supervisor charges preflight/calibration ≤1,800 s, training ≤21,600 s,
evaluation ≤4,500 s and reporting ≤900 s to one monotonic 28,800 s ceiling.
Calibration seed 10340 runs 40 updates per arm, split 20/20, with raw/EMA reload
and 400 s per-process / 190 s per-stage caps. Both timing receipts must complete.
No score enters allocation: rate is the maximum of process seconds/40 and each
stage seconds/20 across both arms. The shared terminal count is
`min(1600, 100 * floor(21600 / (6 * 1.25 * rate * 100)))`; below 400 stops the
attempt. Three paired training seeds 10341/10342/10343 use alternating orders
1→2, 2→1, 1→2. Each run continues Adam/collector/EMA across two equal halves,
with 3,600 s process and 1,790 s half-stage caps. Fixed count is the screen's
terminal allocation, not convergence or the full-capacity study endpoint.

Before scientific training the supervisor writes exact recipes, seeds, counts,
source/native/environment/input/world identities and timing receipts to
`resolved-plan.json` and `calibration.json`. Clean-source checks precede each
phase. A two-hour feasibility check and between-run timing checks preserve the
remaining phase reserves. Child process groups are killed on deadline/disk/failure.
The allocation never expands in response to scores or partial runs. All failed
attempts and bytes remain retained. The mini's existing native binary is reused
when native sources match; PR240/242 `.runs` evidence is preserved.

Raw midpoint and endpoint checkpoints each play 100 games per seed-arm versus
source-pinned greedy: common deals 963410–963434, all four seat/deck legs. The
1,200-game cohort uses paired seed/deal resampling with each four-leg block intact;
EMA exports remain unscored. Both checkpoints use these screening deals; there is
no untouched promotion cohort. All games require exact replay, ordinary legality
and viewer-boundary admission. Invalid or incomplete cohorts stop without a
comparative disposition. No score-dependent stopping or checkpoint selection.

Existing notebook/report output includes strength versus cumulative training
seconds (collection/export included), native decisions, learner transitions and
optimizer exposures. Common-support areas and contrasts never extrapolate; two
scheduled checkpoints cannot establish a detailed early learning curve or exact
time to useful progress. The descriptive effect is depth 2 minus depth 1. The
existing screen rule recommends a larger confirmatory allocation only with ≥5
percentage-point endpoint gain, no negative seed effect, positive common-cost
effect and ≤1.25× pooled inference latency. Consistent losses of ≥5 points reject
only this recipe/budget; other outcomes are unresolved. Neither decision promotes
a default model. Bootstrap seed 10634 and 10,000 draws are analysis constants.
Exposure comparisons are descriptive post-treatment accounting.

Launch the delivered clean source using the shared supervisor:

```bash
uv run --extra dev --extra notebook python -m experiments.runners.run_history_input --study depth-screen --campaign COMMIT --out .runs/etu103-mini-depth-20261006-1
```

The detached launch retains stdout, parent/child PIDs, phase receipts and budget
in that output. `supervisor.json` owns total charged time including calibration,
training, evaluation and reporting; nested `study/study.json` records evaluation
only. Do not add those totals. TrainingRun retains sampled resource observations;
host load is observation, not a correction for contention. `study/report.md`,
`study/history-contrast.{json,md}` and the executed notebook retain the shared
report schema; arm labels identify the depth comparison. Launch/result status is
recorded separately below; this protocol is fixed before execution.

## Executable plans

```bash
uv run python -m experiments.runners.capacity_study --write-plan /tmp/capacity-plan.json
```

This exports an ordinary ResolvedStudy and adjacent `.provenance.json`. It does
not construct a model or execute a workload. The default is a 900-second,
one-seed workflow plan with two tiny checkpoints, not a scientific allocation.
The declaration uses ETU-109's pinned Ataraxos baseline and ETU-102's existing
64/1, 64/2, 128/2 ladder, four attention heads. Only width and depth vary:
semantic viewer inputs, value-token pooling, categorical WDL, action capacity,
move-learning rule, Adam, batch/stream sizes, self-play opponents and raw artifact
evaluation stay fixed. Layer/width changes affect the shared policy/value core.
No history, pooling or learning-rule feature is crossed into this comparison.

The existing executor is the only execution path. After separate authorization:

```bash
uv run python -m experiments.runners.run_training_regimes --study model-capacity --profile scientific --plan /path/capacity-plan.json --out /path/new-capacity-run
uv run python -m experiments.runners.run_training_regimes --report-only /path/retained-capacity-run
```

Report regeneration verifies retained protocol, recipe, run-export, checkpoint
and trace digests. It writes the existing metrics/report/uncertainty/notebook plus
`capacity-analysis.json` and three-axis plots. Install the notebook extra for
report execution. Failed attempts remain in the ordinary study and VerifyStore;
partial cohorts do not become a model-selection result. No alternate runner,
checkpoint adapter, score computation or W&B dashboard is introduced.

## Proposed calibration and scientific protocol

First reserve an uncontended laptop window with ETU-91's operator; no worker may
infer resource availability from host load. Run a separately authorized bounded
pilot using the same token baseline, one CPU thread, fixed native world and
source, recording actual model receipts, complete-run costs, sampled process-tree
RSS, host load and any interruptions. Existing concurrent historical-mean
capacity-calibration timings cannot calibrate these token arms. Suggested pilot
ceiling: 30 minutes total, including evaluation/recovery. Preserve every failure.
If even the pilot cannot fit, stop and revise before scoring.

Freeze conservative update counts per arm targeting **at most five minutes**
per unit from the pilot, including collection, learning and checkpoint export.
The authoring input is `CapacityWorkload` JSON: `updates_per_unit` (three positive
integers in ladder order), `calibration_path`, its `calibration_sha256`, and
`prior_campaign_seconds` including all earlier allocations/attempts. The digest
binds the reviewed calibration evidence; the exporter does not certify its
scientific adequacy. Export with `--workload /path/workload.json`. Retain that
file and the authoring receipt with the plan. Also supply `runtime_identities`
with the seven ordinary calibrated runtime/source digests, `projected_disk_bytes`
and `projected_evaluation_seconds`. The exporter checks that evaluation plus maximum
training and a 30-minute report/recovery reserve fit the 24-hour ceiling. Freeze code, runtime/world, content,
setup, ABI and storage projections before the separately authorized launch.

Proposed full allocation: three capacities × three independent training seeds
1031/1032/1033 × 12 calibrated units. Checkpoints follow 1, 2, 4 and 12 cumulative
units. Stage watchdogs are 300, 300, 600 and 2400 seconds; each run has a
3660-second total ceiling. **Full training means the final frozen update count**,
not convergence or exactly one elapsed hour. The common per-arm budget is about
one hour, with 60 seconds of setup allowance. Unused time is not filled with
extra updates. Slow arms may fail; do not replace them or increase limits after
seeing scores. The initial 20-minute progress window and final endpoint are
separate estimands. Early traces remain useful if the final comparison is negative.

The proposed study process/allocation cap is 24 laptop hours (roughly 9.15 hours
maximum training plus evaluation/report reserve), within the unchanged 168-hour
campaign ceiling after charging prior attempts. Calibration and storage/resource
projections must show the evaluation fits before allocation approval; otherwise
reduce the proposed cohort before freezing or retain a bounded inconclusive pilot.
This document grants no allocation. No paid compute is proposed.

At each checkpoint, 25 common deal blocks × four seat/deck legs yield 100 games
per cell. All three fixed anchors (random, scripted greedy, PUCT-64) and paired
small-arm versus each larger arm comparisons reuse EvaluationProtocol. Untouched
endpoint deals use disjoint reserved families and paired training seeds. Both
seats and deck assignments stay together in uncertainty resampling. The existing
paired-seed/common-deal bootstrap supplies descriptive intervals; three seeds
remain a small cohort. One-thread, one-pass inference fixes the execution rule,
not equal latency or FLOPs across capacities. Record those costs; never claim
matched wall-clock inference strength without that qualification.

## Analysis and decision rule

The preregistered descriptive initial-progress threshold is score ≥0.60 against
the fixed random anchor within 1200 measured training seconds. First observed
crossings retain the preceding observation and first hit, not an interpolated
exact time. No hit is right-censored at the last observed checkpoint in that
window; no observation is unavailable. Sparse monitoring and score noise mean
these are scheduled observations, not confidence-certified or monotonic learning
crossings. Monitoring neither stops training nor selects held-out deals.

Curves show score against cumulative training wall time (including collection),
native environment decisions, and optimizer sample exposures. Repeated minibatch
samples are exposures, not decisions or optimizer steps. Each axis uses the
intersection of observed supports across all frozen arms/seeds. The last available
checkpoint supplies a step function; no pre-first-point backfill or post-last-point
extrapolation. Early mean score/area clips that shared time support at 1200 s.
Empty overlap and historical absent exposure counters remain unavailable. Failed
cohorts keep diagnostic crossings/terminal rows but suppress comparative curves.
Collection/export costs stay in measurements; study comparisons retain evaluation
and replay costs, with replay a subset. TrainingRun retains identities, phase costs,
resource observations and all attempts. Report total actual charged study time,
not only optimizer duration, when interpreting the later experiment.

Terminal observations use only the last checkpoint on untouched endpoint deals;
monitoring scores cannot replace endpoint evidence. Compare complete endpoint
cohorts with seed/deal uncertainty and actual training/inference costs. Prefer a
fast experimental model only when early progress and common-support area agree
across seeds without a material competence/legality regression; name a promotion
model only from terminal evidence. Conflicting estimates or intervals compatible
with no useful difference mean inconclusive, not a throughput-based winner.
Repeat a promising or important-negative learning idea at the next capacity in
a separately frozen follow-up, leaving its architecture-feature owner unchanged.
No such follow-up or empirical default selection has run in this software pass.

## Step-target recovery preparation — 2026-10-08

Jack Heart raised the overall capacity-experiment ceiling to **$100**, including
prior attempts, storage, validation and new allocations, and authorized autonomous
step-target protocol preparation. This supersedes the historical $30 ceiling.
ETU-126 remains the single launch owner. Frozen source `68e0fbe9`, its failed
attempt, and Mini training/ordering remain unchanged. New numerical semantics
require a new source and attempt; no CUDA process continuation is available.

ETU-103's read-only contribution prepared seven ordinary Experiment/JobSpec
declarations in ETU-126's retained `.runs/etu126-recovery/proposal/`, including
`protocol-v2.json`, `declaration.py`, source/input hashes and copied frozen
controls. All seven compile without training or provisioning; exact control
comparison permits only declared endpoint, identity, clock and allocation changes.
They remain **proposals, not admitted jobs**. Final source/protocol and timing-based
targets must be frozen before any scientific scoring; compiled proposal bytes
must not be submitted as final plans.

| Declaration | Proposed target | Allocation | Per-job cap |
| --- | ---: | ---: | ---: |
| Large validation, seed 10350 | 512 iterations | 2.5 rental hours | $1.75 |
| Small, each of three paired seeds | 26,000 iterations | 9.5 rental hours | $4.95 |
| Large, each of three paired seeds | 9,000 iterations | 9.5 rental hours | $4.95 |

An iteration includes collection and any empty-filter optimizer skip. Each retains
64 streams × 8 transitions; counts are not optimizer exposures. The small target
projects 4.13 active hours from 17,283 completed updates / 9,884.38 seconds. The
large target projects 4.23 hours from three retained L4 updates, **two of which
skipped optimization**; charging the observed optimization time on every update
instead projects 5.76 hours. These are weak timing projections, not promises of
four hours or equivalent realized cost. Preserve iteration-based Ataraxos rates
and measured exposures; do not rescale schedules to the new targets.

The 512-iteration large pilot must pass the named CUDA numerical gate, ordinary
raw/EMA reload, initial and final development cohorts (100 replayed games each),
source/native identity, artifact publication and confirmed deletion. Its scores
cannot choose targets. Inspect timing, memory and optimizer exposure before
prospectively finalizing the six-run targets; do not infer sustained performance
from mostly skipped updates. A failed pilot retains its costs and stops admission.

The six scientific jobs keep seeds 10351–10353, alternating arm order, original
monitoring/final/random deal reservations and both deck/seat assignments. Stage
watchdogs are 7.5 hours; final cohorts and persistence have separate lease reserves.
Hourly exports now use elapsed run time less export time, including initial waits;
they are not historical active-hour checkpoints. Report both clocks, samples,
exposures and costs. Compare paired seeds and complete deal blocks, use observed
common cost support without extrapolation, and suppress complete-method results
when any intended run or required cohort is missing.

The provisional inclusive reservation is **$46.432386989310**: $4.282386989310
retained estimated costs, $1.75 validation, $29.70 scientific job caps, $0.70
extra guardian reserve, $3 durable storage, $2 controller/reporting and $5 for
unreconciled validation costs. The original proof's $3 reservation is replaced by
its actual $0.115845 charge. Conservative reserves are not additional invoices;
charge every actual attempt once. The remainder is not automatic retry authority.

Read-only live admission on 2026-10-08 observed empty RunPod inventory and L4/A40
quotes of $0.49/hour. These expire as admission evidence and must be queried again.
AWS control-plane SSO was expired; the restricted issuer authenticated but denied
control reads, as intended. Mini was online without ready controller credentials.
No credential was broadened, service installed, or rental started. Renewable
controller authority, final source, fresh prices and successful CUDA validation
remain launch gates. The earlier protocol below remains frozen historical intent.

## Authorized CUDA comparison — 2026-10-06

Jack Heart superseded the twelve-hour total comparison on 2026-10-06 with
**four active training hours per capacity per seed**: two capacities, three
paired seeds, at least 24 training hours plus overhead. Collection and learner
updates count; setup, exports, evaluation-only time, transfer and reporting do
not. A run stops after the first complete update crossing 14,400 active seconds.
An exhausted update safety ceiling or watchdog records failure, never a shorter
successful run. CUDA process restart remains unsupported.

The shared initial $15 includes ETU-123's up-to-$3 proof reservation. Jack Heart
permits a calibration-supported working cap of $30; the larger requested cohort
requires it. Each rental stays below $5. Prior calibration charges, proof charges,
setup, idle time, storage and deletion remain charged. ETU-114 and ETU-91 budgets
are unavailable; historical mini evidence stays separate.

Calibration first compares width64/depth2/heads4 to width384/depth8/heads4 with
feedforward width1536, scalar value-token, no history, identical current-self
Ataraxos learning and selected-match inputs. Sixteen updates per arm (two linked
eight-update stages), seed10350, establish memory, end-to-end timing, artifact
return and CPU arena feasibility. Predictions: the large arm fits 24 GiB GPU
memory, costs at least twice as much per update, and scores no more than five
percentage points above the small arm in the early window. The latter is a
scientific prediction, not a calibration selection rule. Calibration inspects
no strength aggregates. Allocation/VRAM/admission failure stops that attempt;
a smaller width128/depth2 fallback needs explicit timing-based admission and
retains the large-arm failure. No score-based capacity choice or retry.

Three paired scientific seeds are 10351–10353. Exact counts and clocks must be committed before scoring. Initialization
and scored milestones each receive at least 100 balanced greedy games; random
diagnostics have their own schedule, and final deals stay untouched until endpoint
scoring. Evaluation must run during training. A flat curve prompts review, not an
automatic failure verdict. Full-budget means the frozen affordable endpoint;
it does not mean convergence. The original remote CLI returned artifacts only after training. Deploy lifecycle
integration and live evaluator admission remain prerequisites to this cohort.

### Hardware sweep amendment, before the new probes

Jack Heart assigned the two-card performance sweep to ETU-103 on 2026-10-06.
The initial L4 probes remain unchanged: at source `b6bbd1a4`, both capacities
completed 16 updates / 16,384 learner transitions, returned verified exports,
and completed four exact-replayed CPU arena games each. Rental deletion was
confirmed for both; estimated rental costs were $0.0546896941 and $0.0747118629.
Training times were 97.721 and 154.289 seconds. These are short feasibility
observations; no strength aggregates were inspected. CPU evaluation process times
were 45.318 and 38.949 seconds on one paired deal. Game length confounds that
comparison. The configured 256 transitions are **per stream** (four streams),
so the observed update batch is 1,024, unlike the historical mini screen.

The next two jobs use NVIDIA L4 (24 GB class) and NVIDIA A40 (48 GB), in that
order, each with a 2,100-second client allowance, existing pod deadline and
$0.50 deployment cap. Current live quotes for both were $0.49/hour; each launch
must re-admit price. The two-hour calibration clock began with the first retained
rental at Unix 1791328605.990033 and includes this preparation and CPU work;
a new job refuses admission if its full allowance no longer fits. The initial
$15 aggregate ceiling retains a $1 storage/report reserve. No cap increase is
currently justified. Unused allowance does not authorize retries.

The shared 256-row real selected-match input was collected with seed10349 and
four CPU streams, without optimizer work. Its exact serialized SHA256 is
`719fc91a7eb69395303a5ba11cb560336b6c3fc81ae12e0e57e46f8d593cf7a3`.
Both cards receive those same bytes. Model-only cells cross all four existing
capacity rungs with batches 1, 4, 16, 64, 256 and 1024. Batches above 256 cycle
the fixed real rows; no padding is removed. Record cold construction/first call,
two warmup calls and three one-second timed windows for inference and diagnostic
Adam (uniform legal-policy cross entropy plus squared scalar value, all rows).
CUDA synchronization bounds timings; float32 and disabled TF32 are explicit.
These optimizer probes are not Ataraxos updates or useful RL throughput.

Complete-loop cells independently cross every capacity with 4/16/64 streams and
128/512 total learner transitions per update, three ordinary updates each. The
first update is cold; the following two expose steady update coordinates.
Learning/filter/schedule settings remain the retained scalar-token recipe.
TrainingRun/VerifyStore own phase times, actual exposures, skipped updates and
exports. One worker and one CPU thread do not imply one stream. Both cards use
identical cells; report each card's best observed feasible configuration separately,
with these short-run and selection limits. No strength-based configuration choice.

Each cell has a 70-second child cap; each full grid a 900-second cap within its
rental. All OOMs, partial phases, timeouts and unvisited cells remain explicit.
CUDA allocation/reservation peaks, attention slots, validity/padding, source,
native/runtime and input identities are retained. Model-only improvements cannot
stand in for complete-loop improvements. Collection includes both engine and
inference, so its fraction alone does not establish engine starvation; relate it
to fixed-batch inference and stream sweeps before proposing new collection code.
No separate software Task is warranted without a demonstrated missing capability.

CPU coordination: ETU-118's calibration subsequently acquired the shared lease;
a fixed-input attempt declined it without collecting data. ETU-103 waited and
preserved that owner. Both four-game checks and the final input collection used
the shared lease. Concurrent sustained campaigns still require a tested handoff;
no learner is displaced and no timing correction is invented.

### Bounded CUDA learning schedule (before scoring)

The two-card probes remain pinned to `7316da5d`. The L4 grid completed 47 cells
and retained one optimizer OOM at width384/depth8, fixed batch1024. Its closed
277-file, 1,735,865,605-byte bundle verified through an independent bulk return.
The redundant per-file transfer was deliberately interrupted after verification;
the original interrupted receipt remains, deletion was confirmed, and its
intent-to-deletion rental estimate is $0.1903964333. This is transfer recovery,
not a repeated measurement or a rewritten successful deployment. The A40 receives
the identical grid/source. Bulk transfer and explicit phase timestamps are being
added for later jobs; historical phase splits remain unavailable where not recorded.

### Superseded short comparison proposal (never launched)

The fixed-count schedule below is historical. The four-active-hour amendment
replaces its endpoints, clocks and allocation; retained calibration is unchanged.

Timing-only selection would use one common measured GPU, batch and stream count
for both scientific capacities. Width64/depth2 and width384/depth8 remain the
comparison; the small model is scalar value-token, no history, with the same
inputs, Ataraxos learning, filtering, schedules, Adam and current-self opponent.
The fixed terminal count per capacity is
`20 * floor(900 / (1.5 * measured_seconds_per_update * 20))`, split into two equal
linked halves. The rate is the maximum of the selected three-update run/3,
stage time/3, and either of its last two update intervals. Fewer than 20 admitted
updates stops allocation. The 50% timing margin is conservative extrapolation,
not a sustained timing proof. Unused time buys no extra updates. Different counts
are declared cost-targeted endpoints; sample/exposure comparisons use overlapping
observed support. Each run has a 1,000-second watchdog and 490 seconds per half.

Three paired seeds 10351–10353 run small/large, large/small, small/large. Six
1,850-second rental ceilings reserve 11,100 seconds. Each run has four separate
100-game cohorts, each with a 1,200-second CPU process ceiling: initialization
and midpoint versus greedy on development deals 1910103510–1910103534; terminal
raw versus greedy on untouched deals 1910103610–1910103634; then a separately
scheduled terminal random diagnostic on 1910103710–1910103734. Four deck/seat
legs stay together. The 24 cohorts reserve 28,800 evaluator seconds; publication
and reporting reserve 1,800 seconds. These sum to the revised 41,700-second
comparison ceiling described below. Rental and evaluator occupied time are charged separately
even when they overlap. Publication during a rental is a subset of its time;
post-rental publication/reporting consumes the final reserve. Transfers, setup,
idle and deletion stay inside rental time. The $15 ceiling includes all earlier
calibration rentals and a $1 storage/report allowance; no increase is justified.

Initialization is exported without advancing learner or sampling RNG state.
Immutable exports become available to the ordinary CPU checkpoint queue while
training continues. The existing shared evaluator lease is acquired before any
scientific rental; an ETU-118 owner prevents admission without displacing its
learner. W&B is a bounded projection and S3 retains original producer bytes with
transport path resolution. Local evidence survives publication failures. A
checkpoint-safe pause is between complete independent seed/arm rentals only;
there is no remote process recovery or automatic retry. Any training, replay,
legality, cohort deadline or required artifact failure retains the attempt and
stops the comparison. A flat curve alone does not stop it.

The early window is 600 training seconds and the descriptive progress threshold
is 50% score versus greedy. Report first observed crossing intervals or censor
at the last observed non-hit. Common-support step-function area uses development
points only, without extrapolation; initialization and one midpoint give a sparse
curve, not precise learning onset. Final deals are plotted distinctly and never
reclassified as development evidence. Report raw terminal contrasts with separate
seed, paired-deal and joint percentile intervals (10,000 resamples, seed1035106).
Three training seeds remain a small method cohort. One CPU thread and one sampled
policy pass fix the inference rule, not equal latency or FLOPs across capacities.
Retain the current fast-model default unless consistent early gains justify a
follow-up; a promotion-capacity recommendation needs a terminal gain of at least
five points with positive paired-seed interval and no negative seed contrast.
Such a recommendation is input to ETU-118, not automatic baseline or chapter
promotion. Negative or inconclusive outcomes retain both scientific limits and
all attempts.

Before freezing this schedule, one final live-workflow calibration may use at most
1,250 remaining seconds of the revised calibration clock: a 1,100-second rental,
128 small-model updates, one four-game initialization timing cohort, and bounded
collection profiling. The profiler compares the existing 4/64-stream settings
at 512 total transitions for both capacities. It records actual inference batch
sizes and the union of traced GPU kernel/copy intervals within collection;
profiler overhead makes it diagnostic, separate from throughput. GPU gaps do
not by themselves identify the engine as the cause. No collection software Task
is justified merely by high collection share or one execution worker.

A timing-only fit-bracketing amendment adds width384/depth8 at fixed model batch512
on each card, using the same real input, two warmups and three timed windows.
The original 48-cell grids remain unchanged. A40 runs this separate 70-second
child after its grid has closed, within that rental's existing deadline; L4 runs
it in the final live-workflow calibration. Supplemental manifests retain source,
results, exit/OOM logs and costs. This closes the untested gap between batches256
and1024 without changing a scientific model or examining strength scores.

### Calibration time reallocation (2026-10-06, before original deadline)

Following Jack Heart’s instruction to report a concrete revised time plan before
exceeding calibration, Codex reported a 25-minute reallocation at Unix time
1791335743: 8,700 calibration seconds and 41,700 comparison seconds. Combined
time stays 50,400 seconds and the all-in ceiling stays $15. The calibration
deadline is 1791337305.990033. This is an operator interpretation of that
instruction, not a claim of separate approval of these exact phase numbers.
The additional time funds only live evaluation, collection profiling, the L4
batch512 bracket and freeze. Another overrun stops execution. Six reduced rental
limits absorb the reallocation; evaluation and reporting reserves stay unchanged.
The immutable local amendment retains the original clock and completed attempts.

### Retained hardware result and launch boundary (2026-10-06)

Both pinned 48-cell grids completed 47 cells and retained the large-model
optimizer OOM at fixed batch1024. A40’s separate batch512 optimizer completed
with 26.159 GiB peak allocated memory. The corresponding L4 batch512 probe and
collection profiler did not run: the final live-workflow attempt at `e052ac30`
failed the existing 90-second guardian SSH bootstrap deadline before training.
Its original TimeoutError and confirmed deletion remain retained; no retry ran.
The A40 primary per-file return failed with scp exit255 after an independently
verified bulk return. It was not deliberately stopped. Both recovered bundles
retain 277 verified files, with original primary failure receipts unchanged.

At the identical total-batch512/64-stream workload, three-update run times were
L4 small/large 3.455/9.539 s and A40 4.576/10.990 s. L4 was faster in these
short observations at the same admitted $0.49/hour GPU quote. The A40’s larger
fixed optimizer batch fits memory unavailable on L4, but the missing L4 bracket
prevents a measured batch512 comparison. These are short timing and fit results,
not sustained throughput or playing strength. Wider existing streams improved
collection throughput; no missing collection capability has been demonstrated.

Five rental attempts have confirmed deletion. Intent-to-deletion estimates sum
to $0.5359766177 including the failed final guardian probe ($0.0131126277).
Provider billing reconciliation is not available. Private complete receipts,
models, logs and bundles remain under `.runs/etu103-cuda-capacity`; preserve this
checkout. The compact [extract](data/cuda-capacity/capacity-report.json), editable
[notebook](data/cuda-capacity/report.ipynb) and read-only
[HTML report](data/cuda-capacity/report.html) retain measurements, source hashes,
all attempts and unavailable quantities. The compact notebook regenerates the
report without model bytes; full extraction requires the retained private root.

At this historical software boundary, Jack Heart reserved $3 of the existing $15 for ETU-123’s bounded
disconnect proof, leaving at most $12 for ETU-103. Jack Heart requires its
delivered disconnect-safe lifecycle before the longer scientific comparison.
PR253’s original live-export callbacks were client-owned. Integration now uses
PR254’s remote supervisor and rejects those historical client callbacks. The scientific CLI now fails explicitly at launch.
No resolved scientific plan or scientific cohort was frozen or launched. The
prospective schedule above remains a proposal requiring lifecycle integration,
live-workflow proof and remaining-budget admission; the elapsed calibration
clock must not silently reset. ETU-123 owns stable remote job identity, remote
evaluation/upload, reconnection and remote shutdown; PR253 supplies bulk bundle
admission, phase receipts and client-observed exports for reuse. No default or
promotion model is selected from hardware throughput.


### Four-hour preparation and admission — 2026-10-07

At frozen source `68e0fbe9265a689891615274123587d653b5e66b`,
`cuda_capacity --prepare-four-hour` exported ordinary DeploymentPlans from the
existing Experiment baseline and retained calibration input. It prepares L4,
512 transitions per update over 64 streams, width64/depth2 versus
width384/depth8/feedforward1536, scalar value-token, no history. All learning,
input, optimizer and opponent controls are identical. Ataraxos retains its
one-based iteration schedules and clamps; active time determines the endpoint,
not a new learning-rate denominator. One million updates is a safety ceiling,
not the scientific endpoint. Full stage watchdog is five hours, with a one-minute
run setup reserve. These are preparation outputs, not admitted scientific plans.

Initialization and the first completed updates at approximately one, two and
three active hours use 25 development deals × four seat/deck legs versus greedy.
The final raw export at four active hours uses 25 untouched final deals × four
legs, followed by 100 separately scheduled random diagnostic games. The final
stage export suppresses a duplicate hourly monitoring export. EMA remains
retained and unscored. Seeds 10351–10353 and the existing three disjoint CUDA deal
namespaces remain unchanged because no scientific scoring has occurred. Early
progress is the first observed 50% greedy score during the first active hour;
non-crossings are right-censored. Hourly sampling cannot resolve faster crossings.
Paired seed/deal uncertainty, active time, total elapsed cost, decisions and
optimizer exposures remain separate axes; three seeds support exploratory claims.

A fresh RunPod read at Unix 1791343703.70 quoted both L4 and A40 at $0.49/hour
and returned zero pods. The proposed six seven-hour rentals cost at most $21.42
including $0.02/hour storage. Add retained ETU-103 $0.5359766177, the full $3
ETU-123 reservation, $0.60 guardian reserve and $1 durable-storage reserve:
**$26.5559766177**, below $30. The six job caps remain $4.50 each. Seven hours
includes 15 minutes setup, a five-hour learner watchdog, evaluation tail, ten
minutes transfer and three minutes cleanup. Six serial rental ceilings plus one
hour reporting bound the new allocation at 43 hours; prior attempts retain their
own elapsed charges. Admission must re-read quotes, actual shared proof charges
and guardian cost before rental, without treating unused reserves as retries.

Each run budgets six 100-game cohorts, 1,800 seconds each, on one remote CPU
alongside training. The earlier four-game process timings were 45.3175 and
38.9487 seconds: linear 100-game projections are 1,133 and 974 seconds. The
1,800-second allowance is a projection with margin, not a measured 100-game
remote bound; CPU hardware, game length and replay can change it. Remote
100-game timing/live publication admission remains required. The rented six-CPU,
48-GiB declaration leaves a CPU for evaluation; shared memory/CPU contention
must be reported. No laptop evaluator or mini worker is displaced.

Read-only credential discovery found only the default AWS SSO profile, no AWS
keys in Doppler `etude/prd`, no account OIDC provider, and no Roles Anywhere
trust anchor in us-west-2. The `manabot-remote-jobs` role already permits 43,200
seconds, but [AWS role chaining](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html)
still caps a session obtained from SSO role credentials at one hour. Increasing
that role limit or relaxing the local guard cannot authorize a seven-hour job.

On 2026-10-07 Jack Heart authorized a dedicated IAM issuer with only
`sts:AssumeRole` permission on the existing worker role, its key stored in
Doppler `etude/prd`, and matching narrow additional trust preserving the original
SSO trust and unrelated policies. A matching existing issuer/key must be
reconciled before creation, without duplicate keys on retry. The dedicated
profile uses `credential_process`; issuer keys never enter pods, artifacts,
logs or chat. Each issued session retains the single-job-prefix restriction
and must cover the full rental deadline before rental admission.

The initial 2026-10-07 setup attempt stopped on expired SSO, before any IAM
mutation or rental (`cf8c581f`). After Jack Heart renewed SSO, the dedicated
`manabot-remote-issuer` was created with only AssumeRole on the existing worker
role. One key was stored and read back in Doppler; the `manabot-issuer` profile
uses credential_process. Original SSO trust and the worker storage policy remain.
The initial trust update met IAM propagation delay; retry reconciled the same
user and created no duplicate key. A 25,200-second worker session plus the issuer
reserve passed expiry admission and STS identity verification on 2026-10-07.
The private receipt is `.runs/etu103-issuer-admission-20261007.json` (no keys).

A live read at Unix 1791394251.337 again quoted L4/A40 at $0.49/hour and found
zero pods. The conservative $26.5559766177 plan remains valid, including the full
$3 proof reservation rather than spending its unused balance. Actual historical
shared spend remains $0.6518216. No new rental is part of credential verification.

**Initialization admission amendment, fixed before scoring:** each real run's
required 100-game initialization cohort also establishes its remote evaluation
cost. The learner exports its exact initial checkpoint, then waits before creating
the collector or taking an optimizer step. The remote supervisor releases it only
after a complete cohort within 1,800 seconds and successful durable publication.
A failed cohort stops that attempt without learning or a replacement rental.
A failed upload leaves the gate closed under the original watchdog/deadline.
This reuses the scheduled initialization cohort on the same rental; it adds no
smoke rental or scored games. The earlier four-game timings justify the bounded
initial admission attempt, not a claim of measured 100-game remote performance.

Admission wait is excluded from active collection/learning, included in the
five-hour stage watchdog and seven-hour billing deadline. The projected half-hour
initial evaluation leaves four training hours and half an hour stage overhead;
final evaluations retain the remaining rental tail. All six jobs use this same
gate. Initialization scores never select capacity or alter learning/endpoints.
If overhead exhausts a watchdog, preserve the failed attempt and stop the cohort;
CUDA cannot resume. Initialization and final evaluation are serial with learning;
hourly evaluation shares the allocated remote CPUs and memory without a throughput
correction. Neither ETU-118 nor the mini gets a laptop evaluator.

PR254's deploy supervisor owns execution, the evaluation gate, checkpoint queue,
S3 generations and deletion. Reconnecting cannot restart training. PR255 delivery,
the final exact source/protocol freeze and per-rental live admission precede
submission. The old client-driven scientific launcher stays disabled. The editable
hardware notebook and private bundles remain; no capacity winner exists.

Current authoring uses `Experiment.jobs` with `PlannedRun(spec=JobSpec(...))`;
see [machine allocations](../docs/remote-jobs.md#step-targets-and-machine-allocations).
The calibration-era `cuda_capacity --freeze` authoring helper is retired; its
source and two-stage protocol remain in Git at `33f68e07`. Saved schema-1 plans
remain readable. This migration changes no frozen cohort or evidence.
