# Shared evaluation compute

Reconciled — 2026-10-24: explicit complete-block continuation is the implemented
deliverable increment after PR #370. The accepted two-hour cloud startup gate
remains required follow-up work in Etude; it is not part of the completed
continuation claim. Its installed 0.2.4 capability gap is detailed below.

Accepted for pursuit — 2026-10-24. Jack Heart approved the design and said
“all sounds good. lets pursue”. Deliver the complete single-machine shared
inference and performance-report keystone through its internal slices in one PR.
The named later increments remain follow-ups. Hardware, numerical/performance
benchmark thresholds and spending remain unselected; no paid execution is
authorized. Standalone work; no Task or Wave selected.

The runtime/report foundation and generic local API remain implemented against
the upstream sim/eval split. Sync merged main `66b8fb3e7` through `003df4e6f`,
bringing Etude 0.2.4 and explicit checkpoint identities; restored continuation
and activity-chart changes are retained. Broad validation and representative
CPU/CUDA performance acceptance remain gates. The accepted browser overview and
its existing proof are recorded in `eval-perf-demo.md`; no measured whole-job
gain or capacity choice is claimed.

## What to build

A budgeted evaluation pool that keeps frozen models resident, batches decisions
from concurrent games, and spends its capacity on an inspectable, programmable
priority queue while preserving each comparison's evidence.

Jack Heart: “it really just makes sense to think about it as just one giant block
of eval compute”. Priority must be a “controllable API on teh queue, with some
sort of language for expressing default priorities”; long studies can have a
“more wide ranging priority order” while currently training checkpoints receive
“a kind of constant middle-level priority”. Batching inference is the first
optimization focus. These select product intent, not a machine purchase or launch.

Additional direction, Jack Heart, 2026-10-24: performance monitoring should
“match and exceed what we have put in the latest perf report prototype for
training” and explain “where we're spending our time and what is worth optimizing
next”. Investigate CPU/GPU and Rust/Python switching costs, including whether
Rust inference or custom CUDA kernels would help. Those are hypotheses, not
selected rewrites.

## The demo

Submit two checkpoint-versus-anchor panels and a checkpoint-versus-checkpoint
panel. Concurrent games share resident policy instances and produce batches
larger than one. Completed deal blocks retain their original seat/deck legs,
Commands, replay checks, and checkpoint coordinates.
Open the same experiment's Perf view: compare scalar and batched jobs, drill
from total eval latency into serving/engine/evidence work, and inspect a linked
CPU/GPU timeline. Identify the next optimization from measured lost time and
its possible end-to-end benefit; missing attribution remains visibly unknown.

In the later queue increment, submit a study whose stages span priorities 20–80
and training checkpoints defaulting to 50. Override one pending request to 90;
the next eligible work comes from it. Explain the matched rule and blockers.
Cancel pending work, restart the coordinator, and retain completed results and
failed attempts. These numbers illustrate behavior, not accepted defaults.

## Shape and ownership

**Additive series.** The keystone is a bounded, single-machine shared inference
runtime for ordinary frozen-policy arena evaluation. It ships as one PR with
internal slices. It accepts several deal blocks, so batches are not limited to
four/eight games. CPU supplies correctness coverage; CUDA is an explicit serving
backend whose end-to-end gain must be measured. No distributed serving is needed
to prove this cut.

Proposed ownership: Manabot owns game execution, batching, policy caches, and
comparison definitions. Etude owns durable demand, priority, claims, capacity,
spending, and result reuse. The keystone has a bounded in-memory work list, not
a second durable queue. Etude's current `play(Match) -> Tables` boundary remains
the result unit; concurrent callbacks can connect to one allocation-scoped
inference runtime. Upstream persistent-queue capabilities need their own design
and installable SDK before that follow-up cuts over callers.

## Baseline before this change

[`batched-eval-inference-evidence.md`](batched-eval-inference-evidence.md) records
the inspected paths and historical measurements. The original arena `play_cell`
starts one process per game, loads players there, and executes games serially. Monitoring,
Etude benchmarks, and duels converge on it. Registrations explicitly require CPU
batch size one. Training already owns a reusable `PolicyInference` cache; arena
does not use it. `BatchedSampler`/`run_vector_games` serve different rollout and
mirror-game contracts and cannot replace arena evidence. Existing Etude match
parallelism alone does not combine model calls across workers.

## Data structures and key functions

Implemented typed contracts:

- `EvalCompute`: stable serving identity (version, device/dtype, maximum active
  games, batch rows/delay, residency/cache settings, memory and request limits).
  `EvalExecution` adds the allocation deadline and measurement switches, retained
  in attempt receipts rather than model/benchmark identities.
  One declaration feeds derived registrations and receipts; conflicting compute
  declarations fail admission.
- `PolicyKey`: checkpoint digest, architecture/input/content binding, serving
  implementation, device/dtype and inference mode. Equal architecture alone does
  not make different weights batchable.
- `DecisionRequest`: game/attempt identity, actor, revision, policy key,
  player-decision ordinal, seed, owned viewer-safe observation, and deadline.
  `DecisionReply` echoes identity and returns the selected legal offer.
- `EvalPerformance` owns counters, host measurements, bounded spans/latencies and
  trace references. Its snapshots are the serving, cell and allocation receipts;
  there is no separate `ServingReceipt` type. `reports/evaluation.py` projects
  resource-specific trees and explicit coverage into the report.
  `PerformanceSpan` links request and batch IDs with process/thread/stream
  identity and timestamps; device/host clock alignment remains unavailable.
- `InferenceRuntime.infer(DecisionRequest) -> DecisionReply` is a bounded blocking
  call multiplexed internally through futures and sender/receiver threads.
  `submit_game` returns a future from the bounded game executor. The public
  `eval.evaluation.evaluate` accepts a runtime or callback endpoint; lower-level
  `eval.match.play_cell` accepts a runtime. Single-game execution uses the same
  supervisor. There is no public decision `submit`/`receive` pair.

The inference process exclusively owns immutable admitted models and one
`PolicyInference` scope per resident policy. It groups ready requests by exact
policy/input compatibility and flushes at the row cap or bounded delay; no
barrier waits for the slowest game. Saturated queues apply backpressure. Eviction
only removes idle policies and closes their scopes; in-flight work pins a model.
Game workers own native state, observation encoding, Commands and replay evidence.
They never load a second copy of a centrally served checkpoint.
One allocation supervisor starts the inference process and bounded game workers;
callbacks receive a private local IPC endpoint. It alone grants game slots across
callbacks, so Etude worker count cannot multiply the configured game limit.

## Cache boundaries

1. Reuse verified checkpoint loading and `PolicyInference.catalog()` across
   games for the lifetime of unchanged resident weights. This is the keystone.
2. Reuse already completed identical evaluation units through Etude identities;
   reference their evidence rather than count reuse as independent samples.
3. Defer whole-observation/logit caching until measured hit rates justify hashing
   and memory costs. A future key must cover the complete model input, legal
   mask, history/belief inputs, policy and numerical mode. Never cache sampled
   actions or use hidden world state as the policy-input key.

The model is a policy/value encoder, not an autoregressive text server; an LLM
KV-cache assumption does not define useful reuse here.

## Performance monitoring and optimization decisions

Monitoring is part of the keystone, beginning with the scalar baseline. Reuse
the report vocabulary and interactions in `experiments/report_prototype/`:
jobs and seed coverage, dotted baseline, synchronized flame/icicle drill-down,
resource selection, totals/shares/normalized costs, and visible missing data.
Extend that report surface with real eval receipts; no separate renderer or
hand-maintained timing source. The prototype remains an executable reference,
not a claim that production Etude report integration has shipped.

The default view answers: **how much useful evaluation did this capacity buy,
where did time go, and what should we try next?** Show completed valid blocks
and games per wall-hour/dollar, request latency p50/p95, queued-work age, failed
and interrupted work, cold/warm costs, and time per million evaluation decisions.
Label those as eval decisions, never learner transitions. Keep measured cost and
estimated cost distinct. Show allocation-level idle/shared overhead once; job
views expose an explicit allocation method and unallocated cost instead of
charging every concurrent request the full machine price.

Accepted demo decision, Jack Heart, 2026-10-24: “Yes, keep this overview”.
The default groups activities as Inference / Game engine / Evidence / Other
(visible labels expand the last two to Evidence and replay / Other and unmeasured).
Detailed phases remain behind drill-down. This changes presentation, not clocks,
coverage or retained resource keys; overlapping intervals remain separate.

Retain these stages: queue/admission; startup/model load/catalog preparation;
native stepping and observation encoding; Python input assembly; IPC;
batch formation; host-to-device copy; inference; sampling; device-to-host return;
Command validation/durable acknowledgement; replay; output retention. Count
boundary calls, transferred bytes, synchronizations, kernel launches, batch rows,
resident-model hits/evictions and catalog builds. Unmeasured subdivisions remain
unknown. Native-call wall time includes real engine work; it is not a measurement
of binding overhead.

Add a timeline beside timing trees for concurrency, GPU gaps and backpressure.
Keep request elapsed time, worker wall time, process CPU time, GPU event duration
and allocated machine time separate. Trees partition only compatible nested
intervals; overlapping workers/streams cannot be summed into one wall-time pie.
Do not invent a critical path from aggregate counters. Mixed-coverage job
aggregates show their measured population rather than substitute zero timers.

Two instrumentation levels:

- Default bounded counters and host spans introduce no extra device waits and
  preserve per-request RNG. Export coarse snapshots, with bounded samples for
  latency distributions. Mark launch-side host timers as such.
- Short diagnostic windows use CPU/CUDA profiler traces and GPU events to
  distinguish launch, copy, execution and synchronization. Collect native symbols
  or opt-in internal spans where Rust work needs subdivision. Keep profiler
  overhead and synchronized diagnostic runs separate from ordinary throughput.

Every suggested optimization names the observed bottleneck, coverage, relevant
waiting/dependency evidence, expected mechanism and a matched validation run.
For genuinely serial nonoverlapping work, show the Amdahl bound
`1 / ((1 - f) + f / s)`; a 10% phase disappearing entirely permits at most 1.11×.
For concurrent work, report a hypothesis until a trace or controlled experiment
shows that reducing it shortens completion time. No ranking by raw summed timer
size alone, and no optimization recommendation generated from synthetic timing.

## Switching-cost investigation

Historical scalar evaluation is CPU-only. The implemented shared CUDA backend introduces
device transfers but can amortize them over many ready games. Separate three
questions: Python dispatch/kernel launches, bytes moved and waits across CPU/GPU,
and Rust/Python conversion/dispatch versus engine computation. Moving the caller
to Rust does not by itself remove CPU-resident game observations or GPU kernels.

Choose interventions from evidence:

| Observed limit | First experiment | Larger change only if needed |
|---|---|---|
| Tiny batches, repeated loads/catalog work | Resident policies, existing cache, coalescing | Broader scheduling to fill batches |
| IPC/observation copies or native binding overhead | Reusable contiguous buffers, fewer/coarser calls, bounded shared memory | Native game/serving orchestration |
| GPU launch gaps or Python dispatch | Bounded shape buckets; compare `torch.compile` and CUDA graph replay | Rust inference runtime with measured dispatch savings |
| CPU/GPU copies and forced waits | Transfer whole batches; return compact actions; overlap independent games with safely owned staging buffers | More device-resident preprocessing/sampling |
| Identified expensive operator or unfused chain | Library/compiler-generated kernels and fusion | A narrowly scoped custom Triton/CUDA kernel |
| Native rules/encoding dominates | Profile and optimize that native work | Inference rewrites cannot address this directly |

Measure batching, caching, compilation and any transfer change separately before
combining them. Charge compilation/capture/warmup, shape churn and workspace memory
to the model's actual residence time; short-lived checkpoints may not amortize
them. Bound shape buckets and cached variants. Input admission stays outside
captured device regions; preserve RNG semantics rather than silently replacing
the sampler for graph capture. CPU sampling versus device sampling is a measured
tradeoff, with different output-transfer costs and numerical contracts.

Rust inference or custom kernels are follow-up candidates only after a current
eval profile identifies addressable cost and an experiment shows end-to-end gain.
Neither belongs in the initial rewrite. Any changed backend retains checkpoint
admission, masks, numerical checks, seed semantics, explicit execution identity
and replay; mixed precision is a separate intervention.

## Compute placement and local evaluation

Direction added by Jack Heart, 2026-10-24: explore “on demand vs always-running
small gpu cluster”; the primary workload is “checkpoint v checkpoint of whatever
our current model size is (likely going to grow over time)”. Mixed small/large
models and heuristics remain supported cases, not equal-weight sizing targets.
Also preserve “i want to do this quick eval check on my laptop with a small model
against heuristic” and “reuse the right code and also NOT reuse the wrong code”.
These are accepted requirements; the placement mechanisms below remain proposals.

**Same evaluation semantics, separate execution ownership.** A local check is an
ordinary arena comparison with a small explicit cohort and local resource limits.
It does not manufacture a TrainingRun, enter the durable queue, discover remote
credentials, provision capacity, or start the Etude callback endpoint. It loads
local checkpoint bytes and writes local results. Heuristics execute in the game
worker; checkpoint requests use the admitted local inference implementation.
CPU is a portable explicit option, with bounded workers/threads and a whole-run
deadline. No silent cloud fallback or automatic replacement of an expensive
checkpoint with a smaller model. A timeout retains an incomplete attempt.

Public seam:

- `EvaluationRequest`: existing player registrations, world/matchup binding,
  declared deal blocks/seeds, inference contract and purpose; no provider or
  training-run prerequisite. Reuse existing domain types rather than a parallel
  checkpoint/model schema.
- `evaluate_local(request, *, limits, out_dir) -> EvaluationResult`: owns a bounded
  local lifetime and calls the same arena executor and finalizer.
- Current managed adapters invoke the same executor with an allocation-owned
  runtime. Training monitoring adds coordinates to the result; it does not own
  evaluation. Priority, placement and spending wrappers remain later queue work.

Both routes reuse admission, observation/action semantics, actor RNG, supervision,
model/catalog caching, Commands, replay and result/performance records. Durable
queue policy, machine leases, remote artifact staging and callback transport live
in adapters outside the local route. Optional hosted publication is also outside
local execution. A quick check reduces the declared sample, not correctness or
evidence integrity; label its limited purpose without inventing promotion evidence.

Resolved entry gap: `play_cell` is independent of cloud scheduling, while
`experiment.checkpoint_evaluation.evaluate_checkpoint` requires TrainingRun/artifact/training coordinates;
`duel.Duel` encodes a particular two-seat/one-seat study.
`eval/evaluation.py` now owns the generic request, execution and result API.
`eval/admission.py` owns checkpoint/anchor registration extracted from
`experiment.checkpoint_evaluation.checkpoint_manifest`; the monitor keeps its training metadata wrapper.
Managed adapters and monitoring call the same evaluator and paired-deal finalizer.
Duel retains its study-specific validation. Local preparation/CLI lives in
`eval/local.py`; shared `EvaluationRow`/`RateInterval` types live in `eval/results.py`.
The public evaluator atomically claims a fresh panel directory, rejecting even
empty existing directories before evidence writes or execution. Local, monitor,
benchmark and Duel wrappers leave that ownership to it. Reusing completed Etude
deals still skips execution; explicit continuation retains new attempt directories
under the same evaluation, while fresh full repeats require a new evaluation directory.
The CLI reports the observed score but no confidence interval for one deal:
resampling correlated legs cannot estimate between-deal uncertainty. Stored
bootstrap values and frozen monitor/Duel statistics retain their definitions.

**Capacity comparison.** Test scale-to-zero allocations, a small warm pool, and
a warm baseline plus bursts against the same arrivals and requested outcomes.
For Jack Heart's clarified cadence of a few experiments per week, the working
hypothesis is now one on-demand host that drains a group of comparisons and shuts
down. A warm baseline plus bursts remains a candidate if measured demand or
latency requirements justify idle capacity; neither is an approved purchase.
Compare recurring checkpoint pairs first, then a rotating recent-checkpoint set,
then growth-size pairs and mixed/heuristic cases. Same architecture with different
weights still requires separate policy batches. Measure occupancy per checkpoint,
not just total queued games; affinity cannot change the requested comparisons.

Start with CPU-only and one GPU with sufficient CPU/RAM for native games, input
assembly and replay. Candidate GPU classes include commodity RTX, L4-class and
larger-memory L40S-class hosts; current offers need dated whole-host quotes.
Measure both resident opponents plus catalogs, peak activations, allocator/runtime
headroom and checkpoint replacement overlap. Increase CPU supply or memory before
assuming another GPU fixes starvation or cache churn. Shard independent deal
blocks across hosts that admit both opponents before considering cross-host
per-decision inference. A growing model that no longer fits that topology needs
an explicit subsequent design. The current runtime has one device per allocation;
simultaneous CPU-small/GPU-large routing and cluster placement are not implemented.

For each candidate record completed valid blocks per dollar, result-latency p50/p95,
launch/load/drain time, model evictions, CPU/RAM/VRAM and idle paid hours. Compare
`steady_rate × window_hours` against `elastic_rate × billed_hours + other_costs`,
using measured throughput and equal requested work; include interruptions and
recovery. Keeping an on-demand machine running does not itself discount its rate.
Queueable studies can fill troughs only within their requested cohorts/priorities.
Use age/expected service time by model class to propose burst admission; retain
budget caps and drain completed blocks before shutdown. Training/eval co-location
is a separate interference trial with training slowdown charged to the comparison.

The demo adds one small local checkpoint-versus-heuristic panel, without training
metadata or cloud access, and the same request executed by a managed adapter.
`tests/eval/test_local_evaluation.py` covers local execution with hosted/provider
entry points forbidden, shared results/admission, deadline cleanup and incomplete
receipts. The CLI is `uv run python -m manabot.eval.local`; typed callers import
`checkpoint_request` from `manabot.eval.local` and `evaluate_local` from
`manabot.eval.evaluation`. Neither route needs the `evaluation` extra.
No exact CPU/CUDA trajectory parity is implied.

## Queue and evaluation allocation — follow-up contract

Three independent units: a **comparison** requests evidence, a **deal block** is
the correlated completion/statistics unit, and an **inference batch** groups ready
decisions by model. Sharing compute never pools statistical populations.

Proposed Etude API: `submit(request)`, `set_priority(id, value | None)`,
`set_policy(policy)`, `pause(id)`, `cancel(id, mode)`, `status()`, `explain(id)`.
An immutable request binds checkpoint(s), protocol, exact deal blocks, purpose,
and budget; mutable queue controls are separately revisioned. Submission is
idempotent. Claims are leased and fenced; a stale worker cannot commit over a
new owner. Interrupted attempts retain evidence and are not silently retried.

Use a small declarative policy, with ordered predicates over request metadata
(study, stage, tags, checkpoint update, active-training status, waiting age),
an assigned integer priority, and a required fallback. Explicit per-request
override wins; otherwise the first matching rule wins. Higher priorities run
first. Resident-model affinity breaks ties, bounded by a declared maximum wait
before FIFO wins. There is no implicit aging across priority levels. Policy
preview explains the resulting order before applying it. No arbitrary executable
policy code or automatic score-driven selection in the first queue release.

Priority changes affect unclaimed blocks. Cancellation supports dropping pending
work while draining active blocks, or explicitly interrupting active games with
incomplete receipts. Admission reserves enough time for a block and finalization;
hard allocation deadlines still win. New urgent work waits for available capacity,
not for an entire large study to finish. Prevent oversubscription with one owner
for game slots, model memory and allocation limits.

Choose game counts separately from batching: fixed studies retain their declared
cohorts; development requests may ask for a small balanced panel and explicitly
request more blocks later. Any automatic precision-based stopping needs a later
statistical design. Queue status exposes pending/completed blocks, age, estimated
time/cost, priority provenance, cache affinity and why work cannot start.

## Constraints and forbidden outcomes

- Preserve digest/world/setup admission, actor-only observations, complete legal
  choices, per-decision seed derivation, durable pending-Command acknowledgement,
  exact replay, all failed attempts and existing statistical denominators.
- Sample each row from its own existing policy seed using isolated RNG state;
  other games cannot consume its random stream. Validate logits/masks before
  sampling. Batch/device changes can change floating-point results: retain an
  explicit execution identity, test numerical differences and sampled decisions,
  and never claim cross-device bitwise trajectory equivalence.
- Keep native games in killable processes. The supervisor also watches inference
  hangs/death. Service failures fail affected work as infrastructure failures,
  never invented player forfeits; queued unstarted work remains unattempted.
  Flush prefixes and reap children on shutdown. Waiting counts against active
  game wall limits; unstarted queued games have not started their game clock.
- First coverage: flat, stateless checkpoints versus each other or current
  random/scripted/heuristic anchors. Compound/belief/search players keep their
  existing admitted execution path; explicitly requesting unsupported batching
  fails before playing, with no silent scalar fallback.
- Frozen registrations/results remain readable. New executions declare the new
  envelope; no rewriting old scientific protocols or resuming them under it.
- No training changes, cross-host RPC, generic serving platform, approximate cache
  reuse, or discount assumption. A steady machine can amortize setup but idle
  rental time still costs money. Report whole allocation cost, including idle,
  startup, replay, failures and retention; overlapping game times are not bills.
  Current cell/allocation timers cover narrower software intervals, so whole-job
  accounting remains a capacity-acceptance measurement, not a completed claim.

## Delete — do not maintain

Completed keystone cut: `sim/match.py::play_game` delegates to the shared bounded
supervisor with its failure behavior retained. The centrally served branch of
`_execute_game` constructs no checkpoint player; unsupported player kinds retain
scalar construction. One `play_cell` finalizer owns shared rows and evidence.
Existing crash, timeout and checkpoint fixtures remain.

Local seam deletion cut (completed): remove registration construction and anchor
source selection from `experiment.checkpoint_evaluation.checkpoint_manifest` in favor of `admission.py`;
move `EvaluationRow` and `RateInterval` to `results.py`; replace the monitor-only
bootstrap implementation with the generic cohort validator. Replace benchmark,
Duel and monitoring's duplicate panel/result persistence with `evaluation.evaluate`.
Preserve monitor serialized values, source/seed identities, coordinates, existing
failure fixtures and Duel restrictions. No tests or fixtures are disposable here.
The compression cut also removes wrapper-owned panel mkdir/binding and the
calibration `Panel`'s duplicate request fields/completeness check. Named calibration
panels now wrap `EvaluationRequest` and execute through `evaluate`; replayed but
partial cohorts fail the same validator. Existing numerical fixtures remain.

Keep `sim.rollout`, training caches, and frozen evidence. For the later durable
queue cutover, replace `experiment/evaluation_queue.py::CheckpointQueue` and
`experiment/monitor_worker.py::follow_checkpoints` scheduling responsibilities only
after their producers submit equivalent Etude requests; preserve checkpoint
discovery, budgets, failure retention and reports in the same cut. Historical
serving experiments are evidence, not another implementation to revive.

Completed report cut: `experiments/report_prototype/report.html::jobTiming`
uses coverage-aware aggregation instead of missing-child-as-zero averages.
Job/seed alignment and training panels remain; fixtures include unequal coverage.

## Published foundation

PR #370 is the completed runtime/report keystone, at
`79506de9cae0182653e251767d35ffba5427c066`. Its integration history, full acceptance
plan and prior concept review remain in that commit. Uncommitted design detail
before this curation is preserved in Git blob
`1dcb52e5bbdf3747342bc89b95d0cb17e519cd40`; questions and evidence are preserved in
`d8c8916accba563add0badbfd2cf30970958ca91` and
`7bae0e735e83f08273f093dbea10422ed6f416db`. No measured throughput gain, CUDA
acceptance, hardware selection or paid work follows from its software checks.

## Continuation selected — 2026-10-24

Jack Heart selected explicit continuation after interruption at 40/100 complete
valid deals: finish the missing 60 under one evaluation, retain every attempt,
and count each block once. A fresh full repeat remains separate and is not
automatically an independent sample. The implementation contract below replaces
the earlier proposed API/storage discussion, preserved in Git blob
`e68c00105a4f023642db5b8b753473a93787af0f`.

An **evaluation** owns requested evidence; an **attempt** owns execution and failure
receipts. Keep allocation lifetime/limits/accounting internal: running or continuing
an evaluation needs no public allocation creation. Local execution uses the shared
result authority without scheduling/provider machinery; Etude owns managed
scheduling/cache, never a second Manabot durable queue. No native-state resume,
automatic retry, mixed numerical contracts or frozen-evidence migration is implied.

### Compute policy — startup threshold accepted, hardware draft, 2026-10-24

Accepted decision, Jack Heart, 2026-10-24: “aiming for at least 2hrs of use before
spinning up an eval server makes sense”. Default cloud startup waits for enough
eligible queued evaluation work to occupy the proposed host for at least two
useful hours. Estimate time from the requested checkpoint mix and measured host
throughput, not a universal game count. This is a backlog target, not a minimum
billed lifetime: if work finishes early, retain results and release idle compute.
Local laptop checks remain independent. Urgent overrides and maximum queue wait
remain open policy choices; no automatic exception is selected. With the assumed
3–5 minute startup, two useful hours puts startup at about 2.4–4.0% of total time.
The policy is accepted design intent; provisioning is not implemented by this
decision, and hardware and spending remain unselected.

Jack Heart asks for the cheapest reasonable throughput for “a few experiments a
week”. Proposed first topology: existing laptop CPU for small checks; one rented
GPU host for queued checkpoint comparisons, with native games, heuristics,
inference and replay colocated. Keep both opponents resident during each session,
fill policy-specific batches from concurrent games, retain results externally,
then release compute. This preserves the shared-pool direction without assuming
continuous paid capacity. Experiments per week alone does not determine hours:
checkpoint count, deal count and time to a result remain necessary sizing inputs.

First comparison candidates are a modest 24 GB GPU host (A5000 class) and a faster
consumer GPU host (5090 class), with measured CPU quota/performance and enough
RAM. A 48 GB A6000-class host is a memory candidate if two policies, catalogs and
batch workspaces need it. These are benchmark candidates, not proven fit or a
purchase selection. Existing training evidence favors commodity hardware over
H100 among its measured hosts, but does not establish eval performance. CPU-only
remains a measured baseline; small policies may not repay GPU/IPC overhead.

Current advertised Runpod examples are $0.27/h A5000, $0.59/h A6000 and $1.19/h
5090, subject to host availability and additional charges. At an illustrative
12 billed hours/week these cost $3.24, $7.08 and $14.28; 168 hours cost $45.36,
$99.12 and $199.92. This is arithmetic, not a runtime forecast. Dated sources and
host resources are in the evidence file. Compare total paid cost per valid
complete block at acceptable latency, including startup, replay, failures and
storage. Interruptible capacity becomes a candidate once completed-block
continuation and durable evidence are proven on the selected managed route.
No rental or calibration run is selected here.

Startup amortization: for paid startup duration S and useful execution duration W,
startup's share is S/(S+W). At most 10% requires W >= 9S; at most 5% requires
W >= 19S. An illustrative 3–5 minute startup therefore needs 27–45 useful minutes
or 57–95 useful minutes, respectively. Startup is unmeasured on the proposed
hosts. Historical local probe012 completed 32 games in 67.693 command seconds
(about 28 games/minute); using that only as a scale illustration gives roughly
800–1,300 games at 10%, or 1,600–2,700 at 5%. It used a 384×8 initialization
against anchors, not current learned checkpoint pairs or shared CUDA serving,
and repeated existing deals. These are execution counts, not independent samples
or a current throughput forecast. Measure paid startup separately from unbilled
provisioning delay; retain shutdown/upload cost in whole-session accounting.

## Continuation implementation plan — 2026-10-24

Accepted behavior: Jack Heart authorized this follow-up in the current conversation.
One logical evaluation retains every attempt; continuation executes only missing
complete blocks. Fresh repeats use a fresh directory. Allocation remains internal.

Concrete reversible persistence decision: retain the first attempt at the existing
evaluation directory and later attempts under `attempts/<sequence>/`. The initial
`evaluation.json` freezes the full request and identity; it is never rewritten by
continuation. Status is derived from retained attempt receipts and durable rows,
not a second mutable progress database. A UUID distinguishes logical repeats.
A nonblocking OS file lock fences writers on this shared filesystem, including
managed callbacks. Process death releases it; no expiring lease can admit a stale
writer. Cross-host object-store claims are not implemented by this local lock.

`EvaluationResult` owns block validation and cohort statistics. It validates every
row against the request before selecting complete valid blocks. Each seed contributes
once; conflicting duplicate blocks fail closed. Attempts may change deadline and
instrumentation, not compute identity, player bytes, setup or scientific limits.
Paths may be relocated only after verifying registered bytes. Incomplete blocks
are rerun whole. Status references original trace paths instead of copying evidence.

`play_cell` replays and durably publishes each finished block while other games
continue. Immutable per-block traces permit safe reuse even if the process dies
before the full panel returns. The existing final aggregate trace remains the
panel-level diagnostic/calibration artifact. Reports read the row-referenced shards
to avoid counting aggregate and block traces twice. One runtime still batches
across concurrently active blocks; no per-block model reload or serialized scheduler.

Public operations: existing `evaluate` starts a fresh evaluation;
`evaluation_status(directory)` reconstructs progress;
`continue_evaluation(directory, execution=..., inference=...)` explicitly runs the
missing cohort. Local CLI adds `--continue` and `--status`; continuation uses the
frozen compute settings and a fresh deadline. Managed benchmark/Duel callbacks
use the same continuation operation on explicit re-entry into an existing match.
Etude retains match scheduling/cache; Manabot validates underlying evidence and
never submits a second queue. A managed callback is one Etude complete deal;
a multi-deal managed endpoint request uses the same shared evaluation authority.

Delete — do not maintain (follow-up, completed): all-or-nothing end-of-panel
replay in `eval.match.play_cell`, the separate cohort-validation loop, and managed
callbacks' existing-directory failure on explicit re-entry are removed. Incremental
replay preserves final trace/card evidence, subsets, failures and Perf receipts;
ordinary `evaluate` still requires fresh output. No frozen evidence or queue is deleted.
`EvaluationResult._complete_blocks` indexes rows by deal/leg once for validation,
statistics and continuation, without repeated cohort scans or registration hashing.
Duel summaries discover evaluation receipts even when the first attempt published
no rows; only historical panels use the rows-only reader. Per-game regeneration of
the growing aggregate trace is removed: immutable block shards and fsynced Command
journals retain progress, and `play_cell` writes the diagnostic aggregate once on
settlement, including coordinator exceptions. No deletion target remains in this cut. The durable-queue producer cutover above still belongs to later work.

Cloud startup gap, re-inspected in installed `etude-sdk==0.2.4`, 2026-10-24:
`etude.launch.scheduler.Scheduler._plan` first fills idle sessions, then selects
an eligible run for a new rental. Rental lifetime uses `Job.max_hours`, cohort
pending-job ceilings divided by available parallel slots, setup allowance and
dollar caps. None estimates useful work. `OnDemand` and `launch.client` expose no
measured host/model backlog threshold or estimator hook. `eval.tournament.hold`
executes callbacks on existing compute; `CheckpointQueue` monitors local
subprocesses. Neither owns server startup admission.

The remaining implementation owner is Etude's launch scheduler: estimate eligible
remaining evaluation work for the proposed host/model mix, then gate new rental
creation at >=7200 useful seconds under the existing supply lock, before
`_session` and the store's `plan`. Reuse of already available compute stays
independent of this startup gate. `_tick_supply` already ends non-kept idle
sessions; the target must not become a minimum billed lifetime. Manabot still
needs measured workload estimates and integration with that upstream capability.
No second queue or provider helper is needed. Urgent overrides and maximum wait
remain unresolved decisions; neither is silently enabled.

### Agreement with the synced base

Benchmark and Duel seat training exports with
`CheckpointRef(key=CheckpointKey(run=run_id, name=filename), artifact=ArtifactRef(...))`.
Call sites read `agent.artifact.sha256`/`.uri`, not the old digest/string fields.
The checkpoint key remains distinct from its byte location; equal bytes under
different keys are distinct Etude entrants. Continuation preserves the Manabot
request/compute identity and verifies relocated bytes. It does not migrate old
Etude match keys or add a sample when the SDK reuses a completed match.

Upstream card evidence lives at `artifacts/evidence/<digest>.json.gz` beneath the
eval run root. Etude 0.2.4 delivers these immutable artifacts once; no checkpoint
namespace or index-only publication is substituted. Incremental replay retains
card references in the diagnostic trace, and selected-match manifests enumerate
the actual row-referenced block shards. Frozen panel trace locators remain valid.

Focused proof: synthetic 40/100 interruption followed by local and managed
continuation, whole-block retry, duplicate/contract/tamper rejection, lock fencing,
crash recovery from published blocks, fresh repeats and offline CLI; small real
native/endpoint fixtures verify incremental replay. Full affected suites belong
to gate. Cloud threshold integration remains blocked on the upstream SDK owner.

Compression's completed checks cover the missing-row Duel summary, checkpoint
admission failure fixture, native interruption, concurrent blocks with one resident
model, and removal of repeated aggregate compression. Its wider focused run had
47 passes and one source-identity conflict caused by an edit between execution
and cache reuse; the stable-source rerun passed. Those results remain evidence,
not post-sync checks or hardware proof. Broad suites remain with gate; the browser
proof is retained in `eval-perf-demo.md` without changing the parent report/script.

Check: `uv run --no-sync pytest tests/training/test_etude_evaluation.py tests/training/test_head_to_head.py tests/eval/test_local_evaluation.py -k "etude or direct or continuation or status_recovers" -q` — 16 passed, 16 deselected; `uv run --no-sync pytest tests/eval/test_card_evaluation.py::test_any_evaluation_can_deliver_card_data_beside_its_own_tables -q` — 2 passed; `git diff --check` passed. Installed SDK admission and manifest paths were source-inspected; broad suites and representative CPU/CUDA calibration remain with gate.
