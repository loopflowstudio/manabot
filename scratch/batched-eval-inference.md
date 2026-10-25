# Shared evaluation compute

Accepted for pursuit — 2026-10-24. Jack Heart approved the design and said
“all sounds good. lets pursue”. Deliver the complete single-machine shared
inference and performance-report keystone through its internal slices in one PR.
The named later increments remain follow-ups. Hardware, numerical/performance
benchmark thresholds and spending remain unselected; no paid execution is
authorized. Standalone work; no Task or Wave selected.

Software complete — 2026-10-24: the full runtime/report keystone and generic
local API are implemented against the upstream sim/eval split, Etude 0.2.2 and
card-evidence reporting. Sync succeeded after ordinary staging (supplied outcome).
Broad validation, report browser proof and representative CPU/CUDA performance
acceptance remain gates; no measured whole-job gain or capacity choice is claimed.

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
deals still skips execution; rerunning an attempted panel needs a new directory.
The CLI reports the observed score but no confidence interval for one deal:
resampling correlated legs cannot estimate between-deal uncertainty. Stored
bootstrap values and frozen monitor/Duel statistics retain their definitions.

**Capacity comparison.** Test scale-to-zero allocations, a small warm pool, and
a warm baseline plus bursts against the same arrivals and requested outcomes.
The working hypothesis is a warm baseline plus bursts, not an approved purchase.
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

## Internal slices

1. Complete: scalar instrumentation, real Perf projection, bounded supervisor and
   shared finalization.
2. Complete: resident inference/catalog caching, per-request RNG, bounded batching
   and failure handling; monitoring, benchmark, Duel and generic local consumers.
3. Software complete, acceptance pending: the four-ablation calibration runner
   and CPU fixture proof exist. Representative CPU/CUDA panels, numerical and
   performance thresholds, whole-job costs and a measured optimization decision
   remain unestablished.

Implementation status — 2026-10-24: the runtime/report and local-seam deletion
targets above are complete. `sim/execution.py`, `serving.py` and `supervision.py`
own execution admission, resident `PolicyInference` scopes, bounded policy batches,
per-row CPU sampling and killable game/replay processes. Shared finalization
retains journals, Commands and original statistical units. Local CLI preparation
runs in a bounded child with incomplete receipts on admission/deadline failure.

Monitoring sends multi-deal panels; benchmark/duel callbacks use an authenticated
local endpoint because pinned Etude runs callbacks in spawned processes. One
allocation owns the executor, queued submissions, native slots and serving
process. Scalar/frozen protocols keep their original declarations. Derived
executions retain `seed_compute_envelope_id`, preserving the original pair/deal/
player seed authority across ablations while declaring new execution identities.
Registration and Etude benchmark identities use only `EvalCompute`; allocation
deadlines and instrumentation no longer fragment identical comparisons or cause
Etude tournament-resume conflicts. Full `EvalExecution` remains in allocation
and game receipts. Completed deal reuse is the same evidence, never a new
performance sample; the direct calibration runner still executes fresh attempts.
Benchmark/duel callbacks share one lifetime helper and retain a separate allocation
directory on each invocation, including calls that reuse completed Etude deals.
Duel cohort hashing precedes execution binding so its original opponent remains
the seed authority too. The bounded sender thread stays outside the futures lock;
a stopped service with a pipe-filling payload must still time out and be reaped.
The call budget is the minimum of configured request limit, caller deadline and
allocation remainder, shared by capacity waiting and reply waiting. Policy keys
and execution parsing occur once per game; final retained identities alone own
unattempted counts.
Service errors stop unstarted work without inventing player forfeits. Unattempted
identities and unavailable replay survive explicitly in receipts.

`sim/performance.py` and `reports/evaluation.py` feed real receipts into the
existing report assets; missing-child-as-zero aggregation is removed. Timelines
join requests/batches, shared cost appears once, partial coverage is visible,
and optimization hypotheses name measured phases, dependencies and validation.
Serial scalar cells alone receive an elimination bound. Diagnostic CPU/CUDA
traces and CUDA event durations are separate from ordinary host timings.
Cell timing starts after initial comparison validation and snapshots after replay;
allocation timing spans runtime construction/shutdown. Both exclude local CLI
checkpoint preparation. The CLI deadline includes preparation, but is not an
elapsed-time receipt. `evaluation_seconds` also excludes cohort statistics and
final receipt writing. Serving resident-load counts omit earlier admission loads.
No provider provisioning, complete paid lifetime or billing amount is measured.
The exact boundaries and later compute/queue/allocation intents are retained in
`docs/shared-evaluation.md`; current reports support interval-scoped throughput,
not whole-job cost or an end-to-end speedup claim.
`eval/serving_benchmark.py` validates a frozen four-ablation plan, interleaves
three repetitions, retains failures and performs matched-root numerical and
instrumentation on/off checks in memory/deadline-bounded subprocesses.
Its named panels carry the public request rather than a parallel player/deal
schema. Early admission failures retain their evaluation receipt; report projection
requires retained rows and timings, and numerical probes require an actual trace.
Calibration materializes only the declared root prefix, without unused root hashes
or a separate warmup option. Runtime admission checks each repeated deck/seat setup
once per policy, independent of the panel's deal count.

CPU fixture evidence includes multirow batches, one resident model/catalog,
two-policy separation, eviction, stale replies, service death/hang, unchanged
per-row RNG, native replay through a spawned callback, a complete checkpoint
monitor block and real report projection. The controlled retained-root check
covers all four modes without selecting or running a scientific benchmark.
CUDA parity is an explicit skipped test here. Representative hardware, model
mix, numerical/performance acceptance thresholds and billing are not supplied;
no throughput/cost improvement or cross-device trajectory equivalence is claimed.
The full affected suites and representative calibration remain owned by gate;
no training, paid execution, durable queue or publication occurred.

Later independently valuable increments: **Etude queue** (durable API, priority
rules, claims, producer cutover); **compute policy and cluster operation**
(scale-to-zero vs warm pool vs baseline-plus-burst, budget windows, idle policy,
replenishment and whole-allocation accounting); **adaptive evaluation allocation**
(checkpoint thinning/expansion and statistically justified stopping); **further
serving** (compound/search support, output caching, native inference or specialized
kernels when measurements justify them).
Each receives its own design. No follow-up Tasks are filed here.

## Done when and measure

`tests/eval/test_batched_eval.py` and the shared monitor integration prove one model load and
one catalog build per resident policy, one forward for an assembled multirow
batch, two-policy separation, bounded partial-batch flush, eviction, stale replies,
deadline/native/service failures, per-row RNG independence, numerical parity and
complete replayed blocks through real callers. CPU tests run headlessly; CUDA
tests run on a capable runner and cannot be inferred from CPU success.

`tests/eval/test_eval_performance.py` covers: timing-tree accounting, overlapping
intervals, missing/mixed coverage, cost attribution, batch/transfer counters,
bounded instrumentation and trace joins. Instrumentation-on/off comparison uses
identical request schedules so timing-induced batch changes are not misreported
as RNG corruption. Never require a performance threshold on a loaded CI host.
Headless `./scripts/verify-eval-performance-report` renders recorded,
incomplete and overlapping fixtures through the existing report surface and
checks job comparison, resource/normalization controls, drill-down, timeline,
missing-data labels and optimization evidence links. No display or paid machine
is required. When a native diagnostic change is needed, apply the repository's
debug Rust, clippy, formatting and atomic-extension-rebuild checks.

Remaining gate validation:

```sh
uv run --extra dev --extra artifacts --extra evaluation --extra reports pytest tests/eval/ tests/sim/ tests/model/test_inference.py tests/training/test_monitor_evaluation.py tests/training/test_etude_evaluation.py tests/training/test_head_to_head.py tests/training/test_automatic_game_capture.py tests/training/test_checkpoint_queue.py tests/test_package_boundaries.py -q
./scripts/verify-eval-performance-report
```

Expected: all pass, including new tests collected with existing modules. The
headless `manabot.eval.serving_benchmark` entry point is implemented,
using retained-root selection and `sim.replay.replay_prefix` for matched inputs:

```sh
uv run --extra evaluation python -m manabot.eval.serving_benchmark --plan .runs/eval-serving/plan.json --out .runs/eval-serving/result
```

The frozen plan supplies admitted checkpoints/registrations, deal blocks, device,
all four execution configurations, three interleaved repetitions, numerical
tolerances, time/memory caps and an adoption threshold. Missing fields fail before
execution. This design does not create a benchmark allocation. Compare cold scalar, resident-only,
resident+catalog-cache, and batched configurations on the same panels, including
small and larger models and two-checkpoint mixtures. Retain every repetition and
failure, actual batch occupancy, game/replay time, CPU/RSS/VRAM and input/output
agreement. Optimize valid completed blocks per paid machine-hour and time to a
requested result; model-call speed alone does not satisfy the goal. No numerical
speedup target or hardware selection is accepted yet.

Integration cut completed — 2026-10-24: reconciled stash `187271122326af8955746647d11d3790975c649a`
onto main `884249024` (original base `07a091bca`). `sim.match.play_game` and
`sim.supervision` own native execution; `eval.match.play_cell` owns paired games,
leg subsets, row finalization and optional upstream card evidence. Serving config,
runtime and counters remain in `sim`; comparison binding, requests/results, local
CLI, calibration and allocation callback transport belong to `eval`. Training
admission/execution stays in `experiment.checkpoint_evaluation`; upstream removed
supervised training. Etude 0.2.2 experiment identity and current source paths apply.
Frozen source contracts and serialized arena fields remain unchanged.
Calibration selects roots locally and reconstructs them through `sim.replay`;
no retired profiling module is restored. Local evaluation imports callback
transport only for type checking, so it has no runtime callback dependency.

Delete — do not maintain (integration, completed): removed all guessed new `sim` comparison
modules, obsolete `arena/models.py` and `arena/monitor.py`, the old combined
match implementation, duplicated monitor row/statistics/admission and the old
calibration profile dependency. Preserve native supervision/failure fixtures,
shared request validation, fresh-attempt directory ownership, one-deal CLI
uncertainty, card evidence/report APIs, subset-leg execution and source identities.

Integration fixes also drain pending worker messages after observed exit,
preventing a poll/exit race from losing player attribution or completed replay.
The deterministic native-exit regression failed before that fix.

Integration and compression are staged; sync succeeded after ordinary staging.
Earlier stash errors are historical, not current blockers. Software completion
does not satisfy the remaining broad suites or representative
CPU/CUDA acceptance. Hardware, model pairs/growth sizes, numerical tolerance,
latency/budget targets and improvement/overhead thresholds remain unselected.
No scientific training or paid work is selected.

Check: prior `uv run --no-sync pytest tests/eval/test_local_evaluation.py tests/eval/test_batched_eval.py -k 'calibration or rejects_repeat or describes_deal or concurrent_deal_blocks or execution_binding or benchmark_missing' -q` → 9 passed (73s), with lint/format/compile/calibration CLI checks; reused integration's 77 focused checks, 3 process regressions and 137-test collection. Reconciliation adds only documentation; CLI/import and whitespace checks recorded in the evidence. `./scripts/verify-eval-performance-report` passed after reconciliation (exit 0). Gate/CI retains broad suites; representative CPU/CUDA acceptance awaits a declared hardware/benchmark plan.

## Concept review — 2026-10-24

Outcome: no design change and no open product choice within this keystone.
Review against Jack Heart's accepted experience found the implemented ownership
and interaction consistent with it. This records a review conclusion, not new
approval, performance acceptance or a selection of the parent's next step.

The user chooses opponents, setup and complete deal blocks, then inspects one
attempt's evidence. Checkpoint-versus-checkpoint at growing model sizes remains
the primary workload; a laptop checkpoint-versus-heuristic check uses the same
contract with a smaller declared sample. The existing usage examples in
`docs/shared-evaluation.md` express that experience without a new user-facing
queue, training record or callback concept on the local path.

| User action | Concept and state | API and owner |
|---|---|---|
| Choose local checkpoints or an anchor and explicit deals | Admitted registrations in an `EvaluationRequest` | `eval.local.checkpoint_request`; shared `eval.admission` |
| Run a bounded laptop check | Fresh attempt, completed or incomplete | `eval.evaluation.evaluate_local`; `sim` owns inference, games and replay |
| Run the comparison inside a managed allocation | Same request/result semantics; shared resident policies and capacity | `evaluate(..., inference=...)`; callback transport stays in `eval`, training coordinates in `experiment.checkpoint_evaluation` |
| Inspect a failure and try again | Retained old attempt plus a new attempt directory | CLI `--request ... --out <new-directory>` or the same request passed to `evaluate`; no in-place evidence replacement |

Normal execution preserves the useful distinctions: a comparison requests
evidence, a deal block groups correlated legs, and a policy batch shares inference.
Neither concurrency nor cached models changes the statistical population. Exact
policy identity keeps different weights separate; per-decision RNG and recorded
Commands retain their authority across callers. Card evidence and reporting,
explicit lower-level leg subsets, and Duel's study restrictions remain with
their existing owners rather than becoming local-check requirements.

Failure and recovery also fit the accepted contract. Admission failure or a
deadline retains an incomplete receipt; started games retain journals and
Commands, unstarted work remains explicit, and unavailable replay cannot become
a valid result. Service failure cannot become a player forfeit. Retrying uses
fresh output and revalidates the same requested bytes and cohort. It does not
resume native state or silently combine partial attempts. Completed Etude deal
reuse remains reference to existing evidence, never another sample or a fresh
throughput measurement.

Counterexamples remain constraints: four correlated legs from one deal cannot
establish between-deal uncertainty; identical architecture cannot make distinct
checkpoint weights one batchable policy; and a warm allocation's timer cannot
describe full local preparation or paid machine lifetime. The CLI's one-deal
qualification, retained historical bootstrap fields, policy keys and explicit
timer coverage preserve these distinctions. None calls for a new product choice.

Next proof remains gate/CI's full affected suites. Representative growth-size
checkpoint pairs, CPU/CUDA numerical behavior, latency and whole-job cost need
the separately declared calibration plan before capacity or speedup claims.
Those inputs do not block the opt-in software boundary. Durable queues, cluster
policy and adaptive allocation remain independent follow-ups; this review
selects no hardware, execution, publication or merge.
