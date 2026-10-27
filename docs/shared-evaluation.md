# Shared evaluation inference

Simulation (`manabot.sim`) owns serving, native games and bounded replay.
Evaluation (`manabot.eval`) owns comparison admission, requests, results, local
execution and callback transport. Training monitoring is an adapter in
`manabot.experiment.checkpoint_evaluation`. Local execution imports neither that
adapter nor callback transport.
The former `manabot.arena` package remains absent; serialized arena field names,
seed identities and frozen evidence retain their original meanings.

`manabot.sim.execution.EvalExecution` declares a local evaluation allocation:
CPU or CUDA float32, active-game limit, batch-row limit, batching delay,
resident-model byte allowance, request timeout and absolute Unix deadline.
The default registration remains historical CPU scalar execution. An explicit
execution derives new registrations and a new arena compute envelope; it does
not resume frozen results under different numerical assumptions.

`EvalCompute` is the stable part of that declaration: device/dtype, batching,
residency/cache settings and request limits identify the player and benchmark.
`EvalExecution` adds the allocation deadline and instrumentation/diagnostic
switches. Those attempt settings remain in game and allocation receipts, not
registration hashes. A new allocation window therefore preserves Etude model,
benchmark and match identities when its compute contract is unchanged; changing
the compute contract still creates distinct identities. Duel seed derivation
also retains the original opponent cohort before binding shared execution.

Etude can reuse completed deals under the same identity. Such reuse adds no new
sample or performance measurement, and changing instrumentation alone does not
force a replay. Use the calibration runner's fresh direct arena attempts when
measuring instrumentation or throughput. Diagnostic timings stay separate from
ordinary throughput even though the underlying policy identity is the same.

## Batching and processes

One allocation runs one serving child (`manabot.sim.serving`), the only process
that loads served weights. `PolicyKey` groups requests by checkpoint,
observation/action ABI, world, device, dtype, determinism and implementation.
Matchup and content suite are not model inputs: the card catalog is bound by the
checkpoint's world, so concurrent games that seat one checkpoint share its
resident policy and its batches whatever their matchup. Each registration's
setups are still validated against that policy at admission, and matchup
identity remains in registrations and evidence. This stays `shared-eval-v1`:
the contract never fixed batch composition, which already depended on arrival
order within the row and delay limits, and each row samples from its own
decision seed. Each batch moves to the device in one copy per dtype (float32
observations, int64 catalog references).

Game, replay and serving children start through `manabot.sim.child`, never
`multiprocessing` spawn, so they do not re-import the evaluator's `__main__`.
Model-vs-model game children and replay children never import Torch; games with
an in-process scalar player (code anchors, unserved checkpoints, search) do.
Every child starts with single-threaded library pools
(`manabot.sim.threads.child_environment`) and counts its own threads against a
declared budget: 4 for game and replay children, 16 for serving. Exceeding it
is an infrastructure failure, never a forfeit. Game children record
`threads/game`, `rss_bytes/game` and `torch_loaded/game` in their
`worker_performance`; `serving.json` carries the serving role. Each cell's
`performance.json` records the group worker that ran `play_cell` under
`threads/group_worker` against `max_active_games + 7` with a shared runtime,
or 8 without one; this budget is recorded for calibration, not enforced.
`effective_cpus()` honors affinity masks and cgroup quotas, unlike
`os.cpu_count()`, which reports the host inside containers.

Ordinary flat, stateless checkpoints and random/scripted/heuristic anchors are
supported. Compound, belief and search players retain their existing scalar
path. Explicit shared execution rejects them before playing any games.

```python
from pathlib import Path
import time

from manabot.sim.execution import EvalExecution
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.sim.serving_protocol import InferenceRuntime

execution = EvalExecution(
    device="cpu",
    max_active_games=4,
    max_batch_rows=16,
    max_delay_seconds=0.005,
    resident_model_bytes=1 << 30,
    deadline_unix=time.time() + 300,
)
# A new directory is required for each allocation. Multiple concurrent local
# callbacks may share this runtime; they share its game slots and pending cap.
with InferenceRuntime(execution, Path(".runs/eval/allocation")) as inference:
    result = evaluate(
        EvaluationRequest(
            key=key,
            candidate=checkpoint_registration,
            opponent=anchor_registration,
            checkpoint_paths=checkpoint_paths,
            deal_seeds=(101, 102),
        ),
        out_dir=Path(".runs/eval/panel"),
        inference=inference,
    )
```

`experiment.checkpoint_evaluation.evaluate_checkpoint(..., execution=execution)` evaluates its complete
multi-deal panel through this path. It also accepts an existing `inference`
runtime. `benchmark.evaluate(..., execution=execution)` and
`duel.evaluate(..., execution=execution)` connect Etude's spawned callbacks to
one authenticated local Unix-socket endpoint. The allocation owns the bounded
executor, game slots and single serving process. Etude worker count cannot
multiply those limits. Derived manifests and complete match results remain
separate from original checkpoint coordinates.
Benchmark and duel entry points retain each invocation under a fresh
`allocation/<attempt>/` directory, allowing completed-deal reuse in the same
output directory without overwriting an earlier allocation receipt.

The inference process alone loads served checkpoints and owns `PolicyInference`
scopes. Requests group by checkpoint/input/numerical identity, flush on a row cap
or maximum delay, and sample each CPU row using its recorded decision seed.
`seed_compute_envelope_id` preserves the original seed authority across serving
ablations; the new execution envelope is retained separately. CUDA transfers
whole batches back for CPU sampling. Numerical changes may still change sampled
trajectories; replay verifies Commands, not cross-device policy equivalence.

Native game state and replay execute in killable subprocesses. Workers must wait
for durable pending-Command acknowledgement before stepping. Inference waiting
uses the game's wall allowance. Service death/hangs fail active work without
player forfeit credit; unstarted identities are retained in `unattempted.json`.
Every inference call is capped by the request limit, caller deadline and allocation
remainder; waiting for capacity and a reply spends the same budget.
The allocation deadline also bounds replay. Incomplete replay is unavailable,
not an invented mismatch or a completed valid block.

## Quick local evaluation

Use a current-world flat checkpoint bound to the selected Allies/Lessons setup:

```sh
uv run python -m manabot.eval.local \
  --checkpoint .runs/small/policy.pt --opponent heuristic \
  --deal-seed 101 --out .runs/local-check --seconds 300 --workers 2
```

`--opponent` also accepts another local `.pt` file, `scripted_greedy`, or `random`.
Repeat `--deal-seed` to declare more complete deal blocks. The default block has
four correlated deck/seat legs; `--include-mirrors` requests all eight. For another
admitted native roster, supply `--matchup` with an `AuthoredMatchup` JSON file.
The checkpoint must already admit that setup; no policy or world conversion occurs.
A one-deal check reports its observed score and states that uncertainty is
unavailable from a single deal; correlated legs are not independent samples.
With multiple deals the CLI reports the paired-deal bootstrap interval and block
count. A small cohort remains a quick check, not reliable strength evidence.

The command needs neither training records nor provider credentials or the
`evaluation` extra. It uses CPU with one Torch thread in each bounded worker and
serving process. `--workers`, `--batch-rows`, `--resident-mib`, `--game-seconds` and
`--max-commands` set explicit limits. `--seconds` includes checkpoint preparation,
serving startup, play and replay; preparation runs in a killable child. It never
rents hardware, enters an Etude queue, fetches a remote checkpoint or starts a
callback endpoint. It exits 0 only for a complete replayed cohort, and 1 for an
incomplete attempt. An admission timeout retains `admission-failure.json`.
This whole-lifetime deadline is a limit, not a measurement of total CLI elapsed
time: checkpoint preparation is outside the Perf cell/allocation timers below.

Typed callers can prepare an existing request before allocating a local lifetime:

```python
from pathlib import Path
import time

from manabot.eval.local import checkpoint_request
from manabot.eval.evaluation import evaluate_local
from manabot.sim.execution import EvalExecution

request = checkpoint_request(
    Path(".runs/small/policy.pt"), opponent="heuristic", deal_seeds=(101,),
)
result = evaluate_local(
    request,
    limits=EvalExecution(device="cpu", max_active_games=2,
                         deadline_unix=time.time() + 300),
    out_dir=Path(".runs/local-panel"),
)
```

`EvaluationRequest` also accepts existing `PlayerRegistration` and `EvaluationKey`
values, checkpoint paths, exact deals and matchup settings. Its registrations
retain their original seed authority when bound to compute. An unknown checkpoint
training seed remains `None`; it is never inferred from evaluation deals. The CLI
saves `request.json`, which can be passed to `--request` with a new output
directory for a fresh full repeat. Repeated seeds do not become independent
samples merely because execution is fresh.

`evaluate(request, out_dir=..., inference=...)` is the same execution entry point
used by monitoring and managed benchmark/Duel callbacks. It accepts an existing
`InferenceRuntime` or allocation endpoint. Optional `card_evidence_dir` retains upstream verified card evidence during bounded
replay; `trace_card_tables` can project its references alongside outcome tables.
The lower-level `eval.match.play_cell` also preserves explicit leg subsets.
Both public evaluators require a new
output directory and reject an existing one before writing or executing anything;
this includes empty, completed and failed attempts. The monitor wrapper retains training
coordinates and its frozen registration values; Duel retains its equal-update,
world, inference-budget and study-cohort restrictions.

`evaluation.json` retains the typed request, effective execution, rows, status,
elapsed wall time and paired-deal intervals. Incomplete cohorts never emit a
score interval. Stored bootstrap values retain their historical definition,
including the degenerate range for a single deal; that range cannot estimate
between-deal uncertainty. Native journals, Commands, source identities, replay and Perf
records use the existing arena formats under the panel directory. A CLI panel
is at `<out>/panel`; pass that directory to the report builder below. A small local
check is labeled `local-check-not-admission`, not promotion or held-out evidence.

## Continue an interrupted evaluation

```sh
uv run python -m manabot.eval.local --status .runs/local-check/panel
uv run python -m manabot.eval.local --continue .runs/local-check/panel --seconds 300
```

After 40 of 100 valid complete deals, continuation runs only the missing 60.
The local command reads the original opponents, setup, seeds and compute settings;
only the deadline changes. It needs no scheduler, cloud credentials or callback.
A partial deal is rerun whole. Inspection reports completed deals and produces
paired-deal intervals only when the full requested cohort is complete.

Typed callers use `evaluation_status(directory)` and
`continue_evaluation(directory, execution=limits)`. Managed callers pass their
existing `inference` runtime or endpoint instead. Deadline and instrumentation
can change, but CPU/CUDA, numerical settings, batching and residency limits must
match the original compute contract. Changing the request requires a fresh
evaluation. Relocated checkpoint paths may be supplied explicitly; their digest
and size are rechecked before continuation.

The first attempt's `evaluation.json` freezes the logical identity and full
request. Continuations retain new receipts beneath `attempts/<sequence>/` and
never replace earlier traces or results. `evaluation_status` derives the current
result from these attempts; the initial receipt continues to describe its own
attempt. Each completed block has an immutable Command shard and a row digest.
Replay publishes blocks incrementally, so coordinator failure cannot discard
already published blocks. Corrupt evidence and duplicate completed blocks fail
validation. Original attempts retain incomplete games and their failure details;
those games do not enter the combined statistical population.

Filesystem claims admit one writer. Managed execution also claims its attempt;
continuation refuses a still-running supervisor and seals old attempts against
delayed callbacks. These are local filesystem claims, not distributed leases.
Moving an evaluation requires retaining its entire directory, including block
receipts, traces and attempts. Historical receipts without a continuation identity
remain readable; they are not silently upgraded or combined.

Etude still owns match scheduling and its completed-match cache. Explicitly
re-entering a benchmark or Duel continues a failed match through this validator;
a successful cached match adds no sample. The callback transport can also run a
multi-deal request through the same API. This does not provision remote compute
or create a second durable queue. Performance receipts remain per attempt;
reusing prior blocks does not create another throughput measurement.

## Performance evidence

Each cell writes `performance.json`, `rows.json`, original journals and Command
traces. Each allocation writes `execution.json`, `allocation.json` and coarse
`serving.json` snapshots. Receipts include phase populations, bounded first-span
samples, model loads/catalog builds, batch occupancy, queue age, transfer bytes,
request latency and memory. Worker and serving spans join by request/batch ID.
Full counters continue after sample caps. Snapshots occur at most once per
second during serving, plus finalization; normal instrumentation adds no device
synchronization. Native-step timers include rules work and observation encoding,
not pure Rust/Python binding overhead.

Set `diagnostic_batches` to a bounded positive count for CPU/CUDA profiler traces.
CUDA diagnostics also record stream event duration with an explicit wait. These
runs include profiler overhead and cannot be mixed with ordinary throughput.
Device events have no host epoch alignment; the profiler trace is the linked
CPU/GPU timeline. Unmeasured kernel counts, pure binding cost and billing remain
explicitly unavailable.

Render retained evidence through the evaluation-only report viewer:

```sh
uv run python -m experiments.report_prototype.build \
  --eval-receipt .runs/eval/panel --out .runs/eval/report
```

Repeat `--eval-receipt` to compare jobs/seeds. `report-run.json` can supply a job,
seed and shared serving receipt path; ordinary callers retain that join.
The report copies its evidence links and diagnostic traces into its output.
It defaults to During games, grouping measured phases into Inference, Game engine,
Evidence and replay, and Other and unmeasured; selecting an activity reveals its
phases. Comparison elapsed, Game CPU usage and Inference service expose separate
clocks. Concurrent game time and inference waits overlap other views and cannot
be added into one elapsed total. Missing seed timers are never averaged as zeros. Shared
allocation cost appears once, remains explicitly unallocated to jobs, and never
turns an estimated dollar value into a bill. Training reports use Etude experiment files. The former prototype training
builder and definitions are removed; this evaluation-only adapter retains the
accepted view because Etude 0.2.5 does not yet accept these resource-specific
evaluation receipts. No production Etude evaluation-performance API is claimed.

The measurement boundaries matter when comparing costs:

- Cell elapsed starts inside `eval.match.play_cell`, after initial comparison
  validation, and ends at its performance snapshot after replay. It excludes
  earlier checkpoint registration and later result persistence; an externally
  owned runtime may already be warm when the cell starts.
- Allocation elapsed spans `InferenceRuntime` construction through shutdown.
  It includes serving startup and work/idle time within that lifetime, but excludes
  the CLI's earlier checkpoint preparation, provider provisioning and any paid
  lifetime outside the runtime. The serving-process timer is a separate interval.
- `evaluation.json` records evaluator elapsed time before cohort statistics and
  final receipt writing. It is neither the CLI lifetime nor a provider bill.

Report throughput uses these measured intervals and only the included cells.
Local preparation loads checkpoints for admission before the serving process
loads its resident policies; a resident load counter does not count that earlier
work. Whole-job latency, whole-job cost and end-to-end speedups need separate
measurements covering preparation, finalization and the actual paid lifetime.
Current runtime receipts supply neither billing amounts nor hourly rates, so
their cost fields remain unavailable.

## Local calibration

```sh
uv run --extra evaluation python -m manabot.eval.serving_benchmark \
  --plan .runs/eval-serving/plan.json --out .runs/eval-serving/result
```

`BenchmarkPlan` is the executable plan schema. Plan fields are required;
`EvalExecution` and `EvaluationRequest` retain their documented defaults.
It requires four explicit configurations
(`cold_scalar`, `resident_only`, `resident_catalog`, `batched`), three interleaved
repetitions, named panels wrapping the same `EvaluationRequest`, retained-root population, numerical
and action-agreement tolerances, speedup and instrumentation-overhead thresholds,
process-tree memory cap and absolute deadline. Cold scalar uses original
per-game checkpoint construction. Each panel uses the shared evaluator's fresh
attempt and complete-cohort validation; replayed partial panels cannot pass.
Every repetition/failure is retained; numerical
and instrumentation probes run outside throughput timing, on identical retained
inputs and row seeds in killable processes. Memory limits include descendants.
No hardware is acquired, training launched or durable demand queue created.

The CLI reports whether the declared thresholds were met, not an automatic
adoption decision. Representative small/larger models, two-checkpoint mixtures,
CUDA execution and an agreed performance target are still necessary before a
capacity recommendation. CPU fixture tests establish correctness only.

Headless report proof: `./scripts/verify-eval-performance-report` (requires the
locked `academy/ui` npm dependencies and Playwright Chromium). Gate owns the full
affected suites and representative hardware calibration.

## Accepted direction and later increments

Jack Heart's accepted direction (2026-10-24) makes checkpoint-versus-checkpoint
evaluation at growing model sizes the primary capacity workload. Small local
checkpoint-versus-heuristic checks retain the same evaluation semantics without
training, cloud, queue or callback machinery. The single-machine inference and
real Perf reporting software is implemented; CPU fixture evidence does not
establish CUDA parity, capacity suitability or a performance gain.

The accepted cloud startup policy is a target of at least two hours of estimated
eligible useful work on the chosen host/model mix before a server starts. It is
not a game-count threshold or a minimum paid lifetime: idle compute should be
released when work finishes. Hardware, spending, urgent overrides and maximum
queue wait remain unselected.

That startup gate remains **required follow-up work**, separate from the
implemented continuation increment. Inspection of Etude SDK 0.2.4 on 2026-10-24
found that `Scheduler._plan` sizes rentals from job/cohort time ceilings,
parallel slots, setup allowance and dollar caps. `OnDemand` and the public client
expose no measured useful-backlog threshold or estimator hook. `tournament.hold`
runs on existing compute and cannot gate provisioning. Etude's launch scheduler
must check eligible remaining useful seconds for the host/model mix under its
existing supply lock before creating a new rental; Manabot needs measured
workload estimates for that integration. Reusing existing compute and releasing
non-kept idle sessions remain independent of the startup threshold. Local checks
remain independent too; no alternate queue substitutes for this capability.

The following increments remain designs to settle, not selected infrastructure:

- **Compute policy and cluster operation:** compare scale-to-zero, a warm pool,
  and a warm baseline plus bursts against the same checkpoint arrivals, model
  sizes and requested outcomes. Measure both opponents' residency, CPU supply,
  load/drain time, idle paid time and whole-host costs. One allocation currently
  uses one CPU/CUDA device and serial policy batches; per-policy device routing,
  multi-host placement, replenishment and budgeted cluster operation remain later
  work. No topology, provider or purchasing commitment is selected.
- **Durable Etude queue:** a controllable API with ordered default-priority rules,
  explicit overrides, inspectable ordering, fenced claims, cancellation and
  producer cutover. Studies may span priorities; active training checkpoints
  should have a middle default. Exact priority numbers remain unselected.
  Resident-model affinity may break ties only with bounded waiting. The current
  bounded local work list and callback endpoint are not that durable queue.
- **Adaptive evaluation allocation:** checkpoint thinning/expansion and
  statistically justified stopping need their own design. Fixed studies keep
  their declared populations, study rules and frozen identities.
- **Further serving:** stateful compound/search support, output caching, native
  inference or specialized kernels require measured bottlenecks and matched
  validation before adoption.

Capacity acceptance still needs representative checkpoint pairs and growth-size
receipts, arrival/cadence and distinct-model counts, latency and budget targets,
numerical tolerances, instrumentation-overhead and improvement thresholds, and
CPU/CUDA hardware evidence. These follow-up intents authorize no purchases,
rentals, training, scientific execution or automatic adoption. Production Etude
report integration is also separate from the executable prototype.
