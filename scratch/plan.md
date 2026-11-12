# 855bff — standing pipeline speed

## Outcome and constraints

One bounded local command measures training and shared-inference evaluation across
`w64`, `w128l4`, `w160l8`, `a384l8`, records a reference commit and suite interface,
and compares both sources interleaved on one machine in one job. It reports stage
attribution, batching utilization and supplied-price rates.

No learning, collector, native, frozen recipe or evidence changes; no paid runs,
rentals, induced contention or cross-runner timing gate. CPU runs prove the
harness, never GPU performance or a speed claim. Batching optimization,
learning-equivalence admission and long-run slowdown acceptance are separate.

## Settled design

The reference direction in comment `262b941c-2188-4426-8ff0-35f7e52eaa10`
specifies no standing host (exact words in `scratch/questions.md`). Keep three
concepts: a report retains attempts and samples; a reference selects source and
worker interface; a comparison derives ratios from this job's admitted samples.
Provider selection is outside this Task and grants no spending authority.

- `scripts/bench_training_update.py --suite` uses the existing `_execute` training
  executor. `scripts/bench_pipeline.py` owns `WORKLOADS`, `run_suite` and
  `_measure_attempt`: one bounded worker tree, cleanup and retained failures.
  The benchmark owns its current-world Elemental identity: v6/512 facts,
  float32, 256 streams and three updates; evaluation uses initialized policies
  and four replay-verified BG/GR legs. New workloads use the same coordinator.
- `training_reading` excludes update one from timings through
  `update_metrics.read_updates` / `stage_rows`. Batching counters include warmup;
  calls/update is a stage average. `SeatRoutedCollector._collect` / `_settle_step`
  retire streams after their quota and terminal game, shrinking active batches
  toward the tail. Larger serving limits do not remove this idle cost.
- `Reference` contains only `commit` and `suite_interface: 1`.
  `reference_from_report` requires a complete clean standalone reading.
  `_require_full_ladder` requires all sizes and all device-supported workloads
  before recording or comparison workers start. Subsets remain diagnostic.
  No baseline has been recorded or committed; the retired persisted-Report
  design had no stored baseline needing migration.
- `admit_reference` reads the other checkout's literal `SUITE_INTERFACE` without
  executing it. Both clean sources need their own matching environment and
  release extension. Historical hosts and rates never supply the denominator
  or bind later jobs' options.
- `compare` alternates reference/candidate across three repetitions under one
  deadline, retains every sample and selects whole minimum-duration samples.
  It checks fresh workload, host (including Python/Torch), device, precision and
  behavior identity. Both workers receive the same requested options. CPU/CUDA
  support both workloads; MPS supports training only, without substitution.
- `docs/development.md` owns usage, recovery and the named one-job CUDA check;
  training/evaluation docs link it. The harness tests are in the evaluation CI
  group. No CI timing comparison is introduced.

## Evidence and limits

Merge `0a561319c` brought Python 3.13/dependency changes (`b7c102f2c`) and MTG-273
outcome tables (`1659b0e29`). `read_updates` still owns outcome-table expansion;
timing keys remain intact. MTG-273's long-run acceptance is separately documented
in `experiments/self-play-collection-273.md`.

Merge `e3e68601c` incorporates main through `668555c09`: the 0.2.1 release,
MTG-272 native hand snapshots/reset bookkeeping (`0b296e617`) and the
`experiment-performance` workflow (`830161795`). The metric reader and frozen
size declaration are unchanged by this sync. `VectorEnv.reset` and
`CapturedVector` now depend on the merged native API; the next clean proof needs
a matching rebuilt extension, not the extension used for pre-sync QA.
`.lf/skills/experiment-measure.md` already anticipates this Task's shipped
ladder/reference; it remains on per-size attribution until that delivery.

The early dirty `402dfd5b2` CPU subset proofs preceded that merge: a measured
second training update of 6.144 s / 399 learner decisions, whole-stage 1.743
rows/call and 0.1285 idle fraction; after correcting deck keys, four replayed
evaluation games at 0.5197 games/s in a 13.15 s invocation. These establish narrow
integration only. Commands and retained failure facts are in `scratch/questions.md`.

QA on dirty `2f24ceb09` (2026-11-11, laptop CPU) timed out after 3,598 s under the
old 3,600 s default, retaining six rows. The separate a384l8 run finished both
rows in 2,548 s. Completed attempt times sum to 5,466.9 s (≈5,500 s), not one complete
invocation. Six ladders imply an estimated ≈9 h comparison, not a measurement.
w64 overlapped an accidental QA run and is perturbed. a384l8 reported 685
calls/update, 77.8 rows/call and 0.696 idle. Full receipts and readings:
`.runs/qa-ladder-001/report.json`, `.runs/qa-a384l8-001/report.json`, `scratch/qa.html`.

## Implemented deadline repair

Completed: removed the shared 3,600-second suite CLI default and stale examples.
Standalone suites now default to 7,200 s; comparisons to 36,000 s. Explicit
`--seconds` overrides either without changing workload identity. Startup/cleanup
accounting and failures remain retained. The single-recipe deadline is unchanged.
`SUITE_SECONDS` and `COMPARISON_SECONDS` own CLI selection and help;
`test_suite_cli_deadline_defaults_and_override` checks standalone, recording and
comparison budgets, explicit overrides, startup accounting and unchanged config
with injected workers. Invalid deadlines retain errors before workers start.
The previous step's requested repair is complete; these checks do not establish
that either budget suffices for a complete real invocation.

## Delete — do not maintain

No remaining file, symbol or exclusive-fixture deletion targets. Archived context
and exact decisions remain located by `scratch/questions.md`.

## Concept review — 2026-11-11, after main sync

No design change or open product choice. Keep the measure → record source →
compare interaction. An attempt must remain distinct from its admitted sample:
a timeout still has logs, elapsed time and cleanup evidence, but no rate. Recovery
starts a fresh job and retains the failed report; stitching successful rows from
separate jobs would break the one-job comparison contract. There is no additional
resume or reference-host lifecycle to build. The repaired deadlines bound failure;
they do not settle whether the workflow is practical. The remaining acceptance
below must answer that with a complete invocation, without shrinking the workload.
Usage docs and the experiment-measure skill already describe the intended boundary;
no proposed usage or implementation change is needed. Loop-or-next can advance
from concept review; this does not complete measurement or delivery.

## Remaining acceptance — gate/measurement

1. On clean source with matching release extension and locked environment,
   retain one complete default CPU ladder, actual invocation time and eight rows.
   Resolve capacity/runtime failures without changing workload identity. Record,
   review and commit `manabot/perf/baseline.json` through `--record-reference`;
   keep measured rates in the report. This establishes a local harness proof.
   Recheck w64's unattributed collection time without the earlier accidental
   overlap; overlap is a possible cause, not an established explanation.
2. Run a real same-host interleaved reference/candidate comparison with matching
   options. Establish repeatable behavior admission and retain incomparable
   results/failures. The full CPU/CUDA comparison has 48 worker attempts sharing
   one deadline; controlled fixtures do not prove its runtime or repeatability.
3. Gate checks merged metric-reader integration and the configured evaluation CI
   collection. CUDA remains a named same-pod check on an already provisioned
   host. CUDA/MPS have not been exercised here and do not block this repair.
   No speedup or learning-equivalence claim is established.

Retained pre-sync validation: `uv run --locked --extra dev --extra evaluation pytest tests/training/test_pipeline_speed.py -q`
— 57 passed in 1.57 s. Focused Ruff lint/format, Python compilation, CLI help,
`uv run --locked python scripts/check_loopflow_workflows.py` and `git diff --check`
passed. This supersedes QA's earlier 46-test count; merged integration checks
remain with gate. Realignment re-read the two receipts and the deadline code and
tests without repeating workloads. No product decision remains; CUDA/MPS and
provider selection retain the limits above.
