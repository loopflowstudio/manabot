# 855bff — standing pipeline speed

## Outcome and constraints

One bounded local command measures training and shared-inference evaluation across
`w64`, `w128l4`, `w160l8`, `a384l8`, retains a reference commit and suite interface,
and compares real sources interleaved on one machine in one job. It reports stage attribution, batching utilization and
supplied-price rates. The default ladder's actual runtime must be measured;
the 3,600-second deadline is not evidence that it is practical.

No learning, collector, native, frozen recipe or evidence changes; no paid runs,
rentals, induced contention or cross-runner timing gate. CPU measurements do not
establish GPU speed. Batching optimization and long-run slowdown acceptance remain
separate outcomes.

## Implemented

The baseline now contains only `commit` and `suite_interface: 1`.
`reference_from_report` admits a complete clean standalone reading before
recording metadata; absolute rates and host facts stay in `Report`.
`admit_reference` validates the clean reference checkout and its literal
`SUITE_INTERFACE` declaration without executing its code. No historical report
or host participates in fresh sample admission. Both comparison sources must be
clean; the same requested suite configuration reaches both workers.

`scripts/bench_training_update.py --suite` keeps `_execute` as the training executor.
`scripts/bench_pipeline.py` owns `WORKLOADS`, `run_suite` and `_measure_attempt`:
one bounded worker tree, retained attempts and cleanup, fixed v6/512-fact input,
float32, 256 streams and three updates. Evaluation uses initialized checkpoints
and four replay-verified BG/GR legs. Docs link the command and CI includes its tests.

`training_reading` excludes update one from collection/learning timing, using
`update_metrics.read_updates` / `stage_rows`. Batching counters are cumulative:
calls/update is a whole-stage average, rows/call and idle fraction include warmup,
not individual update measurements. In `SeatRoutedCollector._collect`, complete-game
streams idle only after meeting their quota and finishing a game; active rows then
shrink toward the tail. The existing counters expose this without a second store.

`compare` alternates clean reference and candidate sources for three repetitions,
retains all samples and selects whole minimum-duration samples after workload,
host, device, precision and behavior admission. Python and Torch versions are part
of host identity. Historical durations alone cannot produce a speedup. A reference
checkout is explicit and must contain the suite interface. MPS supports training
only; CPU/CUDA support both workloads. No device substitution.

`_require_full_ladder` is shared by recording and comparison: all four
sizes and the device's supported workloads are required before workers start.
A successful subset remains a diagnostic reading. Rejection persists the error
without replacing a reference. Controlled tests cover CPU/CUDA/MPS admission,
subset rejection and full eight-row interleaving; actual device measurements
remain separate acceptance.

## Reconciled with main

Merge `0a561319c` includes the Python 3.13/dependency changes (`b7c102f2c`) and
MTG-273 outcome-table repair (`1659b0e29`). The latter preserves timing keys while
`read_updates` expands selected outcome-table references. Keep that reader as the
owner; no benchmark-specific compatibility layer is needed. Python version was
missing from host admission and is now recorded and compared.

The earlier narrow CPU receipts were produced on dirty `402dfd5b2` source before
this sync; they demonstrate integration, not the merged source or a standing
reference. MTG-273's bounded index fixture does not prove long-run flat RSS or
collection time; neither will this three-update ladder. See
`experiments/self-play-collection-273.md` for that separate acceptance.

## Delete — do not maintain

- Removed `Report` as the persisted baseline and historical sample/host admission
  in `admit_reference` and `compare`, including their exclusive test fixtures.
  No baseline has been committed, so no stored-data migration is needed.
- Keep `Report` for run evidence, all attempts and failures, complete-ladder
  recording validation, explicit clean reference checkout admission, and fresh
  workload/host/behavior checks with whole-sample minima.
- Removed the duplicate record-and-compare rejection test; the parameterized
  admission test covers it with controlled sources and baseline preservation.
  Full-device reference admission now runs through the recording test, covering
  persisted metadata and receipts for CPU/CUDA/MPS. No deletion targets remain.

## Remaining acceptance

1. On clean source with its matching release extension and locked environment,
   retain a complete default ladder, actual invocation time and all eight rows.
   Resolve capacity/runtime failures without silently changing workload identity.
   Label this local CPU run as a harness proof, not a speed claim or GPU evidence.
   Review and commit the commit-and-interface `manabot/perf/baseline.json`; no
   baseline exists yet. Keep absolute rates in the run report.
2. Run a real same-host interleaved reference/candidate comparison with matching
   suite options. Establish repeatable behavior admission; retain failures and
   incomparable results. Controlled fixtures alone do not satisfy this outcome.
3. Gate checks the merged metric-reader integration and configured evaluation CI
   collection. CUDA remains the documented one-job, same-pod comparison check
   on an already provisioned host; neither CUDA nor MPS
   has been exercised here. No speedup or learning-equivalence claim is established.

Latest focused validation: `uv run --locked --extra dev --extra evaluation pytest
tests/training/test_pipeline_speed.py -k 'comparison or reference or suite_interleaves or missing_or_failed'
-q` — 39 passed, 9 deselected in 2.88 s. Ruff lint/format, Python compilation,
CLI `--help` and `git diff --check` passed. Controlled CPU/CUDA/MPS fixtures
exercise a CPU-recorded reference on another host/device with different requested
options, one shared deadline, complete interleaving, clean sources, rejected
metadata/interfaces, and baseline preservation. No device workloads ran in this
implementation step; gate owns affected suites and remaining acceptance checks.

## Settled reference direction — 2026-11-11

Jack Heart rejected the proposed standing hosts: "i dont really thin kany of these
options are acceptable tbh". To the alternative of a reference commit with fresh
same-job controls he said: "no i think you have it right. this is a strong reason
to have at least one more stable gpu. maybe aws gpu perf is more consistent?"
Source: comment `262b941c-2188-4426-8ff0-35f7e52eaa10`.

There is no reference host to designate. The committed baseline names a reference
commit and suite interface. Each comparison reruns both sources interleaved on
one machine in one job (on one pod for a rented GPU). Absolute rates remain
per-run history; only same-job ratios can support comparison. CPU runs are free
harness proofs, never a speed claim. No rental, launch or spending is authorized.
Whether to add a stable GPU provider remains outside this Task.

This supersedes the earlier concept review's host-choice blocker and its
host-bound reference model, including the historical walkthrough in
`scratch/concept-review.html`. Preserve complete-ladder admission, retained failed
attempts, fresh workload/behavior checks and whole-sample minima. A failed or
subset run cannot replace the reference. The reference checkout remains explicit
and clean, with its own environment and extension. No implementation or
measurement was launched by the documentation step.

## Shipping docs — write-docs, 2026-11-11

The next implementation step must read these edited files in this checkout:

- `docs/development.md`, Training and evaluation speed: measurement example,
  commit-and-interface reference, fresh interleaved admission, recording and
  recovery, local CPU proof, exact one-job CUDA comparison command.
- `docs/training.md`, update-time guidance: batching counters and same-job
  comparison, with CPU scope explicit.
- `docs/evaluation.md`, Performance evidence and calibration: reference identity,
  same-pod comparison and historical rates.

The implementation now matches the reference behavior in these docs; the
full-ladder and real-source measurement obligations above remain open. The
existing CLI flags and full-ladder recording prerequisite are preserved.

Presentation: `lf open --host desktop --file docs/development.md --file
docs/training.md --file docs/evaluation.md` returned "No Desktop Files view
attached; nothing opened." Files remain at these paths. Reviewed the examples
against the CLI flags and checked the edited prose; `git diff --check` passed.
No workloads or tests were run for this documentation-only step.

## Compress — reference metadata, 2026-11-11

Flattened the comparison and coordinator loops with `itertools.product`, keeping
the same size/workload order and alternating reference/candidate order. The
receipt types and worker registry remain: attempts must retain failures without
samples, and workloads share one coordinator. Consolidated the redundant
reference tests described above.

Validation: `uv run --locked --extra dev --extra evaluation pytest
tests/training/test_pipeline_speed.py -q` — 46 passed in 5.75 s; focused Ruff
lint/format and `git diff --check` passed. No benchmark workloads ran. Full-ladder
runtime, baseline adoption and real-source comparison remain open.

Proposed correction for the read-only Infrastructure memory: replace its stale
“Reference-host designation” obligation with the settled commit-and-interface
direction above. Complete measurements and real interleaving remain required.
