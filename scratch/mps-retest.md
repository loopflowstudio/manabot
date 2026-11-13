# MPS retest implementation

## Outcome and constraints

Measure the bucketed recipe across CPU float32 (CPU4 laptop / CPU8 M2), MPS
float32, both working MPS half dtypes and MLX float32.
`experiments/mps_retest.md` owns the full matrix and deliverables;
`questions.md` preserves the exact steers. Torch is the incumbent. Recommend
from measurements, without an adoption rule or defaults change. No performance
or real-MPS correctness conclusion exists yet.

The operator owns the USD25 rental cap, allocation, release and cost receipt.
Captures and all fitting arms may run on the laptop while M2 capacity is pending.
Start laptop blocks regardless of load; record endpoint loads. Repeat >2× drift
once, retaining both attempts and selecting all arms from the lower mean
endpoint-load attempt (ties keep the first). Each width/stream group still
needs three replays per arm within one hour. Keep hosts separate. If M2 never
arrives before 2026-11-13 13:12:38.971 UTC, report laptop measurements and M2
explicitly unmeasured. The deadline's derivation is in `questions.md`.

## Implementation owners

- `manabot/backends/capture.py::capture` records the real update for both this
  experiment and the pilot. `experiments/mps_retest.py::capture` selects the
  recipe; endpoint loads describe the shared host without gating execution.
- `_interleaved` admits timed arms before starting workers. `_latest_parity`
  reads all arms once per block, scoped to host/software/source, capture and
  width; per-call admission also requires the row count. Retries share that
  evidence. Admission means parity executed, not that gradients agreed;
  numerical agreement remains a report judgment. A refused arm retains its
  parity failure status (including OOM) and names that prerequisite in its error.
  `bench` and `update` execute admitted calls/steps in isolated
  persistent workers; `replays` schedules three repetitions. Workers retain
  their own provenance. All models remain resident, so OOM describes this
  shared-memory setup, not standalone arm capacity.
- `_host_report` selects latest receipts and checks coverage and the same-hour
  span. Failed attempts supersede successes; coverage is separate from
  correctness. `report` writes host-separated tables/charts, using the named
  `full_context.png` for a sole host, including the laptop fallback.
- Receipts distinguish capture host/threads from replay host/threads. The shared
  engine component uses four CPU threads; replay uses four on the laptop,
  eight on M2. Cross-host phase sums identify the capture origin. Unknown
  provenance leaves coverage incomplete; conflicting origins are refused.

## Remaining execution and acceptance

1. Begin the w384l8 256-stream capture unconditionally, with four threads
   and a configured 20 GiB budget (peak unmeasured). It needs a real Etude run
   identity and committed executable source, including the parity-status repair.
2. Establish whether a full width/stream group fits one hour. Three complete
   blocks must fit, including interleaving gaps. A 900-second worker-call timeout
   is failure, not OOM. Preserve errors and fix implementation failures.
3. Capture 512/1024 streams and attempt the full matrix, including both MPS
   half dtypes. Actual dtype support and gradient agreement remain unverified.
4. Run M2 comparisons when access arrives, subject to the fallback deadline;
   cite the operator's cost receipt and keep per-host captures/charts separate.
5. Publish captures (including unused retry attempts), receipts, tables and
   charts with `manabot.data`; embed the named full-context chart, interpret
   measured results/limits and recommend in the report and backend README.

Gate owns affected suites and final acceptance. Publication and Task movement
belong to later Graph steps. No Wave is bound. Implementation checks do not
complete this experiment.

Upstream reviewed at `e82a20033`: the only new main commit prepares v0.2.3
release metadata and upgrade docs; it changes no retest model or measurement
behavior. Keep the measurement revision explicit if that metadata is synced.
The initial Task description still carries the superseded adoption rule and
laptop prohibition; the dated exact steers in `questions.md` and the experiment
report describe the current scope.

## Delete — do not maintain

Removed the duplicate pilot capture, serial timed-arm comparison path,
reference writer's disk reread and repeated host/capture probes during replay
admission. Removed worker-side parity admission: the coordinator already owns
that guard before allocation, so it now also attaches the per-call parity
metrics. No second ledger scan or competing admission path inside a worker.
Keep isolated parity execution and retained CPU arrays. No alternate capture
or rental implementation; preserve frozen evidence and production off-CUDA
refusals. No remaining deletion target is identified.

## Validation and execution evidence

All checks below use controlled fixtures, not performance evidence. Historical
commands and full results through the interleaving change are retained at
`348c8f26b:scratch/mps-retest.md`; earlier implementation checks are in
`9a83ac5:scratch/mps-retest.md`. The removed, already-landed CI Task notes are
in `0ed4f2cb1:scratch/ci-under-8-minutes.md` and do not establish MPS acceptance.

- 2026-11-12: `.runs/mps-report-check/check.py` passed actual PNG rendering with
  synthetic M2/M4 receipts, missing/status cells, timing/memory/correctness
  tables, latest failures, capture provenance and conflicting-origin refusal.
  Visual inspection found and fixed overlapping labels/legends. The first
  compression preserved six PNGs and both tables byte-for-byte; summary JSON
  was equivalent after sorting missing labels. Sole-M2, sole-M4 and combined
  report filename checks passed. No real MPS execution was checked.
- 2026-11-13, before this compression: `.runs/mps-steer-check.py` and
  `.runs/mps-steer-extra-check.py` passed load boundaries, call order, whole-block
  retry selection, pending-repeat coverage, per-call resets, independent
  evolving replay state, actual pipe exchange and timeout cleanup. Rendering
  passed and the corrected full-context PNG was visually inspected.
- Measurement preflight, 2026-11-13T02:01:15+00:00: load 64.67 prohibited
  capture under <16. M2 access notes still had the 2026-11-12 08:47:58 PST
  modification time. No capture or timed arm started; no worker remained.
  Earlier load readings and source freezes are retained in `questions.md`.
- Current compression: `uv run --no-sync python .runs/mps-admission-check.py`
  passed provenance-scoped latest parity, failure supersession, refusal before
  worker creation, retained per-call metrics, one ledger read across retries
  and two-width replay admission. `uv run --no-sync python
  .runs/mps-steer-extra-check.py` passed the interleaving/capture/worker checks
  above after updating its admission fixtures. Ruff check/format, compilation
  and `git diff --check` passed. No model, game or measurement ran.
- Realignment preflight, 2026-11-13T02:09:13.955809+00:00: load 63.00 still
  prohibited capture under <16; M2 access notes were unchanged. Task comments
  supplied no newer direction. No capture or timed arm started.
- Realignment repair: `uv run --no-sync python .runs/mps-admission-check.py`
  passed with assertions for preserved parity failure statuses, no worker
  allocation, and OOM displayed as “does not fit.” Ruff check/format and
  `git diff --check` passed. These are controlled fixtures, not MPS evidence.
