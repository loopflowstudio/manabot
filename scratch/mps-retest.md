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
Start laptop blocks regardless of load; record endpoint loads. Repeat load increases over 2×
once, retaining both attempts and selecting all arms from the lower mean
endpoint-load attempt (ties keep the first). Each width/stream group needs three back-to-back replays per arm with
wall-clock and load endpoints; the latest steer removes the one-hour cap. Keep hosts separate. The latest steer removes host availability as a delivery gate: if M2 access
does not arrive, report laptop measurements and M2 explicitly unmeasured.

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
- `_host_report` selects latest receipts and checks coverage and records the replay
  span. Failed attempts supersede successes; coverage is separate from
  correctness. `report` writes host-separated tables/charts, using the named
  `full_context.png` for a sole host, including the laptop fallback.
- Receipts distinguish capture host/threads from replay host/threads. The shared
  engine component uses four CPU threads; replay uses four on the laptop,
  eight on M2. Cross-host phase sums identify the capture origin. Unknown
  provenance leaves coverage incomplete; conflicting origins are refused.

## Remaining execution and acceptance

1. Laptop 256-stream capture completed at `99210bf8d`; see execution evidence
   below. M2 capture is running as `bfb4d0e22dd7` from the same source in
   `/Users/ec2-user/manabot-mps`, with CPython 3.12.16, Torch 2.15.0, MLX 0.32.4.
   Use its independent `experiments/data/mps_retest/m2` evidence.
2. Run three back-to-back blocks at w384/256 on both hosts under the new
   directional retry rule. No one-hour cap remains. Keep old cancelled series
   separate; completed captures remain valid inputs. A 900-second worker-call timeout
   is failure, not OOM. Preserve errors and fix implementation failures.
3. Capture 512/1024 streams and attempt the full matrix, including both MPS
   half dtypes. Actual dtype support and gradient agreement remain unverified.
4. Run M2 comparisons when access arrives, if access arrives;
   cite the operator's cost receipt and keep per-host captures/charts separate.
5. Publish captures (including unused retry attempts), receipts, tables and
   charts with `manabot.data`; embed the named full-context chart, interpret
   measured results/limits and recommend in the report and backend README.

Laptop execution priorities through 06:17 UTC: w384 at 256/512/1024 streams,
then w768/w1024 at 256, then smaller widths. Report what was not reached.

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

- 2026-11-13 02:21 UTC: `c91b93dcc210` at `99210bf8d` attempted the
  capture at load 69.68; failed before games because the local native extension
  omitted `catalog_tokens.keys`. Failure retained in
  `experiments/data/mps_retest/laptop/capture-256-failed-c91b93dcc210`.
  Rebuilding the extension from this checkout before retry; launcher USD 0.00.
- Controlled high-load fixture passed: load 100 admitted, five arms interleaved,
  both endpoint loads retained, >2× boundary unchanged. Ruff and diff checks passed.

- `f800789ff669` completed the 256-stream capture at `99210bf8d`: 48,064
  decisions, 792 calls, 188 minibatches, 624.57 s capture wall, load
  77.81→85.09, no repeat; launcher USD 0.00. Native wheel rebuilt and
  atomically installed before this run. Replays/parity are next.

- Laptop `86e16cc560fc`, first w384/256 replay: all five arms completed
  in one interleaved block spanning 2,425.13 s. Whole-update seconds:
  CPU4 737.14, MPS f32 536.55, bf16 511.31, f16 472.41, MLX 427.30.
  All sampled-choice disagreement counts zero. Load 54.52→21.48 triggers
  a required whole-block repeat; these are not selected medians or final results.
  Three such blocks cannot fit an hour at the observed duration; acceptance
  must retain the actual span rather than relax the window.

- Laptop continuation `86e16cc560fc` cancelled after the first complete block
  and during its drift retry: observed 40.4-minute block conflicts with the
  three-block hour at that rate; M2 is now the primary host per the latest steer.
  The first laptop block remains provisional, with retry pending and no median.
  Do not describe laptop replay acceptance as met.

- M2 source transferred as a shallow Git checkout of `99210bf8d`, native
  CPython 3.12 extension rebuilt there and installed atomically. Capture
  `bfb4d0e22dd7` uses the isolated `.runs/mps-etude` launcher. Allocation and
  release remain operator-owned; allocation receipt is quoted in questions.

- M2 `bfb4d0e22dd7` completed its 256-stream capture: 1,223.33 s wall,
  48,064 learner decisions, 792 calls, 188 minibatches, 9.79 s shared
  engine/host work. Replay/parity run `5884f474a199` now measures w384.
  Laptop archive and bench receipts/references were published with manabot.data;
  captures are packaged byte-for-byte in `laptop/captures-99210bf8d.tar.gz`.
