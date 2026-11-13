# MPS retest implementation

## Outcome and constraints

Measure the bucketed recipe across CPU float32 (CPU4 laptop / CPU8 M2), MPS float32, both working MPS
half dtypes and MLX float32. `experiments/mps_retest.md` owns the full matrix
and deliverables; `questions.md` preserves the exact steers. Torch is the
incumbent. Recommend from measurements, without an adoption rule or defaults
change. No performance or real-MPS correctness conclusion exists yet.

The operator owns the USD25 rental cap, allocation, release and cost receipt.
The newest steer permits captures and all fitting arms on the quiet laptop
while M2 capacity is pending. Use separate host evidence; require load <16
before laptop blocks. Each width/stream group still needs three replays per
arm within one hour. If M2 never arrives within 12 hours, report laptop-only
measurements with M2 explicitly unmeasured (deadline in `questions.md`).

## Implementation owners

`manabot/backends/capture.py::capture` records the real update for both this
experiment and the pilot. `experiments/mps_retest.py::capture` selects the
recipe; `_quiet` enforces the laptop load limit. `bench` admits per-call parity
receipts before model allocation. `replays` interleaves arms across three
repetitions; `_host_report` selects latest receipts and checks coverage and
the same-hour span. `report` writes host-separated artifacts, using the named
`full_context.png` for a sole host, including the laptop-only fallback.
No rental subsystem is needed.

## Implemented and checked

Capture/replay, bucketed preparation, half-precision output handling and
whole-update completeness are implemented. Report generation now shows every
per-call status; `comparisons*.md` exposes first-call/setup/warmed timings,
memory scopes, gradient/logit/value/loss differences, sampled choices and all
update repetitions. Failed attempts supersede earlier successes. Receipt
coverage is separate from correctness; timings with differing gradients remain
visible and require interpretation.

Receipts carry capture host and actual capture threads separately from replay
host. The shared engine component uses four CPU threads; replay uses four on the laptop, eight on M2.
Cross-host charts explicitly name the M2 engine origin. Unknown provenance
leaves coverage incomplete; conflicting capture origins are refused. Host
charts remain separate. Phase legends survive missing first-width bars.

2026-11-12 check: `uv run --extra reports python .runs/mps-report-check/check.py`
passed with synthetic M2/M4 receipts, actual PNG rendering and assertions for
status/missing cells, first-call/memory/correctness tables, latest-attempt errors,
capture thread provenance and conflicting-provenance refusal. PNG inspection
found and fixed parity-label/title and phase-legend overlaps. These fixtures
execute no model or game and are not performance evidence. Earlier five injected
completeness cases passed (all measured, one/both half unsupported, both half
OOM, unsupported CPU). Earlier focused implementation checks are recorded in
commit 9a83ac5 and its preceding scratch history; gate still owns affected suites.
Ruff format/check, `uv run python -m py_compile experiments/mps_retest.py` and
`git diff --check` passed; `lf context` is within both scratch limits.

2026-11-12 compress: `uv run --extra reports python .runs/mps-compress-check/check.py`
passed seven coverage cases plus split unsupported repetitions, latest-failure
supersession, median phases, same-hour refusal and unknown provenance. Re-running
`.runs/mps-report-check/check.py` passed; six PNGs and both comparison tables
matched the pre-compress bytes exactly, with equivalent JSON after sorting
missing-cell labels. These checks use synthetic receipts only. Compilation and
Ruff passed for `mps_retest.py` and the pilot (one empty section header removed).

## Remaining execution and acceptance

1. Begin the w384l8 256-stream capture when one-minute load is below 16,
   with four capture threads and a configured 20 GiB budget (peak unmeasured).
   The capture needs a real Etude run identity; `main` refuses dirty executable
   source. The current method changes must be checkpointed before any run.
   Parity, per-call timing and triple replays follow at selected widths.
2. Establish whether a complete width/stream group fits the one-hour window.
   `replays` advances isolated persistent arms one call or learner step at a
   time. Three complete blocks must still fit the same hour. A >2× endpoint
   load drift repeats the entire block once, retaining both attempts and
   selecting the lower mean endpoint load. All arms share that selection.
3. Capture 512/1024 streams; attempt both MPS half dtypes and the full matrix.
   Preserve errors; fix implementation failures. A 900-second timeout is
   failure, not OOM. Real MPS dtype support and gradient agreement are unverified.
4. Run the M2 comparison when the operator supplies access; cite its cost
   receipt. Keep per-host captures/charts separate. If no M2 arrives before
   2026-11-13 13:12:38.971 UTC, report laptop measurements and M2 not run.
5. Publish captures, receipts, tables and actual charts with `manabot.data`;
   embed the named full-context chart and explain measured results/limits and
   recommendation in the report and backend README. No default changes.

The inherited CI Task notes were removed from scratch; their complete record
is in `0ed4f2cb1:scratch/ci-under-8-minutes.md` (already landed work, no MPS
acceptance).

Gate owns affected suites and final acceptance. No Wave is bound. Publication
and Task movement belong to later Graph steps. Implementation checks do not
complete this experiment.

2026-11-13 realign: refreshed `origin/main`; no additional upstream commits
are missing from this checkout. Task comments add no direction beyond the
measurement-window steer already quoted in `questions.md`. Rechecked capture,
parity admission, replay ordering, coverage and source-freeze paths against
the plan; no executable repair was needed. Existing controlled checks remain
implementation evidence only. No model or game ran in this pass. The next
uncertainties are real capture completion, MPS half support/gradient agreement
and whether a full replay group fits one hour; no new product decision is
needed unless measurements expose a conflict with those constraints.

2026-11-12 realign: refreshed `origin/main`; it adds no commits beyond this
checkout. Task comments still end with the operator's rental/laptop direction.
The historical Task snapshot's adoption rule, narrow matrix and laptop ban are
superseded by the exact comments in `questions.md`. The report and code retain
the full steered matrix and unchanged load, provenance and same-hour constraints.
Corrected the sole-laptop artifact filename so the fallback can supply the named
chart. Measurements, publication, rental cost and recommendation remain open.
An isolated `report` dispatch check passed sole-M2, sole-M4 and combined-host
filename cases with rendering replaced by controlled fixtures; it ran no model
or game. Ruff check/format and `git diff --check` passed. Scratch remains within
both `lf context` limits. Actual chart rendering checks are the earlier results
above; this pass changed only output naming.

2026-11-12 sanity check: `uv run --no-sync python
.runs/mps-laptop-check/check.py` passed quiet-capture provenance, refusal at
loads 8/90 before output creation, and five laptop CLI paths including small
widths, with execution replaced by controlled fixtures. Ruff check/format
and `git diff --check` passed. No model/game ran. Final 01:17 UTC load was
105.66; the 256-stream capture is ready for the operator's quiet window.

2026-11-12 second compress: receipt admission now precedes model allocation and
capture reads, so missing/unsupported parity needs neither. Dispatch derives
the worker command from its phase, and completeness applies the same-hour
requirement to both hosts. `uv run --no-sync python .runs/mps-simplify-check.py`
passed injected admission, dispatch, timeout/failure, triple-replay and M2/M4
window checks; existing compress and laptop fixtures passed too. Compilation,
Ruff and `git diff --check` passed. No models, games or measurements ran.

## Delete — do not maintain

Removed the reference writer's immediate disk reread and repeated host/capture
probes during replay admission. Workers still record their own provenance.
Removed the serial timed-arm comparison path; `bench` and `update` are
worker generators driven by the one interleaving coordinator. Single-arm
timing is refused at the CLI. Retain the isolated parity path and CPU arrays.
No further deletion is planned. The shared capture owner remains
`manabot/backends/capture.py`; preserve prior evidence and the production
off-CUDA refusals. No alternate capture or rental implementation.

2026-11-13 resumed preflight: `git diff --check` passed and pending source/report
changes were checkpointed as `849769d6f`. No code changed after the previously
passing focused checks, so they were not repeated. No measurement process was
started; load remained above 8. Measurements and acceptance remain open.

2026-11-13 compress: `_reference` returns the CPU arrays it just retained;
readers still validate provenance before loading. `replays` reads admission
metadata once, keeping fresh parity-ledger reads and all worker load checks.
`uv run --no-sync python .runs/mps-reference-check.py` passed exact array/dtype
round-trip, provenance refusal, no writer reread and two-width replay ordering;
`.runs/mps-simplify-check.py` also passed. Compilation, Ruff check/format and
`git diff --check` passed. Only controlled fixtures ran, no models or games.

2026-11-13 method update: implemented the latest operator steer (exact quote
in `questions.md`). Load admission is <16, laptop threads are four, workers
interleave calls/steps, and capture/comparison retries preserve both attempts
and select the quieter complete block. Tables and chart captions label loads.
`uv run --no-sync python .runs/mps-steer-check.py` passed injected boundary,
interleave order, retry selection, pending-repeat coverage, worker cleanup,
actual pipe handshake and evolving replay-state checks; no model/game ran.

2026-11-13T02:01:15+00:00 preflight: one-minute load 64.67, above the new 16 limit. No
capture or timed arm started. M2 access notes still have the 2026-11-12
08:47:58 PST modification time. No measurement process is running. The next
block remains the four-thread w384l8 256-stream capture, configured 20 GiB.
`uv run --no-sync python .runs/mps-steer-extra-check.py` passed per-call resets,
independent evolving replay state, actual worker pipe exchange, injected timeout
cleanup and parity timeout classification. The rendering check passed and the
corrected full-context PNG was visually inspected; labels no longer overlap
neighboring panels. Ruff, compilation and `git diff --check` passed. All proofs
use controlled fixtures, not model/game timings. Gate still owns affected suites.
