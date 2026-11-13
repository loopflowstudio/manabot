# MPS retest implementation

## Outcome and constraints

Measure the bucketed recipe across CPU8 float32, MPS float32, both working MPS
half dtypes and MLX float32. `experiments/mps_retest.md` owns the full matrix
and deliverables; `questions.md` preserves the exact steers. Torch is the
incumbent. Recommend from measurements, without an adoption rule or defaults
change. No performance or real-MPS correctness conclusion exists yet.

The operator owns the USD25 rental cap, allocation, release and cost receipt.
The newest steer permits captures and all fitting arms on the quiet laptop
while M2 capacity is pending. Use separate host evidence; require load <8
before laptop blocks. Each width/stream group still needs three replays per
arm within one hour. If M2 never arrives within 12 hours, report laptop-only
measurements with M2 explicitly unmeasured (deadline in `questions.md`).

## Delete — do not maintain

Remove the obsolete laptop capture/small-width refusals in the harness and
its overflow-only documentation. Preserve the load guard, eight-thread replay
limit, four-thread capture, receipt provenance and separated host charts.
The shared capture owner has already replaced the pilot's inline capture;
`HostReport` is the one selection path. No rental subsystem is planned.

## Implemented and checked

Capture/replay, bucketed preparation, half-precision output handling and
whole-update completeness are implemented. Report generation now shows every
per-call status; `comparisons*.md` exposes first-call/setup/warmed timings,
memory scopes, gradient/logit/value/loss differences, sampled choices and all
update repetitions. Failed attempts supersede earlier successes. Receipt
coverage is separate from correctness; timings with differing gradients remain
visible and require interpretation.

Receipts carry capture host and actual capture threads separately from replay
host. The shared engine component uses four CPU threads; replay uses eight.
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

1. The laptop environment is prepared. Next is a real w384l8 capture at 256
   streams (four threads, configured 20 GiB budget; peak unmeasured), then
   parity, per-call timing and triple replays at selected widths. The 01:14
   UTC load was 71.57, so no measurement started. Return the timing block to
   the operator when busy; do not wait indefinitely or touch other Tasks.
2. Launch through the prepared isolated local Etude home
   (`ETUDE_HOME="$PWD/.runs/mps-etude"`) for a real run identity. Its local
   pool read succeeded; the incompatible default DB was left intact. No
   jobs are queued. Operator-owned AWS capacity is pending and access.md
   still names the released pilot.
3. Capture 512/1024 streams; attempt both MPS half dtypes and the full matrix.
   Preserve errors; fix implementation failures. Schedule three replays within
   an hour using observed durations. A 900-second timeout is failure, not OOM.
4. Run the M2 comparison when the operator supplies access; cite its cost
   receipt. Keep per-host captures/charts separate. If no M2 arrives before
   the deadline, report laptop measurements and M2 not run.
5. Publish captures, receipts, tables and actual charts with `manabot.data`;
   embed the named full-context chart and explain measured results/limits and
   recommendation in the report and backend README. No default changes.

The inherited CI Task notes were removed from scratch; their complete record
is in `0ed4f2cb1:scratch/ci-under-8-minutes.md` (already landed work, no MPS
acceptance).

Gate owns affected suites and final acceptance. No Wave is bound. Publication
and Task movement belong to later Graph steps. Implementation checks do not
complete this experiment.

2026-11-12 sanity check: `uv run --no-sync python
.runs/mps-laptop-check/check.py` passed quiet-capture provenance, refusal at
loads 8/90 before output creation, and five laptop CLI paths including small
widths, with execution replaced by controlled fixtures. Ruff check/format
and `git diff --check` passed. No model/game ran. Final 01:17 UTC load was
105.66; the 256-stream capture is ready for the operator's quiet window.
