# MPS retest implementation

## Outcome and constraints

Measure the bucketed recipe across CPU8 float32, MPS float32, both working MPS
half dtypes and MLX float32. `experiments/mps_retest.md` owns the full matrix
and deliverables; `questions.md` preserves the exact steers. Torch is the
incumbent. Recommend from measurements, without an adoption rule or defaults
change. No performance or real-MPS correctness conclusion exists yet.

Main evidence requires a fresh M2 rental under the USD25 total cap, with
launcher identity, cost and release. Each width/stream group needs three
replays per arm on one host within one hour. Only demonstrated M2 overflow
belongs on the quiet M4 Max, with capture and replay phases labelled separately.

## Delete — do not maintain

No replacement subsystem is planned. The shared capture owner replaces the
pilot's inline capture implementation (already removed). Preserve capture
format, production learner hooks and the pilot's callers. In reporting, replace
success-only plots in place; keep one report path and retain raw failed attempts.
The duplicated per-call/update coverage rules and chart-owned summary mutations
are removed. `experiments/mps_retest.py::_host_report` now selects evidence once
into `HostReport`; `_missing_arms` owns coverage, and the chart/table functions
only render it. No further deletion target remains in this pass.

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

## Remaining execution

1. Supply a verified capped AWS allocation/release path. The access notes and
   their release script were read; AWS confirms the old host was released.
   Etude SSH cost/expiry cannot release an externally allocated Dedicated Host.
   Read-only AWS automation checks found no existing release controller in
   us-east-1. The exact investigation and sources are in `questions.md`.
   No allocation, new run, paid operation or external configuration was made.
2. Freeze source, capture the three real updates, attempt both MPS half dtypes,
   run correctness and the per-call matrix, then interleaved triple replays.
   Fix implementation failures rather than labelling them unsupported. Check
   same-hour feasibility from measured durations before scheduling the expanded
   groups; keep the rule unless a person changes it.
3. Select overflow cells from actual M2 OOM evidence, not just width guards.
   Before laptop timing report the block and known memory need, check load <8,
   and leave it unmeasured for the operator if busy. Do not wait indefinitely.
4. Publish captures, receipts, tables and actual charts with `manabot.data`.
   Embed `full_context.png` in the report and link it first in the eventual PR.
   Write the measured answer, limits and recommendation; update backend README.

Gate owns affected Python checks (`test_inference.py`, `test_mlx_backend.py`,
experiment discovery, package boundaries and preparation coverage), real
capture/MPS acceptance and final report review. No Wave is bound. Publication
and Task movement belong to later Graph steps. Rental execution is blocked;
implementation checks do not complete this experiment.
