# MPS retest: partial results and remaining design work

## Outcome

`experiments/mps_retest.md` contains the measured answer, recommendation,
charts, limits and run history. Both MPS half dtypes executed. Production
backend defaults and off-CUDA refusals are unchanged. This is not a completed
matrix: missing cells remain visible in charts and tables.

Measurements used source `fa23fc364`; original 256-stream captures used
`99210bf8d`. Subsequent edits label plotted cosine as global and consolidate
parity subprocess handling through `CallWorker`. They change no model math
or retained evidence; measurements remain bound to their original source.

Completed three-replay groups, in CPU / MPS f32 / bf16 / f16 / MLX order:

- M4 Max w384l8/256, `5bd37fb3abaa`: median totals 703.20 / 521.22 /
  478.42 / 442.03 / 402.22 s. No retry; all sampled choices match CPU.
- M4 Max w64l2/256, `7fb121dc08a6`: 110.18 / 136.38 / 135.82 /
  136.08 / 95.94 s. Repetitions 0 and 2 each retried once, selecting the
  original lower-mean-load attempt; both retries remain in the ledger.
- M2 w64l2/256, `133dacc3287a`: 107.75 / 141.98 / 144.15 /
  142.93 / 84.00 s. No retry; all sampled choices match CPU.

The last laptop retry ended 2026-11-13T06:19:45Z, continuing the already
running series across the 06:17 window per the explicit no-stop direction.
M2 timing ended at 06:16:29Z. Measurement workers exited on both hosts.
No further measurements were queued. Rental release remains operator-owned;
allocation was 2026-11-13T02:56:06Z, 24-hour minimum about USD21, cap USD25.
Release timestamp is still pending.

## Counterexamples and remaining work

1. **Review resident memory before further M2 scaling.** w384 M2
   `877fa3cf766d` was cancelled after about 105 minutes without a complete
   replay, with used swap 10,518.25 MiB and sustained paging. No OOM was
   inferred. Even the completed w64 setup had 7.90 GiB of swap-outs across
   setup, replay and cleanup while used swap ended near 257 MiB. Raw counters
   are in `m2/host-observations.jsonl`. Pairwise interleaving changes the
   steered all-arm comparison and single CPU baseline; it was not silently
   substituted. Further timing needs a memory-bounded design, not more runs
   of the same pressure-prone setup.
2. **Review the larger-stream capture budget and execution window.** Laptop
   512 `dcf93e86b142` and 1024 `88dd067a2d9b` failed the configured 20 GiB
   process-tree memory guard before learning. Exact partial captures and
   launcher errors are retained. This is not proof of a replay-arm OOM or
   the laptop's physical capacity. Only 256-stream inputs are replayable.
3. **The remaining matrix is unmeasured.** No warmed per-call matrix ran;
   w96l4, w160l8, w512l8, w768l8 and w1024l8, and every 512/1024-stream
   replay group, remain open. The M2 has no complete w384 replay. The laptop
   window is over. Do not substitute one repetition for a median of three or
   treat this implementation step as the completed experiment.
4. **Interpret the half-precision differences before adoption.** Global
   cosine is not a tensor-wise correctness verdict. M2 w384 bf16 global
   cosine was 0.996963478; M4 w384 bf16/f16 had poorer individual tensors.
   The score-bias case is analytically zero and needs norm-aware interpretation.
   The report recommends retaining Torch in production and MLX as experimental.
5. Gate owns the remaining affected suites and acceptance checks. Publication,
   landing and Task movement belong to later Graph steps. Link the named
   `full_context.png` first in the PR body; it is explicitly a partial M2
   matrix, with a separate M4 chart. No Wave is bound.

## Evidence and delivery

The per-host `bench-back-to-back` ledgers retain every attempt and selection.
Root `bench.jsonl` / `updates.jsonl` are byte-concatenated report inputs, not
new measurements. `summary.json`, six PNGs and two comparison tables derive
from them. The charts were visually inspected with real receipts; global
cosine labels and host provenance are explicit, and gaps are not zero-filled.

Published via `manabot.data`: current ledgers, laptop references, both charts
and missing per-call charts, tables, summary, host observations, launcher
receipts, and `laptop/capture-failures-fa23fc364.tar.gz`. The original laptop
captures/failed native preflight are in the already-published
`laptop/captures-99210bf8d.tar.gz`; old cancelled `laptop/bench` evidence stays
separate. The final M2 capture/ledger/reference/receipt archive was transferred as
`m2-evidence-fa23fc364.tar.gz`; its local SHA-256 matched the remote
`264ef59f82285682e3bb656b3f670d6a23f01eea12c25a0bd2a4c6436ec831b9`.
Publication completed; all 38 retest manifest entries are retained. No artifact
transfer or measurement process remains running.

## Delete — do not maintain

Removed the duplicate pilot capture, serial timed-arm path, reference writer's
disk reread, repeated host/capture probes and worker-side parity admission.
The coordinator owns admission before allocation and attaches parity metrics.
Removed the separate parity subprocess launcher and its ledger-length inference:
`CallWorker` now owns replies and cleanup for parity as well as timing. Parity
still completes one arm at a time under one deadline; final receipts follow
runner cleanup. The backend README links to the report instead of repeating
its measurement table. Superseded scratch directions remain at
`adeb2b164:scratch/questions.md`.
Keep isolated parity execution, CPU arrays, original captures and failures;
no alternate capture or rental implementation. No remaining deletion target.

## Validation

Earlier controlled checks and their commands are retained in
`fa23fc364:scratch/mps-retest.md` (older history at `348c8f26b` and `9a83ac5`).
At the measurement freeze, fixtures passed unconditional high-load admission,
directional >2× retry on both hosts, whole-attempt selection, and acceptance
of a 10,000-second three-replay span without an hourly gate. Ruff and diff
checks passed; no production model math changed in that method update.

Report checks before compression: `uv run --no-sync python experiments/mps_retest.py report
experiments/data/mps_retest --bench experiments/data/mps_retest` passed;
independent assertions verified 15 selected medians, both half dtypes, no
pending retry, exact phase sums, explicit gaps, six PNGs and two UTC-labelled
tables. Ruff format/check and visual inspection passed.
`uv run --no-sync pytest tests/infra/test_experiment_data_untracked.py -q`
passed (1 test). The final archive hash/size matched its manifest entry;
`git diff --check` passed. Scratch is within both context limits.

Compression checks: `uv run --no-sync python .runs/mps-parity-worker-check.py`
passed controlled reply, failure, timeout, cancellation, deadline and CLI checks;
all fixture children were reaped and every generator closed its runner before
the final receipt. `uv run --no-sync pytest
tests/eval/test_experiment_files.py -q` — 4 passed (7.29 s). No measurements
were rerun; gate owns the remaining affected suites. Python compilation,
Ruff check/format and `git diff --check` passed.
