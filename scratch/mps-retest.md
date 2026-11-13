# MTG-335 MPS retest — plan

## Done (backed by HEAD)

- Harness ported to current backends: `manabot/backends/torch/capture.py`,
  `experiments/mps_retest.py`, `experiments/mlx_pilot.py` (7924086ad, then
  shared gradient agreement in `manabot/backends/parity.py`, 1508f302d).
  Modules import at HEAD.
- Current-source measurement: M4 Max w384l8/256, seven arms (CPU f32, MPS
  f32/bf16/f16, MLX f32/bf16/f16), three CPU-paired replays each at `140eec28d`.
  Medians: CPU 674.76, MPS f32 277.32, MLX f32 263.20, MPS bf16 261.04,
  MLX bf16 207.58, MPS f16 266.53, MLX f16 203.41 s. All 48,064 sampled
  choices matched CPU.
- `_update_chart` renders only measured host/width/streams panels with a
  learner decisions/s strip; report has the full host×model×streams×arm
  coverage matrix (Measured / Failed with cause / Not reached).
- Report `experiments/mps_retest.md` and the decision line in
  `manabot/backends/README.md`: keep PyTorch in production, MLX experimental.
  Defaults and off-CUDA refusals unchanged. `fa23fc364` results are kept as
  historical sections, not pooled.
- Evidence published: 78 `mps_retest` manifest entries, including
  `full_context.png` and 31 `laptop/paired-update` files.

## Departure from the brief

The brief's pre-registered rule (MPS adopted only if ≥1.2× faster than MLX
at w384l8/256+ with cosine >0.999) was replaced by the repository's current
experiment guidance (no pre-registered verdicts, PR #505); the report is
exploratory against the incumbent. Under either reading MPS is not adopted:
best MPS (bf16, 261 s) is not faster than MLX f32 (263 s) and MLX half is
20–24% faster than MPS half.

## Not measured (stays explicit in the report)

- No 512/1024-stream replay: the 512 capture stopped at the 20 GiB
  process-tree guard before learning (peak RSS 21,598,879,744 B, `7924086ad`).
  So the batch-size question is unanswered.
- No warmed per-call matrix; no w96/w160/w512/w768/w1024 timing; w768/64-row
  correctness only; M4 w64/256 has one repetition, not a median.
- No M2 whole-update result at the current source: w64 and w160 stopped on the
  paging guard during isolated CPU parity.
- Every accelerator arm exceeds the 1e-6 one-step parameter check; MLX bf16
  failed the 1% relative-loss gate (1.07%) once on two real rows and passed
  after sync. Training equivalence is unmeasured.

## Open

- M2 host release timestamp (operator-owned). Operator: instance terminated,
  host releases 2026-11-14.
- PR #513 checks were pending at realign; gate/publish own them.

## Concept review 2026-11-13 — implemented in `76d211074d`

No number, chart, receipt or measured claim changed; no new measurement.

- Report: the Answer gives the w384l8/256 result, the w64l2 width evidence
  (`fa23fc364`, not re-measured) and the unmeasured larger batches, then the
  recommendation. Title drops "partial results"; Cost is a visible section;
  "Arms and inputs" lists what ran; retries, load selection, guard stops and
  the host release sit in the collapsed appendix.
- The coverage matrix stays in full and is the one owner of unreached cells:
  "A separate matrix table in the report lists every (host, width, streams,
  arm) as measured, failed (with why: paging, memory) or not reached" — Jack
  Heart, Task comment 2026-11-13T10:47.
- `manabot/backends/README.md` carries the decision line and links;
  `mlx.md#compute-precision` owns the half-precision gate description.
- `parity.Tolerances.cosine_scope` replaces the dtype-name branch in
  `Report.checks`; thresholds, labels and acceptance are unchanged.

Considered and left alone:
- `Learner(precision=...)` also sets the policy's precision. It mirrors the
  torch `Learner` signature and the steer asks for the option "on the MLX
  learner and policy"; documented in `mlx.md`.
- Torch regimes spell `bf16`; MLX spells `bfloat16`/`float16`, and both types
  are named `Precision`. MLX's values are dtype names consistent with
  `float32`; the regime spelling is serialized and frozen. Map at the regime
  boundary if MLX ever trains from a regime. (My reading, not a stated
  requirement.)
- bfloat16 is "supported" with gates calibrated on the retained checks and
  one real-row failure (1.07% against 1%). The steer asks for exactly this
  ("the parity check gating half by its own measured tolerance"); the guide
  and report both show the counterexample.

No open product choice. Nothing is left to delete: no files, fixtures, stored
evidence or public precision options were removed.

Checks: `uv run --extra dev pytest tests/training/test_update_backends.py -k
parity -q` (4 passed); Ruff check/format clean. Against `afd3bb2ede`, every
result table and the coverage matrix are byte-identical; the only table
change drops the planned-axis grid (design, not evidence; the matrix owns the
unreached cells) and renames "MLX half (new runs)" to "MLX half". Gate owns the backend suites and
publication checks.
