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

## Concept review 2026-11-13 — next implement direction

No number, chart, receipt or measured claim changes. No new measurement.
A reader takes three things from this Task: the report's answer, the README
recommendation, and the MLX precision option. The option reads cleanly
(policy owns precision; loaders, `MLXBackend` and `Learner` pass it through;
float32 default). The report and README are where the reader works hardest
for no benefit.

1. **Report answers the asked questions in its Answer.** The brief asks two
   things (does MPS close the gap; does width change the answer) and a steer
   adds a third (do larger batches). The Answer covers the first and spends
   the rest on process. Rewrite it as three plain answers:
   - w384l8/256: the existing table and two sentences.
   - Width: the w64l2 result is the only width evidence and is currently
     filed under "Historical": CPU beat every MPS precision on both hosts,
     MLX float32 was fastest (source `fa23fc364`, five arms, not re-measured
     at `140eec28d`). Say that in the Answer; keep the section, titled by
     its finding and its source.
   - Larger batches: unmeasured, one sentence why (512 capture stopped at
     the 20 GiB guard before learning).
   Then the recommendation. Retries, load selection, the w64 rerun stop, M2
   guard stops and release status move to the collapsed appendix.
2. **Title and voice.** Drop "— MTG-335 partial results"; incompleteness is
   the coverage matrix and Limits. Remove "wrap-up direction", "the operator
   says", "later direction authorized": state what ran and what did not.
3. **Cost is a visible section** before Limits (house style, and the brief):
   about USD 21, the 24-hour mac2-m2 minimum, under the USD 25 cap, operator
   managed, allocation 2026-11-13T02:56:06Z. The pending release timestamp
   stays in the appendix.
4. **"Planned matrix and inputs" becomes "Arms and inputs"**: what ran. The
   planned-but-unreached cells have one owner, the coverage matrix (which
   stays in full: "A separate matrix table in the report lists every (host,
   width, streams, arm) as measured, failed (with why: paging, memory) or not
   reached" — Jack Heart, Task comment 2026-11-13T10:47).
5. **`manabot/backends/README.md` carries a decision line, not the report.**
   "Which backend on Apple Silicon": the recommendation, one reason, the
   link. The half-precision gate description has one owner,
   `mlx.md#compute-precision`; the README parity section links to it instead
   of restating loss scaling and gates.
6. **`parity.Tolerances` states what gates each dtype.** `Report.checks`
   branches on `precision == "float32"` to pick per-tensor versus
   full-gradient cosine beside a table that already says `step=None`. Make
   the cosine scope a `Tolerances` field so the table is the whole rule.

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

No open product choice.
