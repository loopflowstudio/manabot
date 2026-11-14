# BF16 behavior logs: cause of the update-141 stop

Exploratory. BF16 seed 32401 of the [learning A/B](precision_ladder.md)
stopped at update 141 with `behavior_log_probability_mismatch`. This report
finds the cause, removes it, reruns that recipe on one L4, and compares the
rerun with the A/B's four runs over the updates they share.

**Answer.** The stop was a float32 rounding disagreement between two `exp`
implementations on an action nobody sampled; it had nothing to do with BF16
arithmetic. With the stored copy removed, the same recipe and seed trained 200
updates with no rejected step, at 114 s per update against float32's 196 s
(1.73×). Its loss, KL, clip-fraction and value curves sit inside the spread of
the float32 runs. The rerun never produced a probability small enough to
trigger the old failure, so the direct evidence for the fix is the CPU
regression on the retained incident row, and the run shows no regression.
Playing strength was not measured. Defaults and the `auto` resolver remain
float32 on every model and device.

## Cause and change

The old check compared two collection-time representations, not collection
against learner recomputation. It required device `exp(log p)` and host
`exp(log p)` to agree to 1e-5 relative with zero absolute tolerance. The
retained incident from `2be568a65518` has log p = -94.75248: CUDA stored
7.070952e-42 (5046 float32 subnormal spacings), while CPU recomputation
produced 7.072353e-42 (5047 spacings). One spacing is 1.98e-4 relative here.
The sampled action was a different, high-probability action.

Of the retained 2048 legal entries, 591 differed between device and host
(-2 spacings: 2; -1: 421; +1: 168). Normal-range differences always passed.
This is a precision-independent representability problem; the failing source
already cast logits to float32 before log-softmax.

Packed rollouts now retain only log probabilities. Sampling still exponentiates
on device; learners validate normalized finite legal logs and the selected
log's identity, with unchanged tolerances. No device-exp copy is stored or
compared. Host-derived probability minima and underflow counts remain metrics.
The fixed-weight collection/learner forward is unchanged by this deletion;
its delta is measured separately, never used as a substitute for the incident.

## What the five runs show

All five use `precision_ladder.arm`: w384l8, bucketed shapes without live
tokens, 256 streams, the same 300-update damping schedule, on L4s. MTG-324's
four ran revision `783ccdba`; the rerun (`b00f4500b78d`) ran this branch at
`34b380a554`. It was cancelled after update 200, 40 minutes before its window ended.

| Precision / seed / run | Updates | Rejected steps | Largest legal-logit gap | Smallest legal p | Updates with subnormal p | …with p < 1.4e-40 | …with p = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| float32 / 32401 / `07f84834e617` | 188 | 0 | 90.6 | 4.4e-40 | 9 | 0 | 0 |
| float32 / 32402 / `aa9518f6fc4b` | 170 | 0 | 72.6 | 2.9e-32 | 0 | 0 | 0 |
| bf16 / 32401 / `2be568a65518` (stopped at 141) | 140 | 0 | 124.1 | 0 | 40 | 33 | 12 |
| bf16 / 32402 / `6aa5ba4d5fac` | 290 | 0 | 103.5 | 1.4e-45 | 8 | 2 | 0 |
| bf16 / 32401 / `b00f4500b78d` (with the fix) | 200 | 0 | 68.6 | 1.6e-30 | 0 | 0 | 0 |

Below p ≈ 1.4e-40 one float32 spacing exceeds the old rule's 1e-5, so any
one-spacing `exp` disagreement there rejected the update. The stopped run
passed 33 such updates before one failed; BF16 seed 32402 passed two. The
"rejected steps" column counts optimizer steps refused inside admitted
updates; the update-141 stop was a whole-update rejection and is not in it.

### Does BF16 learn the same per update?

![Loss, KL, clip fraction and value explained variance by update](data/bf16_behavior_logs/learning.png)

The rerun (thick orange) tracks the float32 runs on all four measures through
update 188, the last update a float32 run reached. The two float32 seeds
differ from each other by about as much as BF16 differs from either: seed
32402's value fit rises some 20 updates later in both precisions. No curve
separates the precisions.

### How much faster is it?

![Collection and learning seconds per update](data/bf16_behavior_logs/seconds.png)

Median seconds per update over updates 1–140, which every run reached:

| Run | Collection | Learning | Both | Against float32, same seed |
|---|---:|---:|---:|---:|
| float32 / 32401 | 121.2 | 75.3 | 196.4 | |
| float32 / 32402 | 135.1 | 83.1 | 218.2 | |
| bf16 / 32401 (stopped) | 81.8 | 46.7 | 128.5 | 1.53× |
| bf16 / 32402 | 86.6 | 48.5 | 135.1 | 1.61× |
| bf16 / 32401 (with the fix) | 68.6 | 45.1 | 113.7 | 1.73× |

Collection speeds up as well as learning, because behavior sampling runs the
same forward. In a 7-hour window that is about 200 updates against 130.

## Apple GPU backends

On an M4 Max with torch 2.15.0 and MLX 0.32.4, one real w384l8 minibatch (64
learner rows of a captured CPU update) against CPU float32:

| Arm | Legal logits, max abs diff | Relative loss diff | Full-gradient cosine | Min per-tensor cosine |
|---|---:|---:|---:|---:|
| MPS float32 | 8.4e-9 | 1.8e-8 | 0.9999999987 | 0.99999997 |
| MPS bfloat16 | 6.0e-5 | 1.9e-4 | 0.99823 | 0.9724 |
| MLX float32 | 1.0e-8 | 4.5e-8 | 0.9999999986 | 0.9999996 |
| MLX bfloat16 | 1.3e-4 | 2.9e-3 | 0.99828 | 0.9853 |

The wrong MPS linear-layer gradients seen with torch 2.10–2.13 on an M2 do not
appear: float32 MPS matches CPU, and a reconstruction of that failure's
`F.linear` shape gave a relative error of 3.9e-6 where the failure was 2.25.
Both bfloat16 arms pass the existing bfloat16 parity bars (logits 1e-3, loss
1%, full-gradient cosine ≥ 0.99).

Three real updates per arm at w384l8, 16 streams × 64 transitions, completed
with zero rejected steps and finite parameters on MPS (explicit bfloat16
and float32) and MLX (explicit bfloat16 and float32). MLX has no collector,
so its arms collected with the torch CPU policy and stepped the MLX learner.
Single-minibatch losses were in the same range in both precisions
(MPS: −0.07, 0.00, 1.84 against 0.11, 0.08, 1.78).

## Cost

USD 3.66 for the rerun, by the launcher. A two-run 50-update pair was launched
and cancelled within minutes (USD 0.02); it produced no data.

## Limits

- The rerun did not exercise the failure. Its smallest legal probability was
  1.6e-30; the old rule could only fail below 1.4e-40. That it passed update
  141 shows no regression, and the incident-row regression test
  (`tests/training/test_numerical_health.py`) is what shows the fix.
- Why this BF16 run stayed at logit gaps under 69 when the two earlier BF16
  runs reached 104 and 124 is not explained. The revisions differ (`783ccdba`
  against `34b380a554`), and same-seed runs are not bit-identical, so this is
  one sample.
- No playing-strength evaluation was run. "Learns the same" here means the
  training curves of one fixed run per seed overlap, over at most 188 updates.
- The on-pod CUDA probe (device against host `exp`, and collection against
  learner chosen-log differences under both precisions) wrote `probe.json`,
  but the cancelled run delivered no output files, so those numbers are not
  available. The CPU smoke of the same probe is wiring evidence only.
- The Apple updates ran at 16 streams on a machine busy with other work. They
  show explicit BF16 executes without a numerical stop, and their timings and
  losses support nothing more. Whole-update timings at 256 streams are in the
  [MPS retest](mps_retest.md). M1 and M2 machines were not measured.
- Bucketed shapes with live tokens, the CUDA default layout, has no BF16
  training run here; the A/B recipe turns live tokens off.

## Reproduce

```sh
uv run python -m experiments.bf16_behavior_logs smoke
uv run etude launch experiments/bf16_behavior_logs.py --rent mtg324-l4-ab \
  --cohort mtg345-bf16-proof --max-usd 5 --revision COMMIT
uv run --extra artifacts --extra reports python -m experiments.bf16_behavior_logs_report
uv run --extra artifacts python -m manabot.data fetch experiments/data/bf16_behavior_logs
```

The data directory holds `updates.csv` (every update of the five runs),
`summary.json`, both charts, the CPU smoke receipts (`cpu-smoke/`) and the
Apple receipts and scripts (`mac/`).

<details>
<summary>Launch log</summary>

- 2026-11-13: launch `ebd68228bef4` was refused before recording a run ("job
  dollars do not cover the hours"): Etude adds the on-demand's 900-second
  startup allowance to the job window. The window went from 7.1 to 6.99 hours.
  Two quotes then returned "provider reports no available machines".
- 2026-11-13 17:07 UTC: run `b00f4500b78d` started on
  `mtg324-l4-ab-e0f0acfca5a5`. The job runs the probe as its own process, then
  the unmodified training worker.
- 2026-11-13 23:27 UTC: cancelled after update 200; machine terminated.

</details>

## Follow-up implementation (2026-11-13)

Restored float32 defaults and `auto` resolution on all devices and widths;
explicit BF16 and saved resolved precisions remain available. The probe-only
launch option completes independently of training so output delivery does not
require cancelling a long training job. Its job cap is USD 1.32, the remaining
authorized budget after the two costs above, including startup allowance.

Execution is blocked here: etude-sdk 0.3.2 uses `launcher-17.db`, which has no
providers, storages or on-demands; the `mtg324-l4-ab` on-demand, `runpod`
provider and `r2` storage exist only in the 0.2.x `launcher.db`. As
[the launcher transition](../docs/experiments.md) says, they must be added
again before launching. No launcher configuration was changed, and no
additional rental was launched. CUDA distributions and fixed-weight comparisons, including live-token
layouts, remain unavailable. The retained incident and 200-update run support
considering another learning A/B, but the requested CUDA checks must still
finish before deciding to rerun it. The full A/B was not launched.

Once that configuration is available, launch the committed probe revision with
`uv run etude launch experiments/bf16_behavior_logs.py --rent mtg324-l4-ab --cohort mtg345-cuda-probe --max-usd 1.32 --revision COMMIT --probe-only`.
Let it complete and fetch `probe.json`; do not cancel after probe execution.
