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

## Completed CUDA probe

Run `e9ae5c2bfca4` completed on one NVIDIA L4 with torch 2.15.0+cu132 at
revision `10e77ff53934` on 2026-11-14 UTC. Its
[receipt](data/bf16_behavior_logs/cuda-probe/probe.json) is in the artifact
store.

### Device versus host exp

The probe enumerated all 2,188,903 float32 log values from −104 through −87.3.
CUDA minus host output spacings were −1 for 46,419 inputs, zero for
2,127,031, and +1 for 15,453. CUDA returned zero for 3,660 inputs. The old
`atol=0, rtol=1e-5` comparison rejected 1,149 inputs. These are float32 `exp`
results, independent of whether the logits came from float32 or BF16 forwards;
they reproduce the failure mechanism directly on CUDA.

Before the fix, those 1,149 inputs would fail the redundant device/host check
in either precision. After the fix, that comparison no longer exists: the
packed rollout retains only logs, with normalized-legal-log and selected-log
identity checks unchanged. This is removal of an invalid comparison, not a
measured improvement in the forward's numerical agreement.

### Fixed-weight collection versus learner forwards

An initialized w384l8 policy saw 32 real native observations with seed 32401.
Collection used all 32 rows with inference catalog caching; learner forwards
used four eight-row chunks with a fresh differentiable catalog graph.
Both kept the same weights, observations and sampled actions. The learner
used either the collection's bucket sizes or twice their capacity, capped at
the observation's limits. Values below are absolute differences in the sampled
action's log probability; full bucket dimensions are in the receipt.

| Precision | Layout | Learner buckets | Median | p95 | p99 | Maximum |
|---|---|---:|---:|---:|---:|---:|
| float32 | bucketed | 1× | 0 | 8.64e-08 | 1.19e-07 | 1.19e-07 |
| float32 | bucketed | 2× | 0 | 5.96e-08 | 5.96e-08 | 5.96e-08 |
| bf16 | bucketed | 1× | 7.69e-06 | 2.29e-05 | 2.3e-05 | 2.31e-05 |
| bf16 | bucketed | 2× | 7.6e-06 | 2.19e-05 | 3.13e-05 | 3.5e-05 |
| float32 | bucketed + live | 1× | 0 | 0 | 8.23e-08 | 1.19e-07 |
| float32 | bucketed + live | 2× | 0 | 5.96e-08 | 1.01e-07 | 1.19e-07 |
| bf16 | bucketed + live | 1× | 7.15e-06 | 2.15e-05 | 2.29e-05 | 2.29e-05 |
| bf16 | bucketed + live | 2× | 9e-06 | 1.77e-05 | 2.47e-05 | 2.67e-05 |

All differences were finite. BF16's maximum, 3.50e-5, corresponds to a
probability ratio about 1.000035 at fixed weights. These deltas measure
collection/learner forward variation, a different quantity from the removed
host/device `exp` comparison. The fix changes no forward, so this distribution
applies before and after the deletion; the probe does not replay an older
revision. It changes no health tolerance and does not claim every BF16
recomputation satisfies a 1e-5 absolute bound.

Together with the incident regression and 200-update BF16 run, these checks
support rerunning the controlled float32/BF16 learning A/B with explicit
precisions. They do not establish equal playing strength or live-token
training stability: this probe uses initialized weights and only 32 rows.
The approximately USD 30 A/B still needs a person's approval naming that
amount; it was not launched.

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
and cancelled within minutes (USD 0.02); it produced no data. The probe-only
run was approximately USD 0.02.

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
- The cancelled training run delivered no probe output. The independent
  completed CUDA probe above supplies those measurements at initialized
  weights, not at the retained update-141 checkpoint.
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
# The CUDA probe alone; let it complete so the launcher delivers probe.json.
uv run etude launch experiments/bf16_behavior_logs.py --rent mtg324-l4-ab \
  --cohort mtg345-cuda-probe --max-usd 1.32 --revision 10e77ff53934 --probe-only
```

The data directory holds `updates.csv` (every update of the five runs),
`summary.json`, both charts, the CPU smoke receipts (`cpu-smoke/`), the Apple
receipts and scripts (`mac/`) and the CUDA probe receipt
(`cuda-probe/probe.json`).

<details>
<summary>Launch log</summary>

- 2026-11-13: launch `ebd68228bef4` was refused before recording a run ("job
  dollars do not cover the hours"): Etude adds the on-demand's 900-second
  startup allowance to the job window. The window went from 7.1 to 6.99 hours.
  Two quotes then returned "provider reports no available machines".
- 2026-11-13 17:07 UTC: run `b00f4500b78d` started on
  `mtg324-l4-ab-e0f0acfca5a5`. The job runs the probe as its own process, then
  the unmodified training worker.
- 2026-11-13 23:27 UTC: cancelled after update 200; machine terminated. The
  cancellation delivered no output files, so the probe became its own job
  (`--probe-only`), capped at the USD 1.32 left of the authorized budget,
  startup allowance included.
- 2026-11-14 00:13 UTC: the launcher's `runpod` provider, `r2` storage and L4
  on-demand were added again through its CLI. Probe run `e9ae5c2bfca4` was
  quoted USD 0.69, admitted at revision `10e77ff53934` and completed at 00:15
  UTC. The full A/B was not launched.
- This revision resolves `auto` to float32 on every device and width, with
  explicit BF16 and saved precisions unchanged. That conflicts with the BF16
  default for wide GPU models merged in PR #520 and is unresolved.

</details>
