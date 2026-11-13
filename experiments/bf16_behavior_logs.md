# BF16 behavior logs: numerical failure proof

This is an exploratory crash-fix proof, not a playing-strength comparison.
The question is whether log-only behavior storage removes the numerical gate
that stopped BF16 seed 32401 at update 141. One L4 job has a USD 5 ceiling;
precision defaults and the full learning A/B remain unchanged.

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

## Before

The frozen [learning A/B](precision_ladder.md) supplied these per-update metrics.

| Precision / seed / Etude run | Updates | Largest legal-logit gap | Updates with subnormal p | Updates with p < 1.4e-40 | Updates with p = 0 |
|---|---:|---:|---:|---:|---:|
| float32 / 32401 / `07f84834e617` | 188 | 90.6 | 9 | 0 | 0 |
| float32 / 32402 / `aa9518f6fc4b` | 170 | 72.6 | 0 | 0 | 0 |
| bf16 / 32401 / `2be568a65518` | 140 | 124.1 | 40 | 33 | 12 |
| bf16 / 32402 / `6aa5ba4d5fac` | 290 | 103.5 | 8 | 2 | 0 |

Before the fix, no retained paired collection/learner forward distribution
was measured. That quantity did not trigger the rejection. The before/after
comparison for the actual failure is the retained incident and the CPU
regression that now admits those normalized logs, including underflowed
unsampled actions. No change to head arithmetic is claimed.

## Probe and run

`bf16_behavior_logs.py` enumerates every float32 input in [-104, -87.3],
reports device-minus-host exp spacing gaps and old-rule rejection counts,
then compares same-weight collection and differentiable learner chosen logs
for float32/BF16, bucketed/bucketed-plus-live-token attention and two bucket
sizes. A native trajectory provides viewer-safe observations; collection
uses its cached catalog, while learner chunks build their own gradient graph.
No optimizer step occurs during this probe.

The training phase uses `precision_ladder.arm("bf16")`, seed 32401, w384l8,
bucketed shapes without live tokens and the original 300-update schedule.
Only the watchdog is shortened: 7.1 rental hours less 900 seconds for setup
and probe and 300 seconds for finalization. The ordinary wall-bound stop
retains raw/EMA/optimizer exports. At the failed run's 126 seconds/update,
this covers about 193 updates; faster training may pass 200. The window is
the stopping bound, not a new damping horizon.

```sh
uv run python -m experiments.bf16_behavior_logs smoke
uv run --with 'etude-sdk[s3,eval]==0.2.15' etude launch \
  experiments/bf16_behavior_logs.py --rent mtg324-l4-ab \
  --max-usd 5 --revision COMMIT
```

Dated line, 2026-11-13: CPU smoke completed all eight forward combinations
and the strided exp probe (2138 inputs). This verifies wiring only, with a
w16l1 model and eight native rows; it is not CUDA evidence. CUDA launch and
run results are pending.

## Decision

The recorded failure has a precision-independent cause and a regression
covering the incident. The strength and learning-trajectory questions remain
open. A successful CUDA proof would support rerunning the learning A/B;
this experiment does not authorize that roughly USD 30 rerun or enable BF16
by default. The rerun needs a person's go-ahead naming the amount.
