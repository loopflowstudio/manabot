# Policy numerics and update health

Ordinary `train_self_play` regimes retain masked normalized policy logs through
collection and learning. `TrainingRun` owns local measurements and terminal
incidents; W&B and the editable notebook/HTML report are optional projections.
This contract is identified by `policy_numerics=masked-log-softmax-v1` in the
run's source/runtime identities. Freeze a new source for any new experiment.

## Policy and sampling contract

For each legal offer, `log_softmax` computes normalized log probabilities after
masking padding to negative infinity. Legal logs must be finite. Collection
passes their exponentials to the existing `torch.multinomial` generator and
copies both the sampling weights and complete logs with the observation, action
and chosen-action log likelihood. Learners validate binary masks, nonempty
support, finite/nonnegative probabilities, zero illegal mass, normalized logs
and weights, agreement between their exponentials, legal sampled actions,
positive sampled mass and the exact saved chosen-log identity.

Float32 `exp(-104)` can round to zero (device/kernel dependent). A saved finite
legal log can still support a finite reverse KL when its float probability has
underflowed. This zero is accepted only when exponentiating that same saved log
also rounds to zero. The code introduces no epsilon, mixture, probability floor,
renormalization repair or skipped corrupt batch. Reference distributions still
require positive legal support. Nonfinite legal logits/logs and unrepresentable
log differences fail with named invariants.

The sampling kernel operates on finite precision weights: a rounded-zero action
cannot be sampled. The log-space policy is its mathematical target, not a claim
that infinitesimal actions are executable on float32 hardware. Multinomial's
normalization of approximately unit-sum weights also has ordinary rounding error.
Underflow counts expose this limitation; recovering lost *exploration* would
require an explicitly changed policy/sampler recipe, not just this repair.

Ataraxos retains the clipped sampled-action surrogate and both reverse KLs from
[the existing fidelity contract](ataraxos.md). Reverse KL now uses saved logs,
with padding removed from both products and log differences. Ratios exponentiate
in float64: sampled float32 actions bound the relevant log ratio to roughly 104,
which exceeds float32's exponential range but fits float64. The bad-move branch
remains unbounded; nonfinite objective/gradient state is rejected, never silently
clipped into validity. Advantages, behavior and references remain detached.

The PPO-style regime learner uses the same contract, replacing its historical
`1e-12` KL clamping. This restores the intended KL at small probabilities and
changes saturated-policy gradients relative to that approximation. Sampling's
masked fused normalization also changes floating-point rounding relative to
`Categorical`'s normalization. Expect normal-probability analytic parity, not
bit-identical historical trajectories. Checkpoint shapes/weights are unchanged;
in-memory rollout rows now require complete collection logs. Old runs and
artifacts are not rewritten or admitted as new evidence.

## Reading the dashboard

The primary report keeps numerical charts in one expandable section, with raw
scalar JSON available. W&B includes gap, underflow, gradient effect, sample
selection and optimizer-step panels. No network call occurs in the learner.
Missing measurements stay absent; a line between sampled W&B points is not a
measurement of intervening steps. None of these quantities measures playing
strength, and a lower aggregate RL objective is not lower classification log loss.

| Metric | Population and interpretation |
| --- | --- |
| `legal_logit_gap_max`, `legal_probability_min` | All collected learner rows, before filtering. A log-probability gap equals the within-row logit gap. Minimum includes rounded zeros. |
| `legal_underflow_count` | Legal action entries with zero sampling weight and admitted finite logs. Count, not a fraction or proof of harmless exploration loss. |
| `nonfinite_count` | Zero in admitted batches. A failure names its invariant; private incident summaries count nonfinite values in each tensor. Illegal log padding is intentionally infinite. |
| `actor_rows`, `critic_rows` | Unique selected rows, before epoch reuse. Existing actor/critic exposure counts retain optimizer reuse. Actor-only filtering can leave zero actor rows but train the critic. |
| `forced_rows`, `zero_advantage_rows`, `empty_actor_batch` | Collected forced decisions, exact-zero raw advantages and empty selection. Forced actions have zero policy gradient; critic/shared-representation updates can still move parameters. Zero advantage alone does not zero KL regularization. |
| `gradient_norm_before`, `gradient_norm_after` | Global L2 norm before/after clipping on the sampled optimizer step. |
| `gradient_missing_tensors`, `gradient_zero_tensors`, `gradient_nonzero_tensors`, `frozen_parameter_tensors` | Missing trainable `.grad`, present all-zero `.grad`, nonzero `.grad`, and explicitly frozen parameter tensors. Missing gradients can identify unused branches; these counts alone cannot identify a broken component. All missing gradients or a detached total objective reject the step. |
| `parameter_delta_l2`, `parameter_relative_delta` | Actual whole-model parameter movement on that same sampled step. Relative divides by pre-step norm; a zero pre-step norm uses absolute movement. Adam momentum can move parameters with zero current gradients. Nonzero gradients with zero movement can mean zero learning rate or rounded updates. |
| `optimizer_steps`, `skipped_steps` | Per-update completed optimizer calls and skipped minibatches. Empty filtering is an explicit skip. |
| `rejected_steps`, `rejected_updates` | Terminal failures, recorded separately from completed diagnostics. Distribution rejection before the optimizer is a rejected update, not a fabricated optimizer call. |
| `clip_fraction` | Fraction of ratios outside the clipping interval in the last optimized minibatch, not the fraction whose gradients were actually clipped. |
| entropy, collection/reference KL, loss | Last optimized minibatch values. Padding is excluded and KL uses the same logs as the objective. Filtered actor-only losses use selected rows; these descriptive entropy/KL summaries include all minibatch rows. |

Batch summaries run once per update. Detailed gradient counts and actual parameter
movement sample only the **first optimized minibatch at iteration 1 and every
25th iteration**. Empty updates have no sample; the clock still advances. Loss,
gradient-norm and post-step parameter finiteness checks run on every optimizer
step. These are batch/model reductions, not per-row GPU synchronization. The
additional complete log tensor has the same shape as stored probabilities.

The bounded overhead probe compares plain clipping/Adam with safety checks and
sampled effect measurements on 188,418 parameters, one thread, 100 steps, five
alternating-order repetitions. Run:

```bash
uv run python scripts/benchmark_numerical_health.py --device cpu
```

A 2026-10-07 CPU Torch 2.10.0 probe measured median 0.03690 s plain versus
0.06420 s guarded per 100 steps (1.74×, about 0.273 ms extra per optimizer step).
This intentionally exposes optimizer-only overhead; it is not full collection /
learning throughput, GPU timing or a hardware comparison. Re-measure CUDA costs
on its actual model; the cadence is not a claim of negligible overhead.

## Private terminal incidents

`StageRecord.numerical_failure` retains invariant, attempted iteration and update
health separately from completed-update diagnostics. Every terminal numerical
error attempts a unique `<stage>-numerical-incident/` directory:

- `incident.json`: source hash/commit, recipe/run/stage/iteration identities,
  completed versus attempted work, shapes/dtypes/devices, finite extrema,
  zero/nonfinite counts, state sizes, retained policy references and limitations.
- `tensors.pt`: up to 16 offending flattened row indexes, masks, logs, sampling
  weights, actions and observation slices; up to 32 MiB of tensor payload. Late
  offending rows take priority over early rows. Unaligned tensors retain a bounded
  prefix; metadata retains original shapes and explicit budget omissions. Batches
  up to 16 MiB retain every row and bootstrap observation within the 32 MiB cap;
  the manifest states whether the complete batch was captured. Optimizer failures
  also retain the triggering minibatch, target, selection and flat row indexes.
- `state.pt`: current learner, Adam, EMA, gradients, Torch/CUDA/Python/local NumPy
  and collector sampling RNGs (only model-owned CUDA devices), up to 512 MiB of tensor payload. Oversized state
  is explicitly unavailable. Small serialization metadata is additional to these
  tensor byte bounds; no arbitrary process locals or environment variables are saved.

These are **private debugging artifacts**, potentially containing invalid state.
They never overwrite or masquerade as admitted raw/EMA policies or CPU recovery
snapshots. This is failure-time state, not the missing pre-collection RNG/native
engine journal: it cannot promise replay of the entire iteration or CUDA process
recovery. Multiple earlier minibatches may have committed before a later failure.
The last valid exported checkpoint remains unchanged. Load only trusted,
hash-verified artifacts; `state.pt` contains Python RNG/optimizer structures.

The executor registers files in `StageRecord.rejected_artifacts` using ordinary
path/hash/byte references. Existing `retained_artifacts` and remote snapshots
carry them; ETU-126 owns upload/lifecycle supervision. Capture failure is appended
to the terminal record without replacing the original exception. Terminal store
or export failures leave a best-effort `terminal-error.json`; no disk-backed
system can guarantee persistence after a total disk failure.

Self-play collection and learning timers settle in `finally`, including a failing
operation's elapsed time. Prior setup, waiting and persistence residuals belong
to diagnostic overhead. They never all become active learning just because the
learner raised. Frozen ETU-103 cost receipts remain unchanged.

## Validation and restart admission

The deterministic suite checks each original invariant, finite logits with gaps
through 1,000 against float64 loss/gradient references, padding/permutation,
forced/empty support, malformed masks, sampled-log consistency, real native
collection-to-learner underflow, known-learning and zero-rate/zero-gradient /
detached/invalid controls, missing/frozen components, RNG neutrality, incident
state, failed clocks, optional telemetry and report projection. Existing bounded
regime tests exercise raw/EMA checkpoint export and ordinary reload.

```bash
uv run pytest tests/training/test_numerical_health.py -q
```

This host has no CUDA device. **CUDA validation remains required before restart**;
ETU-103/126 can run this bounded command on an already admitted capable host,
inside the existing total allocation, with the repaired exact source/native build:

```bash
MANABOT_NUMERICS_DEVICE=cuda uv run pytest tests/training/test_numerical_health.py -q
uv run python scripts/benchmark_numerical_health.py --device cuda
```

An explicit CUDA request fails without CUDA; it never counts a CPU fallback.
Retain stdout, source/native/runtime/device identities and the run's costs. No
long cohort or new rental is authorized by these commands. The ETU-103 failed
batch was not saved: deterministic underflow repair does not prove that
underflow caused that historical failure. New source/cohort and remaining-budget
decisions remain with ETU-103/126. Monitoring gains or numerical health do not
satisfy strength or human-challenger acceptance.
