# CUDA training in TF32 and BF16 against float32 on an L4 — MTG-317

## Question and prediction (committed before the scored run)

Can a CUDA stage train under `Execution.precision="tf32"` or `"bf16"` and learn
the same as float32, and how much faster is each? MTG-301 found every learner
and collection kernel running in fp32 with tensor cores idle. **Prediction:**
TF32 is accepted under the rule below, with its learner step at least 1.2×
faster than float32 at w384l8 with padded shapes and at least 1.1× faster at
both sizes with bucketed shapes; BF16 is faster still (at least 1.5× on the
w384l8 learner step) but is not accepted, because its clip-fraction or
policy-loss trajectory leaves the float32 reference spread. **Refuted** if TF32
is rejected or unresolved, if TF32 is under 1.1× on the bucketed learner step,
or if BF16 is accepted. **Cost cap:** USD 3 for the launch (`--max-usd 3`),
inside the Task's USD 5 budget.

### The acceptance rule

Measured on one L4 (`experiments/cuda_precision.py`). One real float32 update of
the frozen MTG-267 w160l8 regime keeps its initial weights, 48 collection
observations and the complete loss inputs of 20 learner minibatches. A *group*
is one size (w160l8 with those weights; w384l8 seeded random) with
`bucketed_shapes` off or on. Each group runs five arms from the same weights:
`float32`, `tf32`, `bf16`, `float32-order` (another minibatch order) and
`float32-repeat`. Each arm trains ten epochs over the 20 minibatches (200
optimizer steps, about one update's worth) with fresh Adam at the captured
learning rate.

- **Deviation.** For policy loss, value loss and clip fraction, the largest
  per-epoch mean difference from the group's float32 arm.
- **Reference.** The larger of the `float32-order` and `float32-repeat`
  deviations: what an arbitrary minibatch order or CUDA's own nondeterminism
  already does to the same learning.
- **Fixed weights.** At the initial weights, on 8 minibatches: the relative
  difference of total, policy and value loss, and the cosine between full
  gradients.

A candidate precision is **accepted** when, in all four groups, every metric's
deviation is at most 1.0× its reference, every gradient cosine is ≥ 0.99, every
loss differs by ≤ 1%, and no step is skipped, rejected or non-finite. It is
**rejected** when any group has a deviation above 2.0× its reference, a gradient
cosine below 0.95, a loss difference above 5%, any skipped, rejected or
non-finite step, or an error (the `index_copy_` dtype class). Otherwise it is
**unresolved** and is not adopted.

**Default.** The accepted precision becomes the CUDA recommendation only if its
median learner step and its median large collection call (≥ 64 rows) are both
≥ 1.10× faster than float32 at both sizes with bucketed shapes on, which the
recipes now use; if both candidates qualify, the faster one. Otherwise float32
remains. `Execution.precision` is already a typed, always-serialized field, so a
recommendation is set in the CUDA recipes' execution settings. It does not
change the `Execution` default, which would change the identity of every regime
that does not name a precision.

## Answer

**Neither reduced precision is adopted; CUDA training stays float32.** Under
the rule above, TF32 is *unresolved* and BF16 is *rejected*. Both agree with
float32 closely at fixed weights, and neither leaves float32's own spread by
more than about 2× during learning. But the 200-step replay is chaotic:
float32 run twice, with only CUDA's atomic-order nondeterminism (1e-7 at fixed
weights) between the runs, drifts as far apart as a changed minibatch order
does. So this instrument cannot tell "same learning" from "different
learning" at the resolution the rule asks for. Speed decides little either
way. TF32 is 1.1× at w160l8 and 1.3–1.5× at w384l8. BF16 is *slower* than
float32 at w160l8 and 2–2.7× faster at w384l8. The prediction is refuted for
TF32 (predicted accepted). BF16's "not accepted" held, but only through one
policy-loss reading at 3.1× a reference of 2.8e-4, a near-zero noise floor. In
absolute terms BF16's numbers do not show broken learning: gradient cosine
≥ 0.992, total loss within 0.6%, and no bad steps. The rule was too strict for
this instrument (see Decision).

## Results

### How much faster is each precision?

![Speed of TF32 and BF16 relative to float32, per group](data/cuda_precision/speed.svg)

Medians in milliseconds. The learner step excludes the first epoch. Large
collection calls are those with ≥ 64 rows (36 of the 45 kept observations), two
timed passes. Peak memory is in GB.

| Group | Arm | Learner step | Large call | Learner peak | Collection peak |
|---|---|---:|---:|---:|---:|
| w160l8 padded | float32 | 382 | 464 | 5.3 | 3.8 |
| | tf32 | 340 | 410 | 5.3 | 3.8 |
| | bf16 | 498 | 719 | 10.7 | 4.4 |
| w160l8 bucketed | float32 | 176 | 193 | 2.2 | 0.8 |
| | tf32 | 159 | 188 | 2.2 | 0.8 |
| | bf16 | 183 | 226 | 2.2 | 1.0 |
| w384l8 padded | float32 | 869 | 1,018 | 12.8 | 4.4 |
| | tf32 | 655 | 677 | 12.8 | 4.4 |
| | bf16 | 397 | 376 | 7.5 | 2.3 |
| w384l8 bucketed | float32 | 381 | 396 | 5.4 | 1.7 |
| | tf32 | 283 | 272 | 5.4 | 1.7 |
| | bf16 | 196 | 194 | 3.4 | 1.1 |

The float32 order and repeat arms read within 2% of float32 in every group, so
host drift is small. Small calls (≤ 8 rows) are host-bound at 19–31 ms in
every arm. BF16 adds 2–7 ms to them, the cost of the autocast casts. At w160l8,
BF16's cast and layout overhead exceeds its tensor-core gain, and with padded
shapes it doubles learner peak memory. At w384l8 the gain dominates and memory
falls. The default bar (≥ 1.10× on both the learner step and large calls, at
both sizes, bucketed) fails for both: TF32's large calls at w160l8 bucketed
are 1.03×, and BF16 is below 1× there.

### Does each precision learn the same as float32?

![Deviation from float32 relative to float32's own spread](data/cuda_precision/equivalence.svg)

**Fixed weights** (8 retained minibatches, 45 collection observations per
group). Values are the worst over the four groups:

| | float32 repeat | TF32 | BF16 |
|---|---:|---:|---:|
| Loss, relative difference | 6e-7 | 8e-4 | 6.4e-3 |
| Gradient cosine (min) | 1.0000000 | 0.99983 | 0.9925 |
| Gradient norm, relative | 3e-6 | 1.9e-3 | 7.4e-3 |
| Collection log-probability, abs | 2.4e-7 | 7.2e-6 | 7.1e-5 |
| Collection value, abs | 3.1e-7 | 6.1e-4 | 5.7e-3 |

Both candidates pass the fixed-weight bounds (≤ 1% loss, cosine ≥ 0.99). BF16's
gradient cosine is lowest at w160l8 (0.9925) and 0.9991 at w384l8. No arm
skipped, rejected or produced a non-finite step in 200 optimizer steps.

**Learning** (ten epochs over 20 minibatches). The figure plots each metric's
largest per-epoch mean deviation from float32 as a multiple of float32's own
spread. TF32 ranges 0.35–1.87× (12 readings, 6 above 1). BF16 ranges 0.24–3.10×
(12 readings, 6 above 1, one above 2: policy loss at w160l8 bucketed). Both
scatter around 1 in the same way.

The ratios divide by a reference that is one draw and, for policy loss, nearly
zero. Absolute deviations: the largest per-epoch mean difference from float32
over ten epochs. Policy loss runs from −0.003 to −0.008, value loss from 0.15–0.42
to about 0.10, and clip fraction from 0 to 0.07–0.15.

| Group | Metric | float32 repeat | float32 order | TF32 | BF16 |
|---|---|---:|---:|---:|---:|
| w160l8 padded | policy loss | 3.0e-4 | 7.8e-4 | 4.2e-4 | 3.0e-4 |
| | value loss | 0.020 | 0.017 | 0.037 | 0.016 |
| | clip fraction | 0.045 | 0.059 | 0.032 | 0.067 |
| w160l8 bucketed | policy loss | 2.8e-4 | 1.7e-4 | 4.8e-4 | 8.7e-4 |
| | value loss | 0.016 | 0.026 | 0.019 | 0.025 |
| | clip fraction | 0.056 | 0.059 | 0.064 | 0.064 |
| w384l8 padded | policy loss | 2.7e-4 | 6.4e-4 | 2.4e-4 | 1.5e-4 |
| | value loss | 0.016 | 0.072 | 0.026 | 0.031 |
| | clip fraction | 0.083 | 0.078 | 0.089 | 0.047 |
| w384l8 bucketed | policy loss | 2.8e-4 | 1.3e-4 | 3.6e-4 | 5.0e-4 |
| | value loss | 0.028 | 0.070 | 0.031 | 0.039 |
| | clip fraction | 0.080 | 0.081 | 0.106 | 0.110 |

BF16's rejecting reading is 8.7e-4 against references of 2.8e-4 (repeat) and
1.7e-4 (order). Elsewhere the float32 order arm alone reaches 7.8e-4.

The reference itself is large. By epoch 10, float32 against its own repeat
differs in clip fraction by 0.04–0.08, against levels of 0.07–0.15. In the
first epoch, before divergence compounds, the ordering is visible. The largest
first-epoch value-loss difference from float32 is 5e-5–1.8e-3 for the repeat,
2.8e-3–7.6e-3 for TF32 and 3.0e-3–1.2e-2 for BF16. In every group, TF32 and BF16
perturb learning more than nondeterminism does. That is not evidence that they
learn worse.

## Decision

- **CUDA default: float32**, unchanged. No precision met the rule. Even an
  accepted TF32 would miss the 1.10× default bar at w160l8 bucketed (1.03× on
  large collection calls).
- **The rule was too strict for this instrument.** It compared each candidate
  with a single draw of float32's spread, and took the maximum over ten epochs
  of a chaotic replay. Two float32 runs separated only by atomic ordering
  already differ by about as much as any perturbation does after a few epochs.
  A candidate's ratio is therefore close to a ratio of two noise draws, which
  lands above 1 about half the time and occasionally above 2. The pre-registered
  verdicts stand. The verdict is a property of the test, not evidence that
  either precision learns differently.
- **TF32: unresolved.** Its numerics sit closest to float32 (loss within 0.08%,
  gradient cosine ≥ 0.9998), for a 1.3–1.5× gain at w384l8 and 1.1× at w160l8.
- **BF16: rejected** by the rule, on 200-step noise. It is slower than float32
  at w160l8 and doubles padded learner memory there. At w384l8 it is 1.95–2.7×
  faster, uses less memory, and stacks on bucketing (196 ms against float32's
  381 ms bucketed step).
- **Next test: a short real learning A/B at w384l8**, rather than closing BF16
  on this replay. Train float32 and BF16 from the same seeds with bucketed
  shapes for a few hundred updates. Score strength against the frozen anchor at
  matched updates, with TF32 as a third arm if the budget allows. Strength at
  matched updates is the outcome that matters, and it is not chaotic in the way
  per-epoch replay losses are.
- `Execution.precision` stays as it is: typed, serialized in every stage,
  float32 by default. No recipe changes. The bf16 dtype joins ran without error
  through collection, the learner and both shape modes on CUDA at both sizes.

## Cost

$0.45 (launcher, run `f71ee505f7cc`), under the USD 3 cap.

## Limits

- One L4, one capture update, and one seed for each reference arm. A
  single-draw reference makes every ratio noisy. That is the main finding about
  the instrument, and the reason the rule was too strict.
- The learning comparison replays 20 fixed minibatches for ten epochs, beyond
  the one epoch the trainer takes per update. Clip fraction rises as the policy
  leaves the fixed behavior. It shows numerical drift, not on-policy training.
- w384l8 starts from a random initialization; w160l8 from the regime's own.
- No measurement of playing strength or of whole-update wall time; timings are
  replayed forward and learner calls.

<details>
<summary>Appendix: method, run and evidence</summary>

### Run

- Etude run `f71ee505f7cc` (cohort `mtg317-precision-1`), source `71c044113`,
  on-demand `l4-etude`, machine `l4-etude-9249bc075105`. NVIDIA L4, Torch
  2.15.0+cu132, 128 visible CPUs, 2026-11-12.
- The capture update took 197 s (110 s collection, 81 s learning). It kept 45
  collection observations, because the update made fewer than 576 forwards,
  and 20 learner minibatches.
- Dated line, 2026-11-12: the first `etude launch` at 15:49 UTC was refused
  (no L4 in stock). The run was queued under the same cohort, and
  `etude wait` rented the machine when stock returned. The question is
  unchanged.
- Launch control used `etude-sdk==0.2.15`, the release that wrote the local
  launcher database. The job installed the locked 0.2.16.

### Evidence

`experiments/data/cuda_precision/` (manifest entries in
`experiments/data.manifest.json`):
- `summary.json`: host, capture receipt, per-group verdicts, timings and
  fixed-weight differences.
- `trajectories.json`: every arm's per-step metrics and per-call timings.
- `regime.json`: the capture regime.
- `speed.svg`, `equivalence.svg`.

Reproduce:

```sh
uv run python -m experiments.cuda_precision smoke          # CPU, toy scale
uv run etude launch experiments/cuda_precision.py --rent l4-etude \
    --cohort NAME --max-usd 3 --revision COMMIT --hours 2.0 --usd 1.5
uv run --extra reports python -m experiments.cuda_precision charts \
    experiments/data/cuda_precision
```

</details>
