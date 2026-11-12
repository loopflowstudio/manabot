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

## Results

Pending the L4 run.
