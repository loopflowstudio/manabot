# MTG-324 plan

Changed docs (write-docs step): `docs/training.md` § Reduced precision.

Branch base: merged the open MTG-317 branch (`jack/mtg-317-run-cuda-training-in-tf32bf16-with-a-measured`,
PR #489) into this branch, because the harness `experiments/cuda_precision.py` and the
"Reduced precision" doc section exist only there. If #489 lands first, the merge is a no-op.

The docs as written assume the outcome of two pending measurements; implementation must
make them true or edit them:
- `BF16_MIN_HIDDEN_DIM = 384` (in `manabot.training.regime`, torch-free, next to `Precision`).
  Becomes 256 if the w256l8 run shows BF16 learner step and large collection calls both
  >= 1.1x float32 with bucketed shapes. Update the "threshold" paragraph accordingly.
- The learning-comparison sentence and link to `experiments/precision_ladder.md` (the A/B
  report name chosen here). If BF16 is rejected, `auto` resolves to float32 everywhere and
  that paragraph says what separated.

Design implied by the docs:
- `auto` is a `Precision` literal and the `Execution` default; it is never serialized.
- Resolution needs the stage device and `regime.agent.hidden_dim`, so it runs in a
  `TrainingRegime` validator; off CUDA → float32 (never trips the CUDA-only validator).
- `with_capacity` / `with_agent` must re-resolve stages whose precision came from the
  ladder, while keeping explicit values. A dump→validate round trip loses "came from auto",
  so the builder needs to know (e.g. a PrivateAttr carried by model_copy, or builders
  re-resolve only when the stored value equals the ladder's value for the old width).
