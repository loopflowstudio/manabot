# MTG-345 assumptions

- The Task brief expects the fix to be "keep the policy head's logits,
  log-softmax and action log-probability in float32 inside autocast". The
  failing revision already did that; the retained incident shows a device-vs-host
  `exp` rounding disagreement on a subnormal probability instead. The plan
  fixes that cause (stop storing the device `exp` copy). Assumed in scope as
  "fix it at the cause".
- The brief says "keep float32's [tolerance] unchanged". The cause is
  precision-independent, so the same change applies to float32. No remaining
  tolerance is widened; one comparison is removed with its second operand.
- The proof run uses the A/B's layout (bucketed, no live tokens) so it is the
  same recipe that failed; bucketed + live tokens is covered by the on-pod
  probe only, since USD 5 buys one run.
- The branch was fast-forwarded to main after PR #510 merged. Main carried
  MTG-324's `scratch/plan.md`, `qa.html` and `questions.md`; they are removed
  here (history: `f4e1822fa`).
