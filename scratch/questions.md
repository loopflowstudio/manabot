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

- L4 availability on 2026-11-13 is USD 0.69/h (the USD 0.60 pool refuses
  it). Use the existing `mtg324-l4-ab` USD 0.70/h ceiling, 7.1 rental hours
  and job/cohort USD 5 caps. Preserve the 300-update schedule and use the
  ordinary wall-bound stop (~193 updates at the failing run's pace), rather
  than changing the schedule to force exactly 200 updates.
