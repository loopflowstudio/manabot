# Open questions and assumptions

- MTG-339: the feature-gated payment instrumentation was removed from the
  working tree and is not shipped; it survives only as
  `scratch/payment-baseline/instrumentation.patch` (its `classify` predates the
  direct answer and would need rewriting to apply). Assumed stock-binary
  instruction counts are enough for the after-measurement. Reapply and document
  it in `docs/development.md` if per-class counters are wanted on main.
- MTG-339: a revision-keyed model cache across legal-action computations
  (61% of builds repeat within a step, and priority actions are computed about
  3.6 times per step) was not attempted; `PaymentScan` shares only within one
  computation, where soundness needs no key. Why priority actions are computed
  that often per step is outside this Task.
- MTG-339: `direct_witness` panics (`expect`/`Mana::pay`) rather than falling
  back if its arithmetic ever disagreed with `direct_payable` in release.
