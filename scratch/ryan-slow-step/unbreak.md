# Slow payment recovery — 2026-11-12

Session `session_6f1af8add1a543e894df5268ae665776` is implementing and validating
the payment bound in `managym/src/flow/mana.rs` in this checkout. Original report
and standalone repro are retained unchanged alongside this note.

The saved PyPI 0.2.1 run completed with a 33.65 s final step. The initial run
labelled 0.2.2 imported this checkout instead; its output is now retained as
`local-environment-aborted.log` (step 242: 43.57 s, subsequently SIGTERM).
Do not treat it as a released 0.2.2 measurement.

Sampling that local process with symbols at 15:46:25 showed all 398 samples
inside `PaymentModel::witness`, reached through `legal_activation_x_values`
and `waterbend_payment_possible`. Full sample: `current-sample.txt`.

The usable-mana upper bound summed fixed mana, waterbending and each tapping
mana activation on one source, despite all consuming the same untapped
resource. The candidate fix counts the maximum usable output of those
exclusive choices, while continuing to sum non-tapping activations
optimistically. It changes rejection of impossible search states, not action
order or the first executable payment witness. Validation is in progress.
