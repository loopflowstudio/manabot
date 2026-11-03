# MTG-213 — decisions and assumptions (2026-11-03)

Jack Heart's comment 1540e0ad-743f-412b-ad5c-5cb55b03a0db supersedes the deferred
Task brief: start a preliminary fixed-corpus TLA self-play/evaluation pilot now;
reject whole unsupported decks. Comment 6434b4dd-5329-4904-9952-6493c9968b52
then authorized GPUs under a hard $10 cumulative cap and shipping after checks.
The earlier local-only/no-merge assumptions are superseded.

- The trophy API restricts reuse to 17Lands.com. Use the CC BY 4.0 public TLA
  PremierDraft game dump instead: first 1,000 events with seven observed wins
  in file order; each seventh-win main/sideboard; event/build IDs and recorded
  results retained. Missing loss rows are not invented. This is historical
  trophy data, not today's feed or a uniform sample of all draft decks.
- Native whole-list admission includes sideboards: 134 eligible, 866 rejected.
  Frozen snapshot published through the artifact manifest. No resampling.
- The evidenced MTG-181/183 64x2/four-head damped-policy recipe is preserved;
  current world/full vocabulary/v5 input and the population are necessary
  deviations. No historical checkpoint or dollar allowance is inherited.
- First L4 attempt capped at $1 inside the $10 total, 100-update target,
  2,700-second training watchdog, one attempt. Independent eight-pair/four-leg
  evaluations compare heuristic and initialization. Source/limits/criteria in
  experiments/protocols/tla-trophy-213-v1.{json,md}.
- Jack Heart corrected the credential assumption on 2026-11-03: nested
  Doppler manabot/prd then etude/prd supplies artifact and provider credentials.
  Available configured alternatives remain authorized under the same $10 cap;
  record hardware deviations before scoring. Dated capacity/admission evidence
  and allocation reconciliation live in scratch/mtg-213.md and the result report.
  Comment d2108241 requires a milestone only after actual GPU learner updates.
  The prior credential-blocker comment fce84671 is superseded.
- Local CPU proof is implementation evidence only; measurements and published
  artifacts live in experiments/tla-trophy-pilot.md. T6's separately owned
  learner optimization is not taken over to improve this proof's backward cost.

- Analysis interpretation (2026-11-03): all frozen milestones remain expected
  even if an early endpoint exists; an unknown endpoint keeps two explicit
  comparison slots. Failed Etude callbacks retain failed-match counts, while
  their unknown game attempts remain missing (attempted games is a lower bound).
  Etude's Report panels have no custom inference hook, so the companion
  `experiments.tla_pilot_analysis` CLI owns the qualified JSON result.

- Launch continuation (2026-11-03): source preparation took about five minutes
  and capacity disappeared before admission again. Prepare once with Etude's
  existing `source.freeze`, retain the typed `Run` under `.runs/mtg213/`, then
  quote and submit that same request through `client.launch_cohort`. This avoids
  repeating git history preparation between quote and admission; it changes no
  job, corpus, source identity, resource limits or provider policy. MTG-239
  retains ownership of general launcher improvements.

Inherited unrelated notes remain at
22e4aab716f2b582d704b88dc5d22cb1547ce623:scratch/questions.md.

Main’s MTG-238 assumptions and engine-speed integration notes remain at
`44388ad4211b4292cc161a3087eb1230b2005737:scratch/questions.md`.

- Post-merge reconciliation (2026-11-03): preserve the existing prepared source
  for the pilot continuation, as requested in the 20:04 direction. Current local
  w17 has a different card digest from the corpus even though every eligible deck
  admits. Exact launch-runtime identity was verified at 20:28 UTC; evaluation
  must use that same source/native world. Current-head execution is not silently
  substituted for the retained request. No corpus or protocol regeneration.

- GPU continuation (2026-11-03, 20:28 UTC): four configured templates refused
  quotes; `mtg188-a6000` quoted $0.59/hour and acquired an RTX A6000. Select
  this authorized hardware alternative with unchanged source/corpus/recipe and
  $1 job/$10 cumulative caps; record the hardware difference in pace reporting.

- Admission failure (2026-11-03, 20:32 UTC): the pinned GPU fixture compared
  actual learning time with injected persistence overhead and failed (8.9029s
  versus 8.6478s). Replace the fixture clock, not the numerical contract or pilot
  runtime. The exact-source instruction prevents silently repinning to current
  HEAD, whose card digest differs. A prospective repaired pin must preserve native/evaluation parity.
  Keep $0.7375 reserved until provider billing settles ($0.03275634 estimated).

- Prospective source correction (2026-11-03, before retry): retain the failed
  b8835f63b request unchanged. Build immutable experiment source
  `2adbd0758e40518947daf91569e1938b060d4346` from that exact parent with only
  `tests/training/test_numerical_health.py` corrected. All native, learner,
  corpus, recipe and evaluator bytes remain identical; this is not current HEAD.
  The ordinary Task repair checkpoint is c29bf3a3; the experimental source is a
  git object prepared by Etude, without another Task/branch/worktree. Use the
  corrected archive for both training and evaluation, retaining the original
  attempt and reservation. This reversible source correction is an engineering
  choice under the authorized bug-fix/pilot scope, not a changed learning method.

- Allocation ownership (2026-11-03): Etude 0.2.12 cannot append runs to an
  existing cohort; the first corrected quote stopped on that validation without
  contacting capacity. Use operational cohort `mtg213-pilot-v1-correction1`
  with a $1 ceiling, reserving the first cohort's full $0.7375 until billed.
  Total committed maximum is $1.7375, below $10. The experiment/protocol identity
  is unchanged; cohort names are allocation records, not new scientific samples.

- Capacity counterexample (2026-11-03): stream 181, seed 31482, action 87
  reproduces `facts 140 > 128` with matching pre-command frame. Seven legal
  The Fire Nation Drill activations over ten opposing permanents each generate
  two keyword-removal facts. Native and Python encoders agree. Fixed v5 capacity
  and deliberate fatal treatment of encoding errors prevent a local padding,
  truncation or quarantine workaround. Stop dependent paid work and return the
  representation decision to design/review; keep the scheduled initialization
  evaluation and every failed/missing denominator. No learning claim is possible.
