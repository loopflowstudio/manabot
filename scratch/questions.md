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
