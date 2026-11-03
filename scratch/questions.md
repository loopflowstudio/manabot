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
- GPU launch is blocked: RUNPOD_API_KEY absent directly and through configured
  manabot/prd Doppler. No rental/job, $0 paid spend. No auth changes. Jack Heart's
  d2108241-34f0-4685-88b5-5b91b77ebf32 requires a milestone only once actual GPU
  updates are confirmed. The blocker was sent as Task progress, not a launch
  milestone (progress fce84671-16e1-43d1-a34b-22a1ee63c874).
- Separate local implementation proof: one update, 199 decisions, two games;
  12 evaluation games all replay-valid. No improvement/pace claim. Source patch,
  native bytes, checkpoints, optimizer and results are manifest-published.
- PRs #414 and #383 were open at inspection; T6 unpublished learner optimization
  is not taken over. The CPU proof's backward cost remains evidence for that
  shared work, not permission for a parallel optimization experiment.

Inherited unrelated notes remain at
22e4aab716f2b582d704b88dc5d22cb1547ce623:scratch/questions.md.
