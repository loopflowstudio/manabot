# Task 07c490 — closure and lambda on the ten-deck Elemental recipe (2026-11-05 UTC)

Inherited MTG-213 and fc6ffb notes no longer direct this work; they remain at
`ac539b38d:scratch/questions.md`. The protocol
`experiments/protocols/closure-lambda-elemental-v1.md` owns the design, predictions,
budget and results pointer; this file holds working assumptions and open items.

## Direction

- Jack Heart, 2026-11-05: repeat MTG-251's two comparisons on the ten-deck Elemental
  setup, $25 total, hard cap including failures and setup. MTG-251 landed as #431
  (ac539b38d), so the Task can start.
- Steer (comment ed18e2a5): no full 10x10 grid; sample random deck pairs, fix them and
  their deals before scoring, identical across arms, exchanging decks and seats within
  each pair; still scored against the heuristic.
- Steer (comment bc3b4385): report how much wall time two-seat Elemental runs lose to
  idle streams (`idle_stream_steps`); report it, do not build two-seat exact quota.

## Assumptions taken here (reversible, recorded in the protocol)

- Pairs are drawn with replacement from the 45 distinct unordered Elemental pairs;
  mirrors are excluded because exchanging a mirror's decks changes nothing (4 legs per
  pairing: both decks × both seats). 50 pairings × 4 legs = 200 games per checkpoint,
  the same count MTG-251 scored (25 deals × 8 legs).
- Study 1 is one-seat on both arms (exact quota needs one seat), without the recipe's
  game-error recovery (exact quota rejects it; the complete arm drops it too so the arms
  differ only in closure and floor/quota). The complete arm keeps the recipe's floor of
  64; the exact quota comes from calibration of the complete arm alone.
- The frozen Elemental regime names world w13; the variants relabel it to the current
  world (w17) with new IDs, as MTG-181 did for w10 → w13. `experiments/elemental.py`'s
  own `runs()` still loads the w13 file and would fail the current-world check; out of
  scope here.
- The CUDA numerics gate (`validate_numerics`) stays off, as MTG-251's launches; it is
  job-level, not recipe.
- `learning_signal.py`'s environment digest and conditions shape changed for the Allies
  design too (design record instead of matchup; conditions carry roles). The published
  MTG-251 evaluation is complete and frozen; a fresh `etude eval` of those files would
  treat the old matches as a different environment. Nothing plans one.
- `etude launch` (0.2.12 and 0.2.14 CLIs) cannot read `runs()` signatures with tuple
  parameters, so the Elemental files take `seeds`/`arms` as comma-separated strings and
  each arm launches as its own all-or-nothing cohort.
- Idle wall-time for two-seat runs is reported as an upper bound: collection seconds
  less batch assembly, times the idle share of stream slots. The one-seat closure study
  gives the direct measurement of finishing games against pausing.

- The calibration launches from `experiments/elemental_closure_calibration.py`: Etude
  names a run's experiment after its launch file's stem (`ETUDE_EXPERIMENT`), so it
  stays out of the closure study's evaluation passes.
- Each run's dollar limit equals its reservation (window + 15 min setup at $.60); the
  launcher refused $.36 for a .6 h window. Protocol amended before any run started.

## Log

- 20:07–21:10 UTC: no L4 stock for an hour (launch refusals, no runs, no cost), then the
  dollar-limit refusal above. 21:12 calibration launched: run 03e5613716a4, revision
  b2cd4de45, machine 07c490-l4-586f45c56a62. Terminated at 30 min as "never booted",
  then 21:47 ad86d25e2327 terminated at 15 min the same way. Both were wrong calls:
  each had a job attempt, and Etude shows `setting_up`/"output not received" until
  the end even on healthy runs (setup is budgeted 900 s). Charge both full
  reservations ($1.02) to the cap. "Not booted" means no attempt id in `etude logs`.
  Actual cause, found with `etude logs --run ID --follow`: all three (incl. 22:04
  d167b004c555, which held its machine to ~$.50 until terminated at 23:20) died at once
  with `experiment_differs_from_launch` from a `--calibration` flag that renamed the
  experiment. Calibration attempts so far: 3 failed, ~$1.50 reserved/billed.
- 23:27 calibration 491fa11b9fdf launched from its own file, revision 04c9cba75.

## Open

- Calibration quota (set `CLOSURE_QUOTA` in `experiments/elemental_variants.py`, amend
  the protocol, commit, then launch study 1).
- Provider bills: reconcile `runpodctl billing pods` against the launcher's `usd`.
