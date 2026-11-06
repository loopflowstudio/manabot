# Task 07c490 — closure and lambda on the ten-deck Elemental recipe (2026-11-05 UTC)

Inherited MTG-213 and fc6ffb notes no longer direct this work; they remain at
`ac539b38d:scratch/questions.md`, and main's MTG-243 naming notes at
`f6f171d34:scratch/questions.md`. The protocol
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

- Calibration: four attempts, ~$1.30 billed, $2.04 charged. Three died on launch errors
  of this study's making (an undersized dollar limit; a `--calibration` flag that
  renamed the experiment → `experiment_differs_from_launch`; Etude names the experiment
  after the launch file's stem). Two of those were also terminated early on the false
  belief that `setting_up` meant not booted; healthy runs show `setting_up` and "output
  not received" until done, and setup took 12.5 min. Inspect with
  `etude logs --run ID --follow`. A failed job can keep its machine rented: terminate it.
- 491fa11b9fdf (04c9cba75) trained 9 updates (~70 s each, one seat) and died on
  continuation relations 65 > 64. Study 1 stopped under its kill criterion; protocol
  amended (6d7c84db2). Lambda arms set `recover_capacity`, windows 1.5 h / $1.05.
- Lambda baseline (.5/.8) c608a0fa7d29/04cef7d8ce8e/67f108f57c2e: all reached 75
  updates, ~31–34 s median update, 19.7–19.9k games, 46–51k rows/update (~190/stream
  vs floor 64). Idle share .637–.650; idle collection bound 771–799 s of 1279–1356 s
  collection (+1087–1193 s learning). Controller died 00:43 (0.2.13 can't parse Vast
  providers); restarted from the 0.2.14 overlay (`/tmp/e07c490 controller run`).
- Lambda arms launch one seed per launch (`/tmp/07c490w/launch_seeds.sh`): L4 stock
  rarely covers three at once. Six ghost records filled the 8-machine limit ("it is
  busy") and resist `etude terminate`; `etude ondemand max 07c490-l4 14` (03:15).
  .8/.8: 6952e8b1391f (s70201, stopped at 72), ff12bf198bb3, 2e45a87e9de4.
  Eval pass 1 (elemental-lambda-v1-6860cc47ef30, 07c490-cpu32): 550 matches, 02:42.
- All 15 lambda runs succeeded (05:35). Endpoints: 75 except .8/.8 s70201/s70202 at
  72, .5/.95 s70202 at 64, .8/.95 s70202 at 73 (70-min wall on slow machines: median
  update ~52–63 s vs ~28–36 s; machine speed is bimodal, so wall-time comparisons are
  machine-confounded). Idle share .626–.670 in every run; idle bound 747–1249 s,
  about 29–34% of collection+learning. Capacity overflows: 3 games of ~290k (one each
  in .5/.8 s70201, .5/.95 s70201, .8/.95 s70203), all recovered.
- Billed (runpodctl, by pod): calibration $1.19; everything since 00:06 UTC $8.83 by
  05:59 → $10.01. Ghost `07c490-cpu32-*` records match no pod.
- Eval: a refused cpu32 create (RunPod 503s ~05:41) left a ghost session that pins
  the run id Etude derives from the pending set (`{name}-{digest(pending keys)}`);
  `terminate`/`cancel` don't clear it. Pass 2 played a different set (no u20, a
  temporary commit since reverted): elemental-lambda-v1-e486e50e04a7, 1200 matches,
  07:10. The u20 remainder needs one more pass.
- Ghost `07c490-l4-*` records from refused creates show estimated dollars but no RunPod
  pod exists (`runpodctl pod list` empty); not charged.

## Open

- Study 1 needs a decision beyond the protocol: exact-quota capacity recovery, wider
  continuation relations for both arms, or deleting exact quota.
- Bills for eval passes 2–3 (07:10–11:00 UTC 2026-11-06) had not posted; the report's
  ~$4.8 evaluation and ~$13.3 total are estimates from rental windows. Reconcile with
  `runpodctl billing pods --grouping podId --start-time 2026-11-06T07:00:00Z`.
- The Etude controller is not running: 0.2.13 (`~/.manabot/etude-controller`) cannot
  parse the Vast providers; the 0.2.14 overlay run hung on storage reads after ~2 h.
- `07c490-l4` on-demand max is 14 (raised from 8 for ghost records).
