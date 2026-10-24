# MTG-183 live plan

Frozen protocol: `experiments/protocols/elemental-183.md` (the commit that adds it),
with its amendments (0.03 saturation margin, 2026-10-24 02:10 UTC; "continue on
Etude after w2", 2026-10-24 19:00 UTC). It stays the authority; this note tracks state.

## Decisions

- Jack Heart, 2026-10-23: launch one long seed within the $50 cap.
- Jack Heart, 2026-10-24 ~18:55 UTC: after the migration retired the cohort
  controller, continue the same seed past w2 on Etude with the remaining cap;
  one learner lineage, no new seed. Build continuation on #362; allocate nothing
  until #362 is on main; do not touch w2's pod, MTG-181's follower or S3 folders.

## Delete — do not maintain

- Cohort windows/CLI in `experiments/elemental_long.py` and the follower's
  S3/cohort admission: deleted with #362's remote system. Frozen job specs
  `experiments/protocols/elemental-183-job-*.json` stay as w1/w2 evidence.
- `experiments/elemental_grid.py` and the `play_cell` leg subset are MTG-181's
  unmerged bytes, copied unchanged; whichever branch lands second keeps one copy.

## Exists

- Seed 18301 on the old cohort system: w1 ended at update 1,325; w2 running to its
  pod deadline 2026-10-25 01:47 UTC (≈ update 2,550). Cohort controller
  `manabot.cohort.mtg183-elemental-long` unloaded (plist `.disabled`); w3–w7 never start.
- Follower `gg.manabot.mtg183.follower` (pinned checkout
  `/Users/jack/.manabot/mtg183-source`, detached 8e7df92e4; state
  `/Users/jack/.manabot/mtg183-follow`). Keep `--extra artifacts` on every `uv run`
  there. Ledger at 2026-10-24 18:56 UTC: 37 exports admitted (0…1,750), 44 units
  all `queued`, none played — it still yields to MTG-181's grid follower (pid 23706).
  Its `error` lists w3–w7 `ValueError` (jobs that will never exist); cohort phase
  stays `running`.
- This branch (merges #362's head `e0009ad57`; merge, not rebase, keeps
  `8e7df92e4` an ancestor): `TrainingJob.continuation` in
  `manabot/experiment/job.py`; `experiments.elemental_long stage`/`segment`;
  the protocol amendment; follower admission of Etude segment exports.
- Report: `experiments/elemental_long_report.py` (synthetic test only).

## Remains, in order

1. After w2 ends: fetch w2's final run.json and `self-play-{raw,ema,optimizer}.pt`
   (pinned checkout's tooling reads the S3 job folder), `stage` them, compute
   hours/target/max_usd per the amendment, commit `elemental-183-w3.json`.
2. When #362 is on main: merge main, land this change so w3's source is on main,
   launch w3 (`rent="l4"`, one attempt).
3. Follower cutover per the amendment (restart from the w3 commit with
   `--etude-source`/`--segment`, same ledger). The Etude read path
   (`etude.train.read.read`, tracking URI argument) is unverified against a live run.
4. Local evaluation: the protocol expects ~51 h of grids/pairs, mostly overlapping
   training, but none has started. If MTG-181's follower keeps running, the
   follower's stop deadline (w3 deadline + 2 days + ~9 h) may leave units missing;
   they are reported as missing, not dropped.
5. Results report, publish evidence with `manabot.data publish`, manifest,
   protocol Results pointer.

## Proposed Wave memory (not written; standalone realign leaves `.lf/wave/` read-only)

- One RunPod job was capped below twelve hours, so the old cohort chained
  `last-complete` continuations; on Etude, learner state continues through
  `TrainingJob.continuation` (raw/EMA/Adam + parent run record, refusing another
  parent update or seed; collector/RNG streams restart).
- Run long-lived followers from a detached pinned checkout so the Task branch can move.
- The report's synthetic test showed a strict "gain not significant" saturation
  rule never fires at 5,000 paired games; use a practical margin.
- Local evaluation that yields to another Task's follower can stall entirely;
  declare its expected wall time against the real queue, not idle capacity.

Check (2026-10-24): `uv run --extra dev pytest tests/training/test_etude_job.py tests/training/test_learning_state.py tests/arena/test_elemental_long*.py tests/arena/test_elemental_grid.py tests/sim/test_tracking.py -q` → 39 passed (after compress; not rerun by realign — only the amendment header time changed).
