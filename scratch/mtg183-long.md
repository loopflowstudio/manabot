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

## Concept review (2026-10-24): one Etude segment — implemented

The accepted contract is one learner lineage with exactly one Etude segment (w3):
"No retry, extension or further segment." The follower now follows one run:
`--run <etude run>`, adapter `SegmentSource`, `Ledger.segment_outcome` from Etude's
typed status (any terminal state settles the segment; the report reads it), endpoint
= max admitted update, so a w3 that fails before its first export leaves w2's final
update as the endpoint, as the amendment requires. w1/w2 ledger entries unchanged.
No other concept changes: `LearnerContinuation` (storage keys, pre-launch) vs
`LearningStateImport` (bound local paths, regime admission) are distinct stages of
the same import; window (w1/w2, cohort) vs segment (w3, Etude) matches the
amendment. No open product choice.

## Remains, in order

1. After w2 ends: fetch w2's final run.json and `self-play-{raw,ema,optimizer}.pt`
   (pinned checkout's tooling reads the S3 job folder), `stage` them, compute
   hours/target/max_usd per the amendment, commit `elemental-183-w3.json`.
2. When #362 is on main: merge main, land this change so w3's source is on main,
   launch w3 (`rent="l4"`, one attempt).
3. Follower cutover per the amendment (restart from the w3 commit with
   `--etude-source`/`--run`, same ledger). The Etude read path
   (`etude.train.read.read`, tracking URI argument) is unverified against a live run.
   The live ledger (37 checkpoints, 44 units) loads into the current `Ledger`;
   its stale `cohort_phase`/`cohort_settled` fields are dropped on first write.
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

Check (2026-10-24): `uv run --extra dev pytest tests/training/test_etude_job.py tests/training/test_learning_state.py tests/arena/test_elemental_long*.py tests/arena/test_elemental_grid.py tests/sim/test_tracking.py -q` → 39 passed (2026-10-24 realign, at 626360b32).

### S3 cleanup (Jack Heart via the Etude storage-cost session, 2026-10-24)

When w2 and both old-system followers have stopped: copy keepers (checkpoints
.pt, final run/run.json, latest manifest, small control files, Etude metrics)
from `s3://etudefantasia/manabot/jobs/mtg183-*` and
`s3://etudefantasia/manabot/cohorts/mtg183-elemental-long/` to R2 `manabot` at
the same relative path minus `manabot/`, verify by size/sha256, then delete
every version and delete marker of those folders. Report copies and bytes
deleted. w3 must stage w2's learner state first. Never start old-system runs.

### Direction (Jack Heart, 2026-10-24 ~22:30 UTC): R2 and new Etude code wherever possible

"Let's move over to R2 and the new Etude code as much as possible." Applied to
the remaining MTG-183 work:
- **w3 on Etude, storing to R2.** Etude v0.2.1 (#33, #36) has R2 launcher and
  tracking storage plus live checkpoint publishing; Manabot main still pins
  `etude-sdk==0.2.0`. Bump the pin to 0.2.1 (lock, extras) in this change and
  point w3's launcher/tracking storage at R2 `manabot`.
- **Evaluation on Etude eval, not the bespoke follower.** No follower game has
  been played yet (all units queued), so switching costs no evidence. Express
  the heuristic grids and the cross-checkpoint pairing graph as Etude
  `Benchmark`/`Matchup`s (or `etude.eval.match.evaluate` with explicit
  `Match`es), `play` wrapping the replay-checked arena `play_cell`, run on this
  Mac (local workers, $0), tables in R2, read back with `statsdb`/`compare`
  and drawn by `etude report` at each checkpoint's coordinates. Keep the
  frozen schedules (deal/seat seeds, digests) and the rating fit
  (`manabot/arena/rating.py`) unchanged; record the change in a protocol
  amendment before the first result. Retire `experiments/elemental_long_follow.py`
  once nothing reads it (delete dead code).
- **w1/w2 artifacts to R2.** After w2 ends: stage w2's learner state for w3,
  then the S3 cleanup above (copy keepers to R2, verify, delete all versions).
  Followers then read w1/w2 checkpoints from R2.
- **Report data** via `manabot.data publish` (now R2).
- Open question for Jack: which heuristic anchors the curve (v1 frozen in the
  protocol vs v2 from 639ab7770). Quick 10/cell grid of u2200 vs v2 is running.
