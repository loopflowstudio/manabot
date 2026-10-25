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

- Cohort windows/CLI and the follower's S3/cohort admission: deleted with #362.
  Frozen job specs `experiments/protocols/elemental-183-job-*.json` stay as w1/w2 evidence.
- `experiments/elemental_long_follow.py` and its test: deleted (2026-10-25),
  replaced by `experiments/elemental_long_eval.py` (Etude eval). The pinned
  checkout `/Users/jack/.manabot/mtg183-source` still runs the old follower
  until cutover; its ledger is read-only input for w1/w2 checkpoints.
- `experiments/elemental_grid.py` is MTG-181's unmerged bytes, copied unchanged;
  whichever branch lands second keeps one copy.

## Exists (2026-10-25 00:00 UTC)

- Branch merged main (#362, heuristic v2 `639ab7770`) at `261c62027`; etude-sdk 0.2.1.
- Seed 18301: w1 ended at update 1,325; w2 running to its pod deadline
  2026-10-25 01:47 UTC (≈ update 2,550). Old cohort controller unloaded.
- Old follower `gg.manabot.mtg183.follower` stopped (`launchctl bootout`,
  2026-10-25 00:15 UTC; plist kept). MTG-181's follower had exited and MTG-181
  moved to `elemental_181_new_heuristic`, which the follower's process-name yield
  does not match, so at 23:49 UTC it began `heuristic-000000-full` against
  heuristic v1 and stopped admitting exports mid-unit. Those partial v1 groups
  are not evidence. Ledger `/Users/jack/.manabot/mtg183-follow/follow/ledger.json`
  (copy `ledger-before-stop-20261025T0015Z.json` beside it): 48 exports admitted, 0…2,200+.
- `experiments/elemental_long_eval.py`: one Etude eval run (`mtg183-elemental-eval`
  / `mtg183-eval`), one match per grid row, tables delivered to
  `r2://manabot/evaluations/`; checkpoints from the old ledger (w1/w2) and w3's
  Etude run; `<directory>/state.json`. Report reads `statsdb` + state.
  R2 delivery and `statsdb` read-back verified with a throwaway prefix (deleted);
  one real game each of heuristic/subset/pair played through `play`.
- Protocol amendment "heuristic v2, Etude eval, R2" (2026-10-25 00:00 UTC).
- `fetch_etude_checkpoint` takes an `ArtifactRef` and reads S3 or R2.

## Remains, in order

1. After w2 ends (01:47 UTC): admit w2's remaining exports into the old ledger
   once, without playing: from the pinned checkout, hold `owner.lock` and call
   `experiments.elemental_long_follow._admit`, then write the ledger. Do not
   restart the launchd follower (its loop plays pending v1 units).
2. Start `elemental_long_eval` from a pinned checkout of this change
   (`--windows` the old ledger, `--storage r2://manabot/evaluations`, Doppler
   `manabot/prd` for R2 keys; `--etude-source/--run` added once w3 exists).
3. Fetch w2's final run.json and raw/EMA/Adam exports, `stage` them to R2 Etude
   storage, compute hours/target/max_usd per the amendment, commit
   `elemental-183-w3.json`. `uv run etude storages` refuses: `~/.etude/launcher.db`
   "was written by another version of etude" — resolve before launch (do not delete it blindly).
4. Land this change on main; launch w3 (`rent="l4"`, one attempt, R2 storage);
   restart the evaluator with `--etude-source/--run`.
5. S3 cleanup below (after w2 staged), then point w1/w2 checkpoints at R2 URIs
   (identity unchanged).
6. Results report, publish evidence with `manabot.data publish`, manifest,
   protocol Results pointer.

## Proposed Wave memory (not written; standalone realign leaves `.lf/wave/` read-only)

- One RunPod job was capped below twelve hours, so the old cohort chained
  `last-complete` continuations; on Etude, learner state continues through
  `TrainingJob.continuation` (raw/EMA/Adam + parent run record, refusing another
  parent update or seed; collector/RNG streams restart).
- Run long-lived followers from a detached pinned checkout so the Task branch can move.
- Yielding by another Task's process name fails silently when that Task
  switches scripts; MTG-183's follower then played a stale anchor and stopped
  admitting exports. Admission and play should not share one blocking loop.
- The report's synthetic test showed a strict "gain not significant" saturation
  rule never fires at 5,000 paired games; use a practical margin.
- Local evaluation that yields to another Task's follower can stall entirely;
  declare its expected wall time against the real queue, not idle capacity.

### S3 cleanup (Jack Heart via the Etude storage-cost session, 2026-10-24)

When w2 and both old-system followers have stopped: copy keepers (checkpoints
.pt, final run/run.json, latest manifest, small control files, Etude metrics)
from `s3://etudefantasia/manabot/jobs/mtg183-*` and
`s3://etudefantasia/manabot/cohorts/mtg183-elemental-long/` to R2 `manabot` at
the same relative path minus `manabot/`, verify by size/sha256, then delete
every version and delete marker of those folders. Report copies and bytes
deleted. w3 must stage w2's learner state first. Never start old-system runs.

### Direction (Jack Heart, 2026-10-24 ~22:30 UTC): R2 and new Etude code wherever possible

"Let's move over to R2 and the new Etude code as much as possible": w3 on Etude
storing to R2 (SDK 0.2.1), evaluation as Etude eval with tables in R2 (done:
`elemental_long_eval`, explicit `Match`es; the frozen schedules and the rating
fit unchanged), retire the bespoke follower (deleted), w1/w2 artifacts to R2
after w2 ends, report data via `manabot.data publish`.
**Anchor decided (Jack Heart, 2026-10-24): heuristic v2** (`639ab7770`); recorded
in the protocol amendment. v1 scores (MTG-181, ladder) are not comparable.

Check (2026-10-25): focused suite (`test_elemental_long*`, `test_elemental_grid`,
`test_etude_job`) → 31 passed after `Unit.older`; the wider 49-test set passed before it. Gate owns the wider suites.
