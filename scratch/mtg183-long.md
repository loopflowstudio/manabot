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

## Exists (2026-10-25 01:30 UTC)

- Branch merged main (#362, heuristic v2 `639ab7770`) at `261c62027`, then #360
  (`arena/` split into `sim/` + `eval/`, etude-sdk 0.2.2) at `9871e3545` (pushed).
- Seed 18301: w1 ended at update 1,325, w2 at 2,534 (below). Old cohort
  controller unloaded.
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

## Etude launcher (2026-10-25 00:20 UTC)

`~/.etude/launcher.db` was schema 1 (2026-10-15) with 0 runs/rentals/sessions;
moved aside to `launcher.db.schema1-20261025`, and 0.2.1 created schema 10.
Registered: storage `r2` = `r2://manabot/jobs`; provider `runpod` (storage r2);
on-demand `l4` (NVIDIA L4, 6 vCPU, 48 GB, 50 GB disk, max 1, ≤ $0.60/h, w1/w2's
image digest `cb154fcc…`, startup 900 s, delivery 300 s). Keys: `RUNPOD_API_KEY`
from Doppler `etude/prd`, R2 (incl. `R2_API_TOKEN`) from `manabot/prd`.
`etude availability --rent l4`: $0.59/h, rentable (00:17 UTC; again under 0.2.2
at 01:26 UTC, run as `doppler run -p etude -c prd -- doppler run -p manabot -c prd
--preserve-env=RUNPOD_API_KEY -- uv run etude …`).

## Evaluator (running since 2026-10-25 00:22 UTC; re-pinned ~01:25 UTC)

launchd `gg.manabot.mtg183.eval` (plist in `~/Library/LaunchAgents`) runs
`elemental_long_eval` under `doppler run -p manabot -c prd` from the pinned
checkout `/Users/jack/.manabot/mtg183-eval-source` (`9871e3545`, etude-sdk 0.2.2,
`_managym` from this worktree's #360 build), directory
`/Users/jack/.manabot/mtg183-eval`, `--windows` the old ledger (re-read every
poll), `--storage r2://manabot/evaluations`, `--until 1793584946`. It yields
(silently; `state.json` refreshes each poll) while any experiment from
MTG-181's checkout runs, so its wall time follows MTG-181, not idle capacity.
The pin moved from `d9fc58df2` with 0 rows played because #360 changed
`INFERENCE_CODE` (agent identity). Later restarts (adding `--etude-source/--run`)
keep identities only if the pin's `INFERENCE_CODE` files are unchanged; rows
played under different pins are not comparable.

## w2 end and w3 spec (2026-10-25 02:30 UTC)

- w2 ended 01:47 UTC at update 2,534 (`paused`), $7.2358. One-shot `_admit`
  (old checkout, under `owner.lock`) took the ledger to 53 exports, 0…2,534
  (copy `ledger-before-tail-admit-20261025T0150Z.json`).
- w2's run.json/raw/EMA/Adam fetched to `~/.manabot/mtg183-w2-final/parent`,
  staged to `r2://manabot/jobs/mtg183/w2/`, each verified by sha256.
- Staging first bound update 1,325: #365 made `StageRecord.updates` explicit and
  `TrainingRun` read w1/w2's records as 0 updates (the worker's admission would
  also have refused the parent). Fixed in `418eb456c` (rule moved from
  `RetainedRun` to `RunRecord`; tagged `mtg183-w3-source`); local admission of
  the real w2 record then gave iteration 2,534.
- Etude caps a rental at 43,000 s, so one 57.7 h w3 cannot run (worker would
  refuse its deadline). Amendment "segments of one rental each" (02:30 UTC):
  a chain w3, w4, … sized from receipts. w3: 11.6 h, $7.11, target 3,500;
  quoted $6.99 at $0.59/h. Spec `experiments/protocols/elemental-183-w3.json`.
- Launch as `Run(job, seed=18301)` (the worker needs `ETUDE_SEED`); MTG-186 holds
  the shared `l4` on-demand's one slot, so `etude ondemand max l4 2` before launch.

## Remains, in order

1. Land on main; launch w3 (`launch(Run(job, seed=18301), rent="l4")`) with a
   standing `etude controller run`; restart the evaluator with
   `--etude-source/--run` on a pin whose `INFERENCE_CODE` files match `9871e3545`.
2. After each segment ends: fetch its final run.json/raw/EMA/Adam from Etude,
   `stage` them under `mtg183/w<N>`, size and commit `elemental-183-w<N+1>.json`,
   launch; stop per the amendment. The evaluator must admit every segment's run.
3. S3 cleanup below, then point w1/w2 checkpoints at R2 URIs (identity unchanged).
4. Results report, publish evidence with `manabot.data publish`, manifest,
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

Check (2026-10-25 01:25 UTC, at `9871e3545`, load average ≈ 200 from MTG-181):
`tests/eval/test_elemental_long{,_report,_eval}.py`, `test_elemental_grid.py`,
`tests/training/test_etude_job.py` → 30 passed, 1 failed
(`test_worker_runs_real_learner_and_reports_admitted_policy`: its 30 s subprocess
timeout expired under that load; rerun when the machine is idle, gate owns it).
