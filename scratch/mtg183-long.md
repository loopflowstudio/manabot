# MTG-183 live plan

Frozen protocol: `experiments/protocols/elemental-183.md` (the commit that adds it).
It stays the authority; this note tracks state.

## Delete — do not maintain

- Cohort windows in `experiments/elemental_long.py` (`L4`, `WINDOWS`, `JOBS`,
  `cohort()`, `specs`/`cohort` CLI) and the follower's S3/cohort admission
  (`observe`, `S3JobStore`, `CohortState`): deleted 2026-10-24 with #362's remote
  system. Frozen job specs `experiments/protocols/elemental-183-job-*.json` stay
  as w1/w2 evidence. The `manabot/remote` `User-Agent` repair went with #362.
- `experiments/elemental_grid.py` and the `play_cell` leg subset are MTG-181's
  unmerged bytes, copied unchanged; whichever branch lands second keeps one copy.

## Exists

- Cohort `mtg183-elemental-long`: seed 18301, seven L4 windows (6 × 11.9 h + 6 h),
  $48.70 reserved of $50, exports every 50 updates, learner-state continuation at
  `last-complete`. Controller: launchd cohort service on this laptop.
- `experiments.elemental_long_follow` (launchd) plays heuristic grids and
  cross-checkpoint pairs into `.runs/mtg183/follow`.
- Local h2h worker smoke (2026-10-23, MTG-181 18103 u568 v u200, cross and mirror
  legs): valid, replay-checked, ~3.5 s/game under load.

- Report: `experiments/elemental_long_report.py` (4e707eb52; synthetic test).
  Saturation margin 0.03 amended 2026-10-24 02:10 UTC before any game.
- Operations: pinned worker/controller/follower checkout
  `/Users/jack/.manabot/mtg183-source` (detached 8e7df92e4); cohort state
  `/Users/jack/.manabot/mtg183-elemental-long` (launchd
  `manabot.cohort.mtg183-elemental-long`); follower
  `/Users/jack/.manabot/mtg183-follow` (launchd `gg.manabot.mtg183.follower`,
  until 1793584946). Cohort admitted 2026-10-24 02:02 UTC; w1 deadline 13:56 UTC.
  Keep `--extra artifacts` on every `uv run` in the pinned checkout (a bare
  `uv run` would re-sync the venv and drop the service's dependencies).

Latest check (2026-10-24 02:23 UTC): w1 running (TrainingRun 67ec6c48…), update
32, 0 crashed games, ~49k learner decisions/update, ~120 updates/h; update 0
published and admitted, its full grid queued behind MTG-181's follower (pid 23706).
Snapshot uploads fail intermittently (sticky `snapshot upload failed (ValueError)`,
as in MTG-181); exports appear whenever a later generation succeeds.

## Remains

- After the cohort settles and the follower finishes: results report, publish
  evidence with `manabot.data publish`, commit manifest, protocol Results pointer.

## Proposed Wave memory (not written; realign owns it)

- One RunPod job is capped below twelve hours; a long seed is a cohort of
  `last-complete` continuations. The cohort service re-checks the frozen source
  before each successor, so run it (and its follower) from a detached pinned
  checkout; the Task branch can keep moving.
- The report's synthetic test showed a strict "gain not significant" saturation
  rule never fires at 5,000 paired games; use a practical margin.

## 2026-10-24 18:49 UTC — cohort controller retired by the Etude migration

Jack Heart's migration session unloaded `manabot.cohort.mtg183-elemental-long`
(plist now `.plist.disabled`); w3–w7 never start on the old system. w2 runs to
its pod deadline 2026-10-25 01:47 UTC (≈ update 2,550). w1 ended at 1325.
Follower still running; with no controller, cohort status stays `running`, so
the follower never settles or admits an endpoint by itself. Decision pending:
continue w3–w7 through Etude (protocol amendment, follower must learn new job
IDs) or stop at w2 and declare its final export the endpoint.

### Decision (Jack Heart, 2026-10-24 ~18:55 UTC): continue through Etude

Continue the same seed past w2 on Etude with the remaining cap (≈ $50 − w1 − w2
actual cost), keeping one learner lineage. No new seed.

What blocks it today (verified on #362's branch `jack-heart/remote-on-etude`):
- `manabot/experiment/job.py::run_worker` refuses `ETUDE_CHECKPOINT` /
  `ETUDE_FROM_VALUE` ("policy artifacts cannot resume collector/optimizer
  state"), and requires the whole regime `wall_seconds` to fit the allocation.
- #362 deletes `manabot/remote/continuation.py`, the old learner-state binder.
- The regime still supports it: `TrainSelfPlay.initial: LearningStateImport`
  (raw, ema, optimizer + parent run reference, staged paths; RNGs restart).

Plan (Jack Heart's steer): build on #362 (`jack-heart/remote-on-etude`), allocate
nothing until it is on main; do not touch w2's pod, MTG-181's follower or S3 folders.

Done 2026-10-24 (branch merges #362's head `e0009ad57`; merge, not rebase, so the
frozen launch commit `8e7df92e4` stays an ancestor):
1. `TrainingJob.continuation` (`LearnerContinuation`, `StagedExport`) in
   `manabot/experiment/job.py`: parent run.json + raw/EMA/Adam as Etude inputs,
   bound as `LearningStateImport`; refuses another parent update or seed.
2. `experiments.elemental_long stage` (upload + digest check against the parent
   record) and `segment` (Etude Job JSON for w3).
3. Protocol amendment "continue on Etude after w2" (rules; numbers go in
   `experiments/protocols/elemental-183-w3.json` at staging, before allocation).
4. Follower admits Etude segment exports, settles on Etude run done.
Check: `uv run --extra dev pytest tests/training/test_etude_job.py tests/training/test_learning_state.py tests/arena/test_elemental_long*.py tests/arena/test_elemental_grid.py tests/sim/test_tracking.py -q` → 39 passed.
Compress 2026-10-24: same check → 39 passed (follower `Etude.source` dropped, role/file mapping shared via `LearnerContinuation.files`).

Remaining, in order:
- After w2 ends (deadline 2026-10-25 01:47 UTC): fetch w2's final run.json and
  `self-play-{raw,ema,optimizer}.pt` into a local dir (pinned checkout's old
  tooling can read the S3 job folder), `stage` them to the Etude machine
  storage, compute hours/target/max_usd per the amendment, commit
  `elemental-183-w3.json`.
- When #362 is on main: merge main, land or publish this change so w3's source
  revision is on main, then launch w3 (`queue_cohort`/`launch` with `rent="l4"`).
- Follower cutover per the amendment (stop `gg.manabot.mtg183.follower`, restart
  from the w3 commit with `--etude-source`/`--segment`).
- Then: results report, publish evidence with `manabot.data publish`, manifest,
  protocol Results pointer.
