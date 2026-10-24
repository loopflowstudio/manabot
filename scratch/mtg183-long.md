# MTG-183 live plan

Frozen protocol: `experiments/protocols/elemental-183.md` (the commit that adds it).
It stays the authority; this note tracks state.

## Delete — do not maintain

Nothing replaced. `experiments/elemental_grid.py`, the `play_cell` leg subset and
the `User-Agent` repair are MTG-181's unmerged bytes, copied unchanged (the grid
gained a `games` subset argument); whichever branch lands second keeps one copy.

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

Plan (build on main after #362 lands; do not fork its deleted system):
1. `TrainingJob` takes a learner-state import as Etude `Input`s (parent run
   record, raw, ema, optimizer), resolves them under `ETUDE_INPUTS_DIR`, and
   trains to an absolute update target. Tests; no change to fresh-run behavior.
2. Stage w2's `last-complete` learner exports from the old S3 job folder
   (skipped by the S3 cleanup) into Etude inputs, digest-checked.
3. Protocol amendment, committed before allocation: windows after w2 run on
   Etude; segment lengths/hours and dollar cap from actual w1+w2 cost; same
   export cadence (50) and absolute update coordinates; the endpoint rule.
4. Follower: admit the Etude run's milestone checkpoints by absolute update and
   settle on the Etude run's terminal state instead of the retired cohort
   status (which now stays `running` forever).
