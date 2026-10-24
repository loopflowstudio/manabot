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
