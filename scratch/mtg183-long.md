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

## Remains

- Report script: BT fit/bootstrap (`manabot/arena/rating.py`), payoff matrix,
  residuals, intransitive triangles, held-out log loss, prefix-refit stability,
  heuristic + rating curves per deck, saturation rule; Etude runs/pages.
- After the cohort settles and the follower finishes: results report, publish
  evidence with `manabot.data publish`, commit manifest, protocol Results pointer.
