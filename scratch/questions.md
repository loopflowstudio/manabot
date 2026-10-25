# MTG-186 questions and assumptions

Earlier scratch notes (card evaluation/GIH design, MTG-174/176 history) were
inherited from landed work and are at `1e6015270:scratch/`.

- 2026-10-24: MTG-181 is not on main; its grid evaluator and `play_cell` leg
  subset live on its branch (at 07a091bca, before #360 split `arena/`). This
  branch ports both to main's `eval`/`sim` modules with identical behavior.
  When MTG-181 lands it will conflict on `experiments/elemental_grid.py`,
  `tests/eval/test_elemental_grid.py` and the `legs` hunk; resolve by keeping one copy.
- Training restricted to UR/GW with the existing `cross-balanced` per-stream
  strata. The native uniform sampler admits only the full ten-deck roster;
  extending it was judged larger than the brief's "compatibility change".
- MTG-181 matched-update comparison plays only its two UR/GW cells (100
  games each) at updates 200/400/600, not full grids: the brief asks for the
  same cells at matched updates; full grids would need ~7 extra hours of a
  loaded host. MTG-181's full endpoint grids come from its addendum.
- The heuristic's source digest changed with the `arena`→`sim` move (one
  import line), so per-game player seeds differ from MTG-181's addendum; deals
  and seats stay paired.
- Etude's stock RunPod image has no Rust; `TrainingJob.job` now installs
  rustup 1.98.1 and syncs in setup (the retired bootstrap's toolchain).
- Seed 18601 chosen to follow MTG-181's 1810x numbering.
