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
- 2026-10-25: retry on new on-demand `l4-etude` (Etude stock image with
  python3.12) instead of editing the shared `l4` record, which other sessions
  may use. Its old image cannot boot Etude 0.2.2 pods; Jack Heart may want `l4` fixed.
- Pulling two ~28 GB RunPod images for rehearsal filled this host's disk
  (133 MiB free) and killed the follower mid-group (no failed record written);
  images removed. Do not rehearse with full images on this host again.
- 2026-10-24 sync onto main (#368/#369): kept this branch's `experiments/elemental_grid.py`
  byte-for-byte (its `cells` subset; main's unused `games` subset dropped), so its
  pinned MTG-186 digest still matches. `manabot/experiment/job.py` now also carries
  main's `LearnerContinuation`, so it no longer matches `mtg186-identity.json`.
  That record pins the training source (2de27b5c9, already launched); the
  follower does not check it, and the merge leaves the evaluation path
  unchanged, so follower restarts from HEAD are acceptable. The frozen
  identity record stays as evidence.

# Inherited notes

MTG-174/176, MTG-183 and card-evaluation (#360) notes landed on main and are
retained at `884249024948:scratch/questions.md`. They do not select current work.

# MTG-186 u0 replay (2026-10-25)

- Assumed that the protocol's no-replay rule does not cover groups that never
  started a game. The 181 u0 groups that hit a SyntaxError (mid-merge checkout)
  are being played again, and their failures are kept as evidence. The
  alternative was to report u0 as 96% invalid. See the protocol's evaluation note.
