# MTG-186 questions and assumptions

Earlier scratch notes (card evaluation/GIH design, MTG-174/176 history) were
inherited from landed work and are at `1e6015270:scratch/`.

- 2026-10-25 compress: `experiments/elemental_grid.py` and its test are back to
  main's bytes. The `cells` subset served only the retired heuristic grids, and
  the head-to-head path uses main's `_legs`/`Group`/`GameRow` unchanged. Like
  `job.py`, the file no longer matches `mtg186-identity.json`, which stays as
  frozen evidence of the launched training source (2de27b5c9).
- Training restricted to UR/GW with the existing `cross-balanced` per-stream
  strata. The native uniform sampler admits only the full ten-deck roster;
  extending it was judged larger than the brief's "compatibility change".
- Heuristic-grid notes (matched-update cells, heuristic digest, u0 replay)
  are obsolete after the head-to-head steer; see git history before 5b89224b6.
- Etude's stock RunPod image has no Rust; `TrainingJob.job` now installs
  rustup 1.98.1 and syncs in setup (the retired bootstrap's toolchain).
- Seed 18601 chosen to follow MTG-181's 1810x numbering.
- 2026-10-25: retry on new on-demand `l4-etude` (Etude stock image with
  python3.12) instead of editing the shared `l4` record, which other sessions
  may use. Its old image cannot boot Etude 0.2.2 pods; Jack Heart may want `l4` fixed.
- Pulling two ~28 GB RunPod images for rehearsal filled this host's disk
  (133 MiB free) and killed the follower mid-group (no failed record written);
  images removed. Do not rehearse with full images on this host again.
# Inherited notes

MTG-174/176, MTG-183 and card-evaluation (#360) notes landed on main and are
retained at `884249024948:scratch/questions.md`. They do not select current work.

# Remote evaluation (2026-10-25)

- Superseded by Jack Heart's head-to-head steer: no heuristic grids at all.
- Head-to-head design choices left open by the steer: the four UR/GW
  combinations share 250 deals; the rest block draws 500 ordered pairs with
  replacement and plays each deal on the play and on the draw (exact balance).
- Hand-written Etude Job (not an Etude eval Benchmark); results are JSON
  outputs, not statsdb tables.
