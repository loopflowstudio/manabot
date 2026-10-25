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
  main's `LearnerContinuation`, so it no longer matches `mtg186-identity.json`;
  run the MTG-186 follower from its pre-merge revision (`2362b48a2`). The
  frozen identity record stays as evidence.

# Inherited notes

MTG-174/MTG-176 preflight, payment-repair and recovery assumptions are retained at
`003644bda6d3ed2e1e6754186960b135ff815e76:scratch/{questions,elemental-baseline-preflight,elemental-mana-repair}.md`.
They do not select current work or authorize paid execution.

# MTG-183 assumptions (2026-10-23)

- A RunPod job cannot exceed twelve hours, so "one long seed" is one cohort of
  chained `last-complete` learner-state continuations (MTG-165's $50 ladder
  pattern). Collector/RNG streams restart at each of six boundaries.
- Controller is this laptop; the Mac mini (`mini-heart.local`) did not resolve.
  Guardians bound every rental, so controller downtime only delays successors.
- MTG-182 not merged: deadline-abandoned games at each window cutoff are read as
  deadline pauses (protocol), not game errors.
- Rated spacing, references (200, 2,000), held-out skip-one pairs, 10 games per
  cell for pairs/subset grids and the saturation rule are this session's choices,
  not Jack Heart's numbers.
- Local evaluation yields while MTG-181's follower runs (Task: do not disturb it).
- The 0.03 practical saturation margin (amendment) is this session's choice.

# MTG-183 Etude continuation (2026-10-24)

- Merged #362's branch instead of rebasing (the steer said rebase): rebasing
  rewrote `8e7df92e4`, the published commit w1/w2 ran from. At landing, tag or
  otherwise keep `8e7df92e4` reachable if the PR is squashed.
- Etude's 43,000 s rental cap forces a chain of one-rental segments (amendment
  02:06 UTC), restoring the frozen w1–w7 shape. Segment hours subtract Etude's
  0.25 h setup allowance (its dollar check), not in the 19:00 formula.
- Sizing w2's rate by its stage wall seconds (103.55/h), not active seconds
  (108/h): conservative, matches the watchdog's clock.
- The evaluator's `--etude-source/--run` names one run; later segments need it
  to read several runs (or one restart per segment). Not built yet.
- Follower reads w3's checkpoints with `etude.train.read.read(source, "self-play", run)`:
  the tracked experiment is the stage id (`RunContext` name). The tracking
  storage URI is an explicit CLI argument; not verified against a live run.
- Staged-object reuse compares digests by downloading the object (simple, small files).

# MTG-183 Etude eval and R2 (2026-10-25)

- #362 and heuristic v2 are on main and merged here (merge, so `8e7df92e4`
  stays an ancestor).
- Explicit `evaluate` matches, not a `Benchmark`: the design is non-uniform
  (full vs subset grids, a pairing graph over w1/w2 checkpoints Etude never
  tracked), and `curves` cannot place non-Etude checkpoints on training axes.
- One Etude match per grid row (20 groups, ~500 or ~100 games): 10 R2 match
  deliveries per unit instead of 200; a restart loses at most the in-flight rows.
- Eval tables at `r2://manabot/evaluations/` (run folder `mtg183-eval/`): my
  choice of prefix. `state.json` (checkpoints, endpoint) stays local beside the
  run and is published with the report.
- Checkpoint agent code identity = `manabot/model`, `manabot/env`,
  `manabot/arena/match.py`, `experiments/elemental_grid.py`; editing those in
  the evaluator's pinned checkout would start new agents, so pin it.
- Stopped the old follower at 2026-10-25 00:15 UTC instead of after w2 ends:
  it was playing heuristic v1 (not the anchor) against MTG-181's live process
  and no longer admitting exports. w2's tail is admitted by a one-shot `_admit` call.

# MTG-183 launcher and w3 source (2026-10-25)

- Moved the empty schema-1 `~/.etude/launcher.db` aside rather than deleting it.
  Etude storage `r2` = `r2://manabot/jobs` (cannot be changed later): prefix `jobs`
  matches the S3 cleanup's `manabot/jobs/mtg183-*` → R2 `jobs/mtg183-*` mapping.
  On-demand `l4` boots w1/w2's image digest, not Etude's default, so w3's
  machine matches the lineage.
- The amendment says w3's source is "a published commit on main", but main
  squash-merges, so no branch commit reaches main. w3's revision is this
  branch's pushed code commit, tagged `mtg183-w3-source`; the squash lands the
  same code on main. Recorded in `elemental-183-w3.json`.

# MTG-183 sync onto #360 (2026-10-24)

- #360 split `arena/` into `sim/` and `eval/`. The branch's `legs` subset moved
  onto `manabot.eval.match.play_cell`. The evaluator's `INFERENCE_CODE` now names
  `manabot/sim/match.py` and `manabot/eval/match.py` in place of the removed
  `manabot/arena/match.py`, so its checkpoint agent identities change. The
  evaluator had played no rows, so at 2026-10-25 ~01:30 UTC its pinned checkout
  moved from `d9fc58df2` to `9871e3545` (etude-sdk 0.2.2, the merged worktree's
  `_managym` build with #360's card-capture bindings) before any row existed. The frozen
  `experiments/protocols/elemental-183*` files keep the old paths as evidence.
- `etude-sdk` follows main's 0.2.2 (the steer asked for 0.2.1, which is older than main's).

# Card evaluation (#360, landed)

The 17Lands/GIH design, investigation and implementation defaults landed with #360;
they are retained at `1e6015270:scratch/{17lands-for-bots,17lands-for-bots-evidence,questions}.md`.
