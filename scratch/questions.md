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
- The evaluator reads several segment runs; `--continues` (operator-declared)
  rather than a budget rule decides whether the last run's end settles the seed,
  since the successor's run id only exists after its launch.
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

# MTG-183 w3 relaunch (2026-10-25 03:51 UTC)

- Read "No segment is retried" as covering segments that learned; launches that
  never took a learning step are relaunched (amendment 03:51 UTC). The literal
  reading would have ended the seed at 2,534 on infrastructure bugs, against
  Jack Heart's direction to continue it within the cap.
- Took MTG-186's three Etude worker fixes verbatim; whichever of MTG-183/MTG-186
  lands second resolves the identical hunks.
- Deleted stuck pod `ublorrai3gb3ea` through RunPod's API: `etude terminate`
  refuses a machine with a run, and the worker would never pick up the attempt
  (Etude `Courier.bring_in` records the new inbox before fetching inputs, so a
  failed fetch never retries). Worth an Etude PR.

# MTG-183 evaluation on the Mac mini (2026-10-25)

- Segment end is the tracking bucket's `state.json` (Etude's controller DB is on
  the laptop). A segment killed without finishing tracking never ends there;
  the operator then names its successor or drops `--continues` by relaunch.
- Doppler service token on the mini (read-only, manabot/prd) is this session's
  choice for R2 keys; Etude gives jobs no storage keys. Revoke `mtg183-eval-mini`
  when evaluation ends.
- No yield to MTG-181 on the mini (Jack Heart: the laptop-only yield no longer
  applies), though MTG-181's evaluation shares the mini's 8 cores; games have a
  600 s bound and take seconds.
- The eval job's `max_attempts=1`, `max_hours=190` (to `--until`).

# MTG-183 w4 restart (2026-10-25 ~21:40 UTC)

- Never-trained launches count at the larger of the earlier provider estimate
  and Etude's receipt ($0.15 vs $0.04 for `a04845f2548b`, $0.44 vs $0.09 for
  `3828021cbc80`, whose pod was deleted outside Etude): conservative under the cap.
- A wall-bounded segment exits 0 (Etude `succeeded`) with `etude-result.json`
  at the update reached, below its `until`; its tracking closes `finished`.
- `TrainingJob.stop_reserve_seconds` defaults to 120 s or a tenth of the
  watchdog, whichever is shorter, so short-watchdog experiment jobs still admit.
- The update target uses 3,250 (w4's origin) in place of "w3's final update".
- The raw-only import accepts an `interrupted` parent only; a `failed` parent is
  refused (not needed for w3; widen deliberately if ever needed).
- Branch and tag not pushed here: the brief stops at a committed spec, and the
  operator session launches.
