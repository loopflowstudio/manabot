# Inherited notes

## MTG-269 analysis choices — 2026-11-09

- Keep the owning v1 report/summary paths and add a distinct extension section.
  V2 snapshot layout uses `games/` for runner output and `reconciliation/` for
  dated receipts. Missing paths without receipts are errors, not zero games.
- Use each contrast's complete whole-deal vectors for its nominal and stratified
  interval, retaining all planned slots in missing-evidence sensitivity. This
  applies the existing uncertainty method to four-cell interactions; it does
  not change the schedule, decision rule or launch authority.

## MTG-269 execution choices — 2026-11-09

- Latest schedule: fixed beta 0.01, depth 20, oracle/uniform × 4/8/16 worlds,
  raw sampled policy opponent only, 500 matched deals per cell (3,000 total).
  Earlier 5,000- and 2,500-game schedules were superseded before any new games.
- Keep the existing SDK/game environment frozen. Select the already installed
  schema-13 Etude source only for launcher reads; no DB migration or reset.
- An unresolved allocation for other work is not free capacity even when SSH
  shows no game process. Preserve it and stop execution pending its owner's
  reconciliation. Do not create a second pool/machine name or launch around it.
- For the inclusive seven-day ceiling, use this step's 18:16:29 UTC start on
  November 9, ending November 16 at 18:16:29 UTC (1794852989). Share this absolute
  deadline between smoke and campaign. Delay does not replenish the allowance.
- Store full private search/Command evidence in per-game compressed attempt
  streams. Preserve incomplete attempts and require explicit reconciliation;
  do not call them losses or automatically replay them. Keep v1 evidence intact.

MTG-174/MTG-176 preflight, payment-repair and recovery assumptions are retained at
`003644bda6d3ed2e1e6754186960b135ff815e76:scratch/{questions,elemental-baseline-preflight,elemental-mana-repair}.md`.
They do not select current work or authorize paid execution.

# MTG-269 design assumptions (2026-11-08)

Execution reconciliation (2026-11-09): use the established analysis environment
at `/Users/jack/src/manabot.search` through `uv run --project ... --no-sync`,
importing source from this frozen checkout. Publish final game-file pointers in
the existing store and preserve prior content hashes in a reconciliation receipt.
Leave the 4,899-game HTML/summary intact until the following analysis Graph step
can make the necessary report corrections together. No environment upgrade,
engine rebuild, launcher DB migration or new execution is needed.

- The existing runner is the retained campaign's authority for settings and
  schedule; `experiments/protocols/search-mirror-v1.md` reconstructs it without
  retroactive preregistration. The original source copy lacks a complete launch
  inventory, so current disk hashes cannot prove what every worker loaded.
- Simpler proposed follow-up: reuse the 500 existing deals per opponent and
  existing controls, label selection exploratory, add only beta 0.01 at 8/16
  oracle shuffles and uniform comparisons at those settings and the four-world
  reference. The 5,000-game / 96-hour / 672-worker-hour / $0 ceilings are design
  choices for review, not numbers supplied or approved by Jack Heart.
- No new training seed or fitting: this is one frozen trained policy. New
  shuffle samples and noise resamples cannot substitute for independent learners.
- The current directive's interactive-review boundary takes precedence over
  interpreting the broad continuation request as authority to launch follow-ups.
  Leave the six live mini workers alone; worker 2 has completed its assigned keys.
- MTG-268 is idle, not still implementing. Its recorded unmet latency target
  and unresolved w13 admission remain dependency limits; this step does not
  restart it or choose a relaxed performance acceptance threshold.

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

# MTG-269 analysis assumptions (2026-11-09)

Use seed 26901 for whole-source-deal noise bootstrap as well as the protocol’s
stratified game bootstrap; only the game bootstrap seed was prescribed. Keep
noise position weighting unchanged, resampling source deals jointly across
opponents and retaining all their positions/draws. PUCT comparison families use
their own common-complete subsets across all eight mirror settings so missing
PUCT evidence does not discard complete sampled-policy comparisons.

## Mirror mini execution — 2026-11-09

Direction: “ok so lets use it” and “clear the curriculum stuff”. The canceled
`curriculum-held-out-707e62cad5fe-r3` had no remote boot claim, receipt, or worker.
Etude's SSH bootstrap returned no end receipt for that absent startup, while
its scheduler waited indefinitely for a creation receipt. A process-local
transport correction lets the existing DirectoryJournal close a session without
both receipt and boot claim; sessions with a boot claim retain uncertainty.
Under Etude's mini lock, its `_end_session` recorded shutdown and cancellation.
No database rows were deleted, environment upgraded, or other pool operated.
The exact repair and retained diagnostics are in the v2 reconciliation receipts.

Reversible operational choices: stage committed source `0063d781` in mini
`~/src/manabot.w13-mirror-v2`, preserve the original source directory, reuse its
frozen environment and exact native bytes. Smoke means six separate smoke-key
games (one per factorial cell), three one-thread workers, oracle then uniform,
at most two hours inside the unchanged absolute campaign deadline. No strength
claim follows from that tiny admission. Campaign stays seven workers/3,000 games.
