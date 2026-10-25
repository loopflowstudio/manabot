# Shared evaluation follow-up — 2026-10-24

Accepted by Jack Heart, 2026-10-24:

- After interruption at 40/100 complete valid deals, explicit continuation runs
  the missing 60 under one logical evaluation. Keep all attempts and count each
  block once. Fresh full repeats remain distinct. No automatic retries, native
  state recovery or mixing CPU/CUDA/numerical contracts.
- Local small-checkpoint versus heuristic checks share evaluation semantics
  without training, scheduling, remote credentials or callback transport.
- Keep compute lifetime and accounting internal; no public allocation creation
  is required. Checkpoint pairs at growing model sizes are the primary workload.
- Aim for at least two useful hours of eligible queued work before starting a
  cloud eval server, estimated for its measured host/model mix. Release idle
  compute when work finishes. This is no minimum billed lifetime. No urgent
  override or maximum queue wait has been selected.

Implementation assumptions, reversible and scoped to the supplied checkout:

- Preserve the initial attempt in the existing panel directory; subsequent
  attempts live beneath `attempts/<sequence>/`. `evaluation_status` derives
  progress from immutable block receipts and retained attempts. A UUID separates
  fresh evaluations. No second queue or progress database.
- A filesystem lock fences the logical writer. Managed RPC execution holds an
  additional attempt claim; continuation seals old attempts against delayed RPCs.
  These claims are for a shared local filesystem, not cross-host object storage.
  Moving evidence requires moving the entire evaluation directory; checkpoint
  path changes require the same digest and byte length.
- The same complete-block validator owns final statistics and continuation.
  Incomplete blocks retry whole; conflicting duplicates fail closed. Deadline
  and instrumentation may change; compute settings and scientific limits may not.
  Frozen pre-continuation results stay readable but are not silently migrated.
- Per-block replay and immutable shards are published before another block can
  erase progress. The aggregate trace remains diagnostic/calibration evidence;
  reports use row-referenced shards to avoid counting both copies.

Upstream gap rechecked on installed `etude-sdk==0.2.4`: job/cohort rental ceilings
still are not measured useful-work estimates. Etude's launch scheduler owns the
missing >=7200 useful-second admission gate for new rentals; Manabot needs
host/model workload measurements for integration. Existing idle-session release
remains correct. The design records the inspected symbols and exact gap. This is
still-required compute-policy work, separate from the continuation deliverable;
no alternate durable queue or provider helper is selected.

Unselected: actual model mix, measured current evaluation throughput, memory,
host, budget, maximum wait, latency and numerical/performance thresholds. A few
experiments per week suggests on-demand as a hypothesis, not a purchase.
Historical hardware rates, 3–5 minute startup and probe012's roughly 28 games/min
are illustrations, never runtime estimates for this policy. Whole-job cost,
CUDA parity and an end-to-end gain remain unmeasured. Broad suites belong to gate;
this implementation uses focused headless fixtures only.

The published runtime/report foundation is PR #370 at
`79506de9cae0182653e251767d35ffba5427c066`. Earlier uncommitted questions are
preserved in Git blob `d8c8916accba563add0badbfd2cf30970958ca91`, design in
`1dcb52e5bbdf3747342bc89b95d0cb17e519cd40`, and evidence in
`7bae0e735e83f08273f093dbea10422ed6f416db`. Inherited MTG-174/176/180/182
assumptions remain at `07a091bca:scratch/questions.md`; upstream MTG-183/card
history remains at `884249024:scratch/questions.md`. None selects work here.

Concurrent report demo edits and `scratch/eval-perf-demo.md` belong to the
ongoing conversation's report work and are preserved, not reimplemented here.

# Upstream historical assumptions

These inherited notes preserve context; they do not select current work.

# Inherited notes

MTG-174/MTG-176 preflight, payment-repair and recovery assumptions are retained at
`003644bda6d3ed2e1e6754186960b135ff815e76:scratch/{questions,elemental-baseline-preflight,elemental-mana-repair}.md`.
They do not select current work or authorize paid execution.

# MTG-181 assumptions (2026-10-23)

- Evaluation follows Jack Heart's 10×10 heuristic grid steer; the earlier
  MTG-174 per-pair/initialization/greedy cohorts are not run. Play/draw per game
  comes from one RNG seed (18190001), so seats are not exactly balanced
  (2,452 play / 2,548 draw); play and draw are reported separately.
- Cadence 200 updates, ten local CPU workers, chosen from a local full-grid
  calibration (~45–55 min per checkpoint). No paid evaluation.
- Budget split: three concurrent 5.0-hour L4 jobs at ≤$0.62/h ($3.10 each) plus
  guardian proofs; $0.64 left for deletion uncertainty. A failed seed is not
  replaced.
- The pods' own Etude training runs plus locally rebuilt Etude grid runs form
  the "linked Etude training report"; SDK 0.2.0 lacks the historical
  `etude.reports.Evaluation` link API used by MTG-174's isolated owner, so the
  link is the per-seed run config (job, TrainingRun) and the shared report page.
- The world moved from w10/elemental-world-v1 to w13/elemental-world-v2 since
  MTG-174; that is the re-pinned post-gate world, not a recipe choice.

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

# Sync resolution — 2026-10-25

- Accept main's retirement of `experiments/report_prototype/`; branch report-demo edits remain in the pre-sync Git history. Runtime evaluation and performance work remains intact.
- Keep shared inference and incremental replay while applying upstream checkpoint resolution and successful-game journal cleanup in the shared supervisor.
