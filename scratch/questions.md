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
- One Etude segment (w3) to the remaining cap rather than several ~12 h ones:
  fewer stream restarts and setup/gate costs, no Etude-output→input chaining.
  A failure keeps every export tracked up to it.
- Follower reads w3's checkpoints with `etude.train.read.read(source, "self-play", run)`:
  the tracked experiment is the stage id (`RunContext` name). The tracking
  storage URI is an explicit CLI argument; not verified against a live run.
- Staged-object reuse compares digests by downloading the object (simple, small files).

# MTG-183 Etude eval and R2 (2026-10-25)

- #362 and heuristic v2 are on main and merged here (merge, so `8e7df92e4`
  stays an ancestor). Publication still waits for `elemental-183-w3.json`
  (after w2 ends 01:47 UTC).
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
  `manabot/arena/match.py`, so its checkpoint agent identities change. No w3
  evaluation rows existed yet, so nothing has to be re-admitted. The frozen
  `experiments/protocols/elemental-183*` files keep the old paths as evidence.
- `etude-sdk` follows main's 0.2.2 (the steer asked for 0.2.1, which is older than main's).

# Card evaluation design — 2026-10-23

- Interactive correction from Jack Heart: MTG-179 already accepted retiring the
  mixed `arena/` name because it conflicts with MTG Arena. Its detailed split was
  deferred, not rejected. Reusable play/replay/policies belong in `sim/`, evaluation
  and card statistics in `eval/`, checkpoint followers in `experiment/`. This
  change missed that direction; the working design now records the required
  consumer cutover and removal. The review itself moved no code; the subsequent explicit migration request
  authorized and completed the direct cutover. `manabot/arena/` is removed,
  simulation imports no higher layers, and frozen serialized identities survive.

- Jack Heart selected card ranking by GIH as the first result, with future
  17Lands-style API expansion and fidelity in the data-processing path.
- Implementation design: `scratch/17lands-for-bots.md`; investigation:
  `scratch/17lands-for-bots-evidence.md`. Jack Heart subsequently requested the
  pursuit after review; the dated implementation defaults below settle engineering choices.
- Engineering defaults: CLI/SQL first, one subject model
  and explicit evaluation scope, existing authored deck layouts, true draws
  excluded and reported separately, no implicit minimum-sample filter.
- Arena compatibility initially means a versioned, offline-tested log subset
  parsed by the pinned public 17Lands client. Full downstream metric parity
  remains unproved without reference fixtures; copying parser hand sightings
  directly into GIH would be incorrect.
- Native cards need a verified mapping to Arena printing IDs for export.
  Missing mappings cannot be invented. Rankings use content-bound native IDs.
- No Wave or Task was named; standalone design needs no placement lookup.

Prior MTG-176 recovery assumptions, sync provenance and remote-proof limits
remain at `003644bda6d3ed2e1e6754186960b135ff815e76:scratch/questions.md`.
They do not select current work or authorize paid execution.

## GIH implementation defaults — 2026-10-23

- Jack Heart's launch request supersedes the earlier design-only note. The
  keystone keeps CLI/SQL, copy-weighted draws, and the existing eight-leg authored
  matrices; no later metrics, API, UI, paid execution or external submissions.
- Installed `etude-sdk[eval]==0.2.0` supports the inspected evaluation APIs.
  Its delivery only includes metrics/checkpoints/tables/matches, not arbitrary
  artifacts. Evidence therefore lives under the evaluation's `evidence/`;
  explicit evaluation publication copies and verifies these immutable bytes via
  Etude Storage before delivering Etude's indexes. Plain `etude sync` alone is
  not a card-evidence publication command. This fills a concrete SDK gap without
  another match scheduler, database, or evidence store.
- Use the existing `evaluation` extra in commands. Confidence intervals resample
  whole deal seeds across all selected pairings. Native capture is private,
  setup-enabled and absent from clones. No rule/world version change is intended.
- The pinned public client reports mulligan counts of -1 when no mulligan prompt
  exists. Preserve that unavailable value; do not invent a mulligan decision.
  Exported logs document this bounded subset and cannot certify server metrics.
- Scheduled-admission default (2026-10-23): freeze the native content digest
  alongside world, engine source and content-manifest identities in the existing
  unreleased Etude environment. Missing identity fails admission; never infer it
  from the reader's runtime or rewrite historical evidence. Deck/seat validation
  uses the existing authored matrix, including complete deck and sideboard setup.
