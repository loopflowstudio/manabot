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

# MTG-183 publication held (2026-10-24 19:25 UTC)

- Not published: the branch carries #362's whole unmerged diff (#362 still open,
  head 38ddc7153, merged here at 05ac39fa4), and `elemental-183-w3.json` cannot be
  committed until w2 ends (2026-10-25 01:47 UTC). A ready PR against main now
  would be a dependent PR that could land #362 ahead of its own review. Publish
  after #362 is on main and w3's spec is committed (plan steps 1–2).
