# Intelligence evidence contracts

These are the existing R1–R4 and standing evidence requirements moved from
`GOAL.md` during the belief-forming branch reconciliation. Thresholds and claim
boundaries are unchanged. R1–R4 retain the historical results-first contract;
they do not prescribe the order of all agent implementation. Current measured
results and limitations belong in `../MEMORY.md`; Project KR state is in Linear.

## Results ladder and standing requirements

The first four are the 2026-07-18 results ladder
([docs/plans/results-first-roadmap.md](../../../docs/plans/results-first-roadmap.md));
each closes only with a committed, replayable artifact on an instrument that
already exists.

- **R1 — the flip.** A frozen, exactly replayable fixture in which changing
  only the typed condition (e.g. `Has(Counterspell)` vs `Lacks(Counterspell)`)
  flips the advised top action under paired seeds at a declared offline
  budget, served through the versioned advice comparison. A measured no-flip
  at ~64× the fixture budget is an acceptable closure with the negative
  result retained.
- **R2 — live beliefs.** One full game through `./scripts/play` can request
  advice mid-game whose belief receipt hashes the live exact-Bayes posterior
  tracked from the compatible-deal prior — no static authored payload on the
  path, same fail-closed identity discipline as the fixture path.
- **R3 — calibration curves.** A committed repro script emits, for seeded
  games, posterior mass on the opponent's actual hidden hand versus the
  prior's mass across the decision sequence, with the action-likelihood model
  byte-locked to a real checkpoint and no remaining `RulesProviderGap` on the
  exercised path.
- **R4 — first strength results.** A committed arena run (ratings, payoff
  matrix, paired-deal uncertainty) over the frozen anchors, the dPUCT
  challenger, and the exact-range player versus a uniform-determinization
  control at matched compute; and one production visit-teacher iteration with
  multi-seed students through the fail-closed harness.

The standing target-state measures:

- Every primary Project produces a runnable manabot, teacher, search system, or
  training loop that executes against real `managym` positions and can be
  exercised with one documented command.
- Search teachers and students are compared in actual selected matchups at
  explicit compute budgets, with legality, competencies, seat-balanced
  strength, calibration, latency, throughput, label cost, and uncertainty.
- One historical/root Observation can be evaluated under the compatible-deal
  prior and typed conditions such as `Has(Bolt)` and `NoLands`, returning
  aligned complete action distributions, values, condition mass, uncertainty,
  and exact provenance without exposing actual hidden truth.
- Advice for one decision is identity-pinned to its viewer-safe belief,
  advisor, planner/evaluator, compute class, seed plan, and evidence bytes. The
  same identity is reproducible through live play and Study, while a mismatch
  returns typed unavailability rather than adapted or invented evidence.
- A supervised belief head maps lossless viewer history to a calibrated
  normalized distribution over managym's world hypotheses. Both policy and
  value are conditioned on that `BeliefState`; actual hidden worlds remain
  calibration targets rather than inference inputs.
- Conditional teacher trajectories, shards, and checkpoints bind world/query,
  belief, history, target, source, seed, and exact byte identities and replay
  through the same semantic Commands as live play.
- Every admitted candidate enters a versioned, world-pinned skill arena. The
  primary hill-climbing signal is a population Elo rating at a declared
  compute class, reported with paired-deal uncertainty and the underlying
  matchup matrix; ratings never cross world or arena-version boundaries.
- A semantic policy consumes viewer-safe runtime facts, typed ability programs,
  and structured legal offers, emits atomic `Command` values, and is evaluated
  on real play—including held-out cards or compositions of known operations.
- Ablations remove card identity, semantic structure, structured decoding, or
  search at the boundary of a working prototype so their effects on learning,
  transfer, strength, and systems cost are directly measurable.
- Policy, search, robustness, and uncertainty evidence can be replayed through
  the versioned Study contract without hidden-information leakage or invented
  client-side meaning.
- Any superhuman claim names the matchup and content boundary, information
  boundary, model and opponent cohort, compute budget, seeds, competencies,
  exploitability evidence where available, and uncertainty.
