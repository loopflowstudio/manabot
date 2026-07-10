# Exp-10 — The Goal-4 Gate: search-with-V vs V-greedy (wave/search C10)

Date: 2026-07-09
Machine: Apple M4 Max (shared with one other measurement agent; heavy phases
sequenced, ≤4 worker processes).
Deck: INTERACTIVE_DECK mirror, both seats. Seat-balanced everywhere (A1).
Accounting: exp-00 cost basis, $1.006/hr wall-clock. Cap: 6 h.

## Why this gate, now

Exp-07 refuted the policy-rollout crank on label economics: an affordable
policy-rollout teacher (8 playouts) labels worse than 256 random playouts,
because policy playouts cost ~6x per decision (~160x per playout). The escape
route the wave pre-registered (wave/search/README.md, goal 4) is a VALUE HEAD
at rollout depth 0 — one forward pass replacing an entire playout of signal —
IF search on top of V still improves on V. The gate is not V accuracy:

> **search-with-V beats V-greedy** — that condition, not V's accuracy, is
> what makes search a policy improvement operator.

If the gate fails, Exit 2 (model-free game-theoretic pivot) fires: exp-07
P2/P3 already armed half its tripwire.

## Pre-registered predictions (recorded verbatim, before any run)

> P1: V trained on search-256 self-play outcomes reaches Spearman ≥0.6 vs
> rollout ground truth on held-out states, but with a measurable per-bucket
> bias (optimistic in board-ahead states, pessimistic in
> behind-but-holding-interaction states — the aggro-bias tripwire from the
> wave README).
>
> P2 (THE GATE): search-with-V-at-leaves at N=64 beats V-greedy (argmax over
> one-step V) head-to-head >55% seat-balanced.
>
> P3: value-search at equal wall-clock beats random-rollout search at some N
> (V evaluation ~1 forward vs ~120-step playout — the economics that failed
> for policy ROLLOUTS should work for leaf VALUES; measure the actual speed
> ratio and pick the honest comparison points).

## Deviation from plan: the inherited artifacts were dimensionally dead

The plan was to train V on exp-07's r0/r0b shards (600g search-256 self-play,
80,568 decisions, winner+seat labels present) and judge against the inherited
`student_r0.pt`. Neither is usable on current main: commit `6339bbd`
("observations: Stage-3 agent visibility") landed after exp-07 forked and
changed the observation encoding (player_dim 27→28, permanent_dim 7→11), so
every exp-07 checkpoint and shard is dimensionally dead in the merged world —
the same world-break exp-07 itself documented for pre-stage-2 artifacts.
Feature re-mapping was rejected (inserted, not appended, features; silent
misalignment risk in the experiment that decides the wave's direction).

Fallback (anticipated in the cycle plan): regenerate everything in the
stage-3 world —

1. fresh search-256 (W=64 x R=4) self-play, 600 games, terminal outcome
   recorded per decision (`.runs/exp10/dataset_v0`, 8 provenance-tagged
   shards, 4 engine-only workers);
2. a fresh BC student distilled from those shards with exp-07's selected
   recipe (hard argmax targets, lr 1e-3, 10 epochs) as the policy baseline
   (`student_bc.pt`) — this replaces "student_r0" in every judged matchup
   below and gives V's encoder its warm start.

## Task 1 — Value head

TODO(training)

## Task 2 — Assessing V (wave README protocol)

TODO(assess)

### Per-bucket bias (the aggro-bias tripwire)

TODO(buckets)

## Task 3 — V-greedy

TODO(vgreedy)

## Task 4 — The gate + economics

TODO(gate)

### Economics

TODO(econ)

### Ladder placement

TODO(ladder)

## Prediction verdicts

TODO(verdicts)

## Cost ledger

TODO(cost)

## Caveats

TODO(caveats)

## Next question (C11)

TODO(next)

## Provenance

TODO(provenance)
