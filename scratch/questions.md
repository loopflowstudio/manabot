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
