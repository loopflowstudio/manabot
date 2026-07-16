# W2-234 Teacher-0: search-supervised policy/value pilot

## Verdict

Proceed to the bounded overnight scale-up. The pilot establishes a working,
legality-checked data → joint policy/value training → checkpoint → gameplay
evaluation path. It does **not** complete W2-234 and it is not evidence that
manabot has MCTS: the present teacher is flat determinized Monte Carlo, its
policy label is a masked softmax of per-action scores, and its saved decision
shards are not yet replayable engine trajectories.

## Controlled question

On one fixed search-generated dataset and identical initialization, split,
optimizer, policy target, and capacity, does adding terminal-outcome value BCE
improve the shared representation without weakening the policy? This isolates
the auxiliary value objective. Semantic-program inputs and structured commands
are intentionally outside the experiment.

## Pilot evidence

The 2026-07-15 preflight used 100 self-play games at search-16, 10 epochs per
arm, and 100 games versus random per arm.

| Signal | Policy only | Joint policy/value |
|---|---:|---:|
| Held-out policy CE, initial → final | 0.9754 → 0.9665 | 0.9754 → 0.9668 |
| Nontrivial top-1, initial → final | 37.0% → 54.0% | 37.0% → 54.4% |
| Held-out value Brier, initial → final | 0.2563 → 0.2536 | 0.2563 → 0.2043 |
| Win rate versus random | 71% | 73% |
| Win rate versus search-16 | 10% (1/10) | 20% (2/10) |

The teacher won 19/20 games versus random (95%; Wilson 95% CI 76.4–99.1%).
The generated dataset contained 11,886 decisions, 2.93 mean encoded legal
actions, and no winner-less rows. Total pilot wall time was 174 seconds.

Every pre-registered substrate gate passed: teacher signal, decreasing policy
CE, policy top-1 above uniform-over-legal, joint value Brier below 0.25, and
joint gameplay no worse than the policy-only arm by more than 10 points. The
student/teacher comparison is deliberately reported but is too small and too
weak to claim teacher recovery.

## What the overnight run can answer

The scale-up increases self-play games and simulations per legal action while
retaining the same two-arm ablation. If validation and gameplay improve while
the teacher gap shrinks, the pilot was substantially data/label-quality
limited. If policy imitation improves but gameplay or the teacher gap does
not, distribution shift and target formulation become the leading branches.
If neither improves, capacity/optimization should be tested before building a
more expensive tree-search teacher.

## Required next experiment

A KataGo-shaped teacher still requires adaptive PUCT/MCTS, per-edge root visit
counts, policy/value-guided leaf evaluation, explicit hidden-information
semantics, and replayable viewer-safe offer/command trajectories. The runner
and learner already accept stable future `visit_counts` and `root_value`
columns so that change can be an experimental teacher substitution rather than
a new training pipeline.
