# TLA frontier self-play at two model sizes (c5fba4 v1)

This protocol is fixed before any scoring. The runner is
`experiments/tla_frontier.py`; the bytes are
[the JSON protocol](tla-frontier-c5fba4-v1.json) and the w64 regime
`experiments/regimes/tla-frontier-c5fba4-w64.json`. The w256 regime is that
regime with the protocol's width, depth and heads. `frozen()` rejects a change
to the regime's hash, the runtime world or the derived tensor shapes. A changed
world before launch needs a new protocol version, not edited bytes.

## Question

How far can self-play on the frozen 17Lands TLA trophy population push play
strength at the current recipe size, and does a network about 15× larger in
compute parameters get further on the same budget? This is a frontier push with
one seed per size, not an ablation grid.

## What is trained

Both sizes use MTG-213 v10's recipe unchanged except where listed:

- **Population.** The same 134 frozen whole lists (corpus SHA256
  `21736009…`), sampled independently and uniformly per seat. No held-out-deck
  claim follows.
- **Learning rule.** The damped policy rule MTG-213 inherited from
  MTG-183's Elemental study (λ_policy 0.5, λ_value 0.8, quantile filter,
  action-type-uniform reference, learning rate 0.5 / iteration^1.1 clamped to
  [5e-6, 1e-4], EMA 0.999), 256 streams × 64 transitions, both seats learning.
  The rule's schedule is absolute in iterations, so the update target is a cap,
  not a schedule input. The open review of this rule for full TLA (MTG-213
  comment 81ec3f17) is closed for this protocol by keeping it: the frontier
  question is about scale, and changing the rule would confound the first
  reading of it.
- **Input.** Policy input v9 (v8 layer-state layout plus naming candidates),
  512 local facts, 128 cards and 64 permanents per seat, 64 policy action
  slots with the accepted matched first-64 overflow for every policy.
  Game-local resource recovery as accepted for v10 (25% rolling threshold).
- **Sizes.** Only width, depth and heads differ:

  | Label | Width | Attention layers | Heads | Compute parameters | Total parameters |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | w64 | 64 | 2 | 4 | 647,682 | 1,958,402 |
  | w256 | 256 | 4 | 8 | 9,712,130 | 14,955,010 |

  Compute parameters exclude the five 4096-bucket hashed lookup tables, whose
  bucket count is unchanged; their width follows the model width. The ratio is
  15.0×, inside the requested 10–100× and toward its low end so the larger size
  still completes several hundred updates on its share of the budget.
- **Budget per size.** One GPU allocation of at most USD38 and 56 hours; the
  regime wall budget is 54 hours (194,400 s) and the update cap 3,000. Training
  stops at whichever comes first and exports the last complete raw learner.
  The sizes therefore receive equal wall-clock compute, not equal updates.
  Training seed 21501 for both. No automatic retry.

## Evaluation schedule

Every comparison plays the same 16 fixed pairs (deck seed 21590001, deals
21591000–21591015), four legs per pair swapping decks and seats: 64 games. The
heuristic is the matched first-64 heuristic. Shared update milestones are 25,
50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1600, 2000 and 2400.

- At update 0 and every reached milestone: each size vs the heuristic.
- At every milestone both sizes reached: w256 vs w64 at equal updates.
- At each endpoint: that size vs the heuristic (unless the endpoint is a
  milestone) and vs its own update-0 initialization.
- Once both runs finish: w256 endpoint vs w64 endpoint, the comparison at
  equal wall-clock compute.

Evaluation runs on rented CPU through Etude while training continues, so
progress is visible at each milestone. Each pass is capped at USD2 and six
hours; cumulative evaluation spend is capped at USD12. Milestones past a size's
endpoint stay in the schedule as unreached slots; failed or truncated games
stay failed in their scheduled slots and are never replaced.

## Reading

Scores are the candidate's mean game score with 95% whole-pair bootstrap
intervals (seed 21592001). Head-to-head rows put w256 first.

- **Learning**, per size: endpoint vs its initialization above 0.5 with the
  interval excluding 0.5.
- **Strength**, per size: the milestone curve and endpoint vs the heuristic.
- **Scale**: the equal-compute endpoint head-to-head is primary; equal-update
  head-to-heads describe sample efficiency.

Report every attempt, failure, resource-limit discard, unreached milestone and
cost. One seed per size supports a descriptive frontier reading, not a
controlled claim that one size is better in general.

## Budget

The Task's ceiling is USD100 including setup, failures, retries and evaluation:
2 × USD38 training + USD12 evaluation = USD88, leaving USD12 for setup and
failure costs. Any retry needs a recorded decision that keeps the total within
USD100. Evaluation never runs on a laptop.
