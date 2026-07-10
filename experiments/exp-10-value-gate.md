# Exp-10 — The Goal-4 Gate: search-with-V vs V-greedy (wave/search C10)

Date: 2026-07-10
Machine: Apple M4 Max (shared with other agents throughout; load average
100-300 during the morning measurement window — all absolute ms/dec are
upper bounds, ratios and win rates are trustworthy).
Deck: INTERACTIVE_DECK mirror, both seats. Seat-balanced everywhere (A1).
Accounting: exp-00 cost basis, $1.006/hr wall-clock. Cap: 6 h — **exceeded;
see the cost ledger for the honest breakdown.**

## Why this gate, now

Exp-07 refuted the policy-rollout crank on label economics: an affordable
policy-rollout teacher (8 playouts) labels worse than 256 random playouts,
because policy playouts cost ~6x per decision (~160x per playout). The escape
route the wave pre-registered (wave/search/README.md, goal 4) is a VALUE HEAD
at rollout depth 0 — one forward pass replacing an entire playout of signal —
IF search on top of V still improves on V. The gate is not V accuracy:

> **search-with-V beats V-greedy** — that condition, not V's accuracy, is
> what makes search a policy improvement operator.

If the gate had failed, Exit 2 (model-free game-theoretic pivot) would have
fired: exp-07 P2/P3 already armed half its tripwire.

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

## Deviation 1: the inherited artifacts were dimensionally dead

The plan was to train V on exp-07's r0/r0b shards (600g search-256 self-play,
80,568 decisions, winner+seat labels present) and judge against the inherited
`student_r0.pt`. Neither is usable on current main: commit `6339bbd`
("observations: Stage-3 agent visibility") landed after exp-07 forked and
changed the observation encoding (player_dim 27→28, permanent_dim 7→11), so
every exp-07 checkpoint and shard is dimensionally dead in the merged world —
the same world-break exp-07 itself documented for pre-stage-2 artifacts.
Feature re-mapping was rejected (inserted, not appended, features; silent
misalignment risk in the experiment that decides the wave's direction).
Fallback: regenerate in the stage-3 world — fresh search-256 self-play with
outcomes, plus a fresh BC student as the policy baseline and V's warm start.

## Deviation 2: datagen stalled; the cycle closed at 225/600 games

The 600-game regeneration was attempted three times and produced 225 games
(3 of 8 shards, 27,423 decisions, 121.9 decisions/game): the first 4-worker
run was reaped by the session harness at ~45 min with 3 shards durable; the
relaunched run died silently overnight (empty log, no shards written) across
a machine-sleep window; a third, supervised small-shard run was killed by a
coordinating-session intervention because the compute cap was already blown.
Root causes, honestly: (a) long-running background jobs in this harness need
external supervision, which the first two runs lacked; (b) **the completion
monitors watched only for success artifacts (shard files), not for worker
liveness/progress**, so ~10 wall-clock hours passed with 3 shards on disk
and nobody noticing; (c) the box was shared, sometimes at load average 300,
and slept overnight mid-run.

Per the loop's redesign-don't-extend rule, the cycle closed with what was on
disk: **225 games / 27,423 decisions** (seat-0 wins 68.9% of these games —
search-256 self-play is strongly play-advantaged in the stage-3 world; exp-07
measured 55.8% on its tranche). Consequences: V trained on 37% of the
pre-registered data; economics cells at 200 games instead of 400. The gate
itself ran at the full pre-registered 400 games.

## Task 1 — Value head

No new architecture: the Agent already carries a scalar value head over the
shared post-attention encoding (`manabot/model/agent.py`); exp-10 gives it
its first meaningful training signal. `manabot/sim/value.py::train_value`
fits sigmoid(V) to the terminal outcome from the decision-maker's
perspective (BCE, split by game, winner-less rows dropped), warm-started
from the BC student. The value loss has no gradient path to the policy head;
`freeze_encoder` additionally pins the shared encoder. Type-error caveat
(wave/beliefs): scalar V(observation) is rung-1 — the same observation has
different values under different opponent ranges; expected to work at
current opponent strength, not in general.

Baseline first: **BC student** (exp-07 recipe: hard argmax targets, lr 1e-3,
10 epochs) on the same 225 games: val acc 0.5205 vs teacher argmax, and
**86.5% [81.1, 90.6] vs random (200g)** — statistically indistinguishable
from exp-07's 600-game R0 (87.0%), so the 225-game regeneration reproduces
the frontier in the stage-3 world.

Value variants (10 epochs, lr 1e-3, batch 512, val split by game):

| variant | val BCE | val Brier | val acc (outcome) |
|---|---:|---:|---:|
| **full fine-tune from BC init (selected)** | **0.5512** | **0.1762** | **0.759** |
| frozen encoder, value head only | 0.6243 | 0.2162 | 0.666 |

Fitting only the value head on top of BC features costs ~9 points of
held-out outcome accuracy — the policy-distillation encoder does not already
linearly contain the win-probability signal. Train BCE 0.330 vs val 0.551
shows real overfitting room at 225 games; more data was the plan.

## Task 2 — Assessing V (wave README protocol)

830 fresh held-out states from BC-student self-play (disjoint from training
games by construction), ground truth per state = mean over legal actions of
64 fresh uniformly-random playouts per action (`Env.flat_mc_scores(16,4)` —
the wave README's unbiased P(win|s) instrument, ≥64 playouts per action). V
is read at the same state from the acting player's perspective.

| metric | value |
|---|---:|
| **Spearman V ~ rollout gt** | **0.646** |
| Pearson | 0.638 |
| Spearman V ~ actual game outcome | 0.362 |
| mean bias (V − gt) | −0.160 |
| MAE | 0.258 |

**P1's Spearman threshold (≥0.6) is met: 0.646.** Calibration is monotone
but compressed and globally pessimistic — mean V 0.41 vs mean gt 0.57, and
in V's most pessimistic decile (V < 0.1, n=303) the random-rollout truth is
0.41. A large part of this is instrument mismatch rather than pure error: V
estimates P(win | both continue as search-256), gt measures P(win | both
continue uniformly random), and the acting player converts far more of these
states against a random continuation than against a competent one. Ordering,
which is what search consumes, is the meaningful number.

### Per-bucket bias (the aggro-bias tripwire) — reported prominently

Buckets: board advantage sign (sum of P+T on battlefield, mine − theirs) x
holding interaction (Lightning Bolt / Counterspell in hand).

| bucket | n | mean V | mean gt | bias | MAE | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| ahead / holding | 83 | 0.456 | 0.673 | **−0.217** | 0.298 | 0.643 |
| ahead / no-instants | 250 | 0.585 | 0.688 | −0.103 | 0.227 | 0.687 |
| even / holding | 108 | 0.194 | 0.519 | **−0.325** | 0.360 | **0.228** |
| even / no-instants | 134 | 0.397 | 0.581 | −0.185 | 0.263 | 0.588 |
| behind / holding | 107 | 0.345 | 0.473 | −0.128 | 0.275 | 0.525 |
| behind / no-instants | 148 | 0.293 | 0.399 | −0.106 | 0.196 | 0.700 |

The tripwire fires, but not with P1's predicted sign pattern. V is
pessimistic *everywhere* relative to random-rollout truth (no optimism in
board-ahead states), and the bias concentrates in **holding-interaction
buckets** (−0.22/−0.33/−0.13 vs −0.10/−0.19/−0.11 for the matching
no-instant buckets). The sharpest finding is ordering, not level: in
even-board/holding-interaction states — exactly the states where holding vs
spending interaction is the decision — **V's Spearman collapses to 0.23**
against 0.53-0.70 in every other bucket. The value function is
close-to-uninformative precisely where interaction strategy lives. That is
another appearance of the project's structural bias: cheap approximations
cannot represent plans, and a V distilled from determinized-search outcomes
inherits it.

## Task 3 — V-greedy

`VGreedyPlayer`: argmax over legal actions of V at the state after applying
the action on one determinized clone (hero perspective, non-hero-actor
leaves flipped 1−V). Judged (200g each, seat-balanced, Wilson 95%):

| matchup | V-greedy win rate |
|---|---|
| vs random | 69.5% [62.8, 75.5] |
| vs student_bc | **67.5% [60.7, 73.6]** |

V-greedy beats the policy it was warm-started from decisively — one-step
V-argmax already improves on the distilled policy — while scoring *lower*
than that same policy against random (69.5% vs 86.5%). The non-transitivity
is real and instructive: V was fit exclusively on search-256-vs-search-256
states, and against a random opponent it visits states far off that
distribution, where its pessimism misorders actions (consistent with the
task-2 finding that V degrades off-distribution). Cost: 5.4-5.8 ms/decision
on CPU.

## Task 4 — The gate

Config: `ValueSearchPlayer`, N=64 determinized worlds x 1-step-then-V
(depth 0, rollouts 1) — the pre-registered primary. Degeneracy check before
running: at N=64/d0, value-search agrees with V-greedy on only **74.8%** of
1,582 nontrivial decisions (mean score gap when disagreeing 0.085), so
1-step-then-V with world-averaging is a genuinely distinct policy and no
depth extension was needed. What N=64 buys over V-greedy is exactly the
hidden-information average: 64 determinizations of the opponent's hand and
both libraries, with common random numbers across actions, versus V-greedy's
single sample.

**THE GATE (400g, seat-balanced, Wilson 95%):**

| matchup | win rate | on play | on draw |
|---|---|---|---|
| **value-search-64 vs V-greedy** | **57.5% [52.6, 62.3]** | 19.0% | 96.0% |

**GATE VERDICT: PASS.** The wave README's gate condition — search-with-V
beats V-greedy, >50% — is met with the CI clear of 50%. Against P2's
stricter pre-registered bar (>55%), the point estimate clears it (57.5%) but
the interval spans it: **confirmed at the gate bar, marginal at the 55%
bar.** Search on top of this V is a policy improvement operator — weakly but
measurably, and with only 37% of the intended training data behind V.

The per-seat split (19% on the play, 96% on the draw) is violent and gets
its own caveat: the training self-play was itself heavily play-advantaged
(seat-0 wins 68.9%), so V carries strong seat-correlated signal; in a V-vs-V
mirror this polarizes outcomes by seat. Seat balancing keeps the headline
number fair; the mechanism behind the polarization is unmeasured (candidate
C11 diagnostic).

### Economics

TODO(econ)

### Ladder placement

TODO(ladder)

## Prediction verdicts

| prediction | verdict | number |
|---|---|---|
| **P1** Spearman ≥0.6 + per-bucket bias (aggro pattern) | **PARTIALLY CONFIRMED** | 0.646; bias measurable and concentrated in holding-interaction buckets (−0.22/−0.33 vs −0.10/−0.19), but pessimistic everywhere (predicted board-ahead optimism absent); ordering collapses to Spearman 0.23 in even/holding states |
| **P2 (THE GATE)** search-with-V > V-greedy >55% | **CONFIRMED (marginal at 55%)** | 57.5% [52.6, 62.3] over 400g; gate bar (>50%) passed decisively |
| **P3** value-search beats random-rollout search at equal wall-clock at some N | TODO(p3) | TODO(p3num) |

## Cost ledger

TODO(cost)

## Caveats

TODO(caveats)

## Next question (C11)

TODO(next)

## Provenance

TODO(provenance)
