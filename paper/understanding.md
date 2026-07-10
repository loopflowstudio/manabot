# What we actually know (and why)

The living document the Pacing rule gates on. One section per closed cycle:
*what was measured, why we believe the mechanism, what would change our
mind* — written for the owner at a whiteboard, not for reviewers. Terse is
fine; un-derivable is not. `main.tex` is a frozen v0.1 snapshot for eventual
publication; this file is the project's brain.

> Status: skeleton. Each closed cycle below needs its section written to the
> whiteboard bar before the next cycle launches (Pacing rule). Sections
> marked ☐ are integration debt.

## ☐ C0/C0.5 — the instrument (exp-00, 00c)
## ☐ C1/C2 — shaping was the disease (exp-01, 04)
## ☐ C3 — free intelligence (exp-02)
## ☐ C4 — distillation beats RL per dollar (exp-03)
## ☐ C6a — world growth benign (exp-06)
## ☐ C7 — the crank does not compound by default (exp-07)
## ☐ C9 — the pilot cannot play control (exp-09)
## ☐ C10 — the value gate (exp-10, pending merge)
## ☐ C8 — the opponent installs the strategy (exp-11, pending merge)
## ☐ exp-08/08b — matchup tables are pilot+card confounded

---

## Findings of 2026-07-10 (world w2 unless noted)

**F1 — Single-deal evaluation bug (instrument).** In worlds w0–w1, the
Python wrapper `manabot/env/env.py Env.reset(seed=)` accepted a seed but did
not propagate it to the Rust engine. Every evaluation that re-seeded via
`reset` therefore played all of its games on one deal (one shuffle, one pair
of opening hands). An N-game evaluation through that path measures N
stochastic rollouts of a single deal, not N games. Fixed 2026-07-10
(PR #51); source: `experiments/exp-06-newworld-training.md`. Evaluation
lineages that construct a fresh engine per game (the flat-MC matchup driver)
or step the Rust VectorEnv across auto-resets were not affected.

**F2 — Seat parity under random play (correction).** Deal-averaged
random-vs-random mirrors are at seat parity: on-the-play win rate 49.9%
(interactive deck) and 49.8% (standard deck), n≥1000 each
(`experiments/repro/data/repro_06_seat_parity.json`). The previously
recorded values (93.4% on-play standard; 23.1% on-play interactive,
"seat-advantage inversion") were single-deal artifacts per F1. All
pre-2026-07-10 per-seat claims are void unless re-derived deal-averaged.

**F3 — Retraction: no competence-emergent seat advantage established.** A
measurement of 68.9% seat-0 win rate in search-256 self-play (exp-10 datagen)
was traced to a training corpus containing 3 distinct deals across 225 games.
Deal-diverse data shows no clean seat pattern. The hypothesis "seat advantage
emerges with playing competence" is unsupported; it was briefly asserted on
2026-07-10 before the deal check completed, and is retracted.
Source: `experiments/exp-10-value-gate.md` (Deviation 3, Caveats).

**F4 — Representation growth is benign for training (exp-06/C6a).** Growing
the observation/action encoding from world w0 to w1 (card features 29→37,
action types 7→14, mid-resolution decision kinds added) does not degrade PPO
learning on the same game: 3 seeds of terminal-only reward on
INTERACTIVE_DECK score 69.3/77.3/73.5% deal-averaged seat-balanced vs a
uniform-random opponent, within or above the w0 reference band (60.0–75.5%).
Params +1.1%, training throughput +22% on the same recipe.

**F5 — The opponent installs the strategy (exp-11/C8).** With reward held
fixed (terminal-only) and only the training opponent varied, the learned
behavioral profile tracks the opponent: against a uniform-random opponent,
policies drift passive (passed_when_able up to 0.84, 283-step games);
against a frozen strong opponent, policies are forced into near-total
aggression (cast_when_able 0.86–0.89); in self-play, intermediate profiles.
Self-play produced the strongest policies on both vs-random and
search-ladder metrics among the three conditions (2 seeds/arm).
Source: `experiments/exp-11-curriculum-exploitability.md`.

**F6 — The distilled student is robust to a matched-budget exploiter
(exp-11/C8).** A from-scratch PPO exploiter trained specifically against the
frozen distilled policy `student_r0` (87% vs random, w1-ported), at the same
budget that produced all historical trained policies (262k steps), reaches
only 23.5–26.0% against it. At matched cost, the student is not farmable;
its search-ladder placement is not inflated by exploitable habits. Bound is
budget-relative: stronger exploiters remain untested.

**F7 — Search improves on even a near-broken value function (exp-10/C10,
the goal-4 gate).** Flat determinized search using a learned value head at
depth-0 leaves beats greedy one-step argmax over the same value head
60.25% [55.4, 64.9] (400 deal-diverse games) — despite the value head having
been trained on 3 distinct deals (F3). Search-over-V is a policy improvement
operator; the gate's pre-registered >55% bar is cleared with the full CI.

**F8 — A scalar value head is not a rollout substitute at this scale
(exp-10/C10).** The same value head fails as a leaf evaluator relative to
playing games out with uniform-random actions: value-search-64 loses 76.5%
of games to random-rollout search-64 at equal simulation count, and loses
61.5% even to random-rollout search-16 at approximate wall-clock parity
(per-leaf evaluation is only ~2.8x faster than a full random playout on this
engine, not the theorized ~40x). Value quality deal-diverse: Spearman 0.485
vs rollout ground truth, with ordering near noise (0.12–0.32) in undecided
mid-game states. Conclusion: at 100k parameters and this data scale, replace
neither rollouts nor the teacher with scalar V(observation); the indicated
next step is diagnosis (the 2x2 information-x-continuation ablation), not a
value-guided distillation loop.

**F9 — Reproduction suite, first certification pass (partial).** Of six
load-bearing claims converted to end-to-end reproduction scripts
(`experiments/repro/`), four have run and passed deal-averaged in w2:
search strength without learning (88.3/97.7/98.7% vs random at N=16/64/256,
monotone; 256 over 16 head-to-head 80.7%); control incompetence of the
search pilot (max scenario score 0.34 of a known-correct line; random
outperforms search on the two hold-scenarios); batched-inference throughput;
seat parity (F2). Two remain in flight (reward-shaping comparison;
distillation vs matched-cost PPO). A claim without a passing reproduction
script is provisional.
