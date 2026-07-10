# Exp-07 — Expert Iteration (wave/search C7)

Date: 2026-07-09
Machine: Apple M4 Max (shared with a Rust build agent and a 3-seed training
run during parts of the window; heavy phases were sequenced and capped at 4
worker processes).
Deck: INTERACTIVE_DECK mirror, both seats. Seat-balanced everywhere (A1).
Accounting: exp-00 cost basis, $1.006/hr wall-clock.

## Pre-registered predictions (recorded verbatim, before any run)

> P1: batched actor inference lifts net-in-loop throughput ≥10x (2k → ≥20k
> obs/sec on the 100k-param agent).
>
> P2: at EQUAL WALL-CLOCK per decision, search with policy rollouts beats
> search with random rollouts head-to-head (>55%) on INTERACTIVE_DECK, because
> rollouts stop wasting interaction (the aggro-bias-vector-3 fix).
>
> P3: one full crank of the closed loop (distill → student becomes rollout
> policy → stronger teacher → re-distill) yields a student that beats the
> round-0 student head-to-head (>55%) and places ≥ N=16 on the ladder.

## Task 1 — Batched inference (wave goal 1, P1)

### What changed

1. **Agent hot path** (`manabot/model/agent.py`): every `forward()` was
   paying for ~a dozen eagerly-evaluated f-string debug logs (several forcing
   `.item()` syncs), a `detach().cpu()` copy of raw logits, six per-call
   ownership-mask allocations, a per-call mask-correctness assertion inside
   attention, and a per-sample Python loop in `get_action_and_value`. All
   removed (raw-logit stashing survives behind `agent.debug`).
2. **Generic batched driver** (`manabot/sim/rollout.py`):
   `BatchedSampler` runs one masked forward for any set of rows of the
   zero-copy numpy buffers the Rust envs write into; `run_vector_games`
   drives K parallel streams on `managym.VectorEnv` (`opponent_policy=
   "none"`, both seats surfaced, routed per-seat via the new
   `current_agent_indices()`), giving seat-balanced eval and head-to-head
   with no per-stream Python round trip. Same sampler serves eval, datagen
   and search rollouts (task 2).
3. **Device**: batching is what makes the M4 Max GPU usable — at batch 16
   MPS is pointless, at batch 256+ it is ~6x CPU. The driver takes
   `device="mps"`.

### Numbers (net-in-loop obs/sec, 100k-param default Agent, INTERACTIVE_DECK)

| configuration | obs/sec | inference share |
|---|---:|---|
| baseline, 16 envs (reports/sps-closeout.md, March) | **2,035** | 97.2% |
| baseline reproduced today, 16 envs, pre-change | 3,841 | 95.7% |
| baseline today, 256 envs, pre-change (batch alone) | 6,546 | 94.0% |
| post-fix driver, 16 streams, cpu | 4,297 | 94.2% |
| post-fix driver, 256 streams, cpu | 7,268 | 91.9% |
| post-fix driver, 256 streams, **mps** | 18,782 | 80.0% |
| post-fix driver, 1024 streams, **mps** | **24,474** | 74.8% |

(`scripts/bench_policy_driver.py`; every surfaced decision on every stream is
answered by the net, so obs/sec is directly comparable to the historical
inference-on SPS. Today's box is shared with a 3-seed training run — the
reproduced baseline was measured under the same load as the "after" rows.)

Forward-pass microbenchmark (post-fix, ms/forward): cpu 4.1 @B=16 /
39.7 @B=256 / 145.5 @B=1024 (saturates ~7k obs/s — the model is genuinely
compute-bound on CPU: attention over ~166 objects per observation); mps
6.0 @B=256 / 24.8 @B=1024 (~42k obs/s).

**Where the time goes now** (1024 streams, mps): select 74.8% (17 per-key
host→GPU copies + forward + sampling round trip), Rust env stepping 25.2%.
The env is visible again for the first time since the SPS wave; the next
throughput lever is obs-transfer consolidation, not torch overhead.

**P1 verdict: CONFIRMED.** 24,474 vs the pre-registered 2k baseline =
**12.0x** (≥10x, ≥20k obs/sec). Honest caveats: against the same-day
reproduction of the baseline (3,841) the lift is 6.4x; the 10x needs the
GPU, which only batching unlocks; CPU-only the ceiling is 7.3k (3.6x),
compute-bound, not overhead-bound.

## Task 3 — Round 0: new-world distillation

**Teacher:** search-256 (random rollouts, W=64 x R=4), self-play mirror on
INTERACTIVE_DECK. **Dataset:** 600 games / 80,568 decisions (145 decisions/
game, seat-0 win 55.8% in the r0b tranche), every decision recording the
full encoded observation, the argmax action, and the raw per-action playout
score vector (soft-target source). Shards carry provenance tags (round,
teacher spec, git commit).

**Soft-target design** (exp-05 not merged to main, so registered fresh
here): targets p ∝ exp(score/τ) over valid actions. At N=256 the per-score
Monte Carlo s.e. is ~sqrt(p(1-p)/256) ≈ 0.03, so τ well below ~0.03
amplifies playout noise and τ >> typical score gaps flattens the teacher.
Swept τ on a 150-game preliminary tranche ({hard, 0.02, 0.05, 0.1, 0.2};
0.1/0.2 clearly worse: 71.0%/60.0% vs random) then {hard, 0.02, 0.05} on
the full data, selected by 200-game quick eval + val accuracy:

| targets | quick 200g vs random | val acc (vs teacher argmax) |
|---|---|---|
| **hard argmax (selected)** | **88.0%** [82.8, 91.8] | 0.5333 |
| soft τ=0.02 | 84.0% [78.3, 88.4] | 0.5299 |
| soft τ=0.05 | 80.0% [73.9, 85.0] | 0.5257 |

**Finding: soft targets do not help at N=256 on this deck** — strength is
monotone in target sharpness; the hard argmax control won. (CIs overlap
between hard and τ=0.02; the ordering is consistent across both the
preliminary and full sweeps.) With 256 sims/action the argmax is already
low-noise, and softening mostly injects the teacher's rollout noise floor.
R0 student = hard config, lr 1e-3 x 10 epochs, fresh 100k-param Agent.

### R0 judge (standard protocol: seat-balanced, stochastic sampling, Wilson 95%)

| matchup | R0 win rate | on play | on draw |
|---|---|---|---|
| vs random (400g) | **87.0%** [83.3, 89.9] | 86.5% | 87.5% |
| vs search-8 (200g) | 44.5% [37.8, 51.4] | 50.0% | 39.0% |
| vs search-16 (200g) | 37.0% [30.6, 43.9] | 38.0% | 36.0% |
| vs search-32 (200g) | 33.5% [27.3, 40.3] | 16.0% | 51.0% |
| vs search-64 (200g) | 20.5% [15.5, 26.6] | 21.0% | 20.0% |

**Ladder ≈ N=7 (just under 8;** the search-8 CI includes 50%, the point
estimate does not reach it). Behavioral profile (400g): cast_when_able
0.512, passed_when_able 0.377 — the teacher's patience inherited, no aggro
fingerprint. Note: this is the FIRST trained policy in the post-stage-2
world (new dims); ladder rungs are not directly comparable to exp-03's
old-world N≈8 — the deck rules and observation space both changed.

The 400-game vs-random matchup ran in **9 seconds** on the batched driver
(historically ~10 minutes of process-pool time) — task 1 paying rent
immediately.

## Status

RUNNING — P2 / R1 sections pending.
