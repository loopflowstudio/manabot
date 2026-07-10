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

## Status

RUNNING — remaining sections filled in as phases complete.
