# Exp-07 — Expert Iteration (wave/intelligence C7)

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

## Task 2 — Policy rollouts inside search (P2)

### Mechanism

`Env.rollout_pool(worlds, rollouts, seed, max_steps)` (Rust) clones the
current decision into worlds x rollouts simulations per legal action —
worlds determinized from the deciding player and shared across actions
(common random numbers), root actions pre-applied. `encode_active()` /
`step_active(actions)` expose every still-running simulation's encoded
observation in caller-owned buffers, so one batched net forward picks all
rollout actions per ply (`PolicyRolloutMCPlayer`); the sampler plays *both*
seats of the determinized worlds, ε-greedy mixed with random (ε=0.1).
Scoring matches `flat_mc_scores` exactly (1/0/0.5, step cap).

**Hybrid rollouts** (`policy_plies=K`): the policy plays the first K plies
of every simulation — the part adjacent to the root decision, where holding
or spending interaction is decided — then `RolloutPool.finish_random()`
completes the tails engine-side at ~0.2 ms/playout with no encoding. This
is what makes a policy-rollout teacher affordable (below).

### Throughput reality (the honest part)

Measured with the trained R0 student on the shared M4 Max:

- Random-rollout search: 18.3 ms/dec at N=16 (2,905 playouts/s, pure Rust;
  exp-02 measured 7 ms/dec on a quiet box — same code, box load differs).
- Policy rollouts cost ~9–16 ms per rollout ply regardless of batch size:
  the per-forward *dispatch* cost (obs slicing, host→device, kernel launch,
  sample round-trip) dominates at pool-sized batches (40–500 slots), so K
  (plies) is the cost lever and N (sims) is nearly free.
- psearch-16 with K=60: 545 ms/dec, 91 playouts/s, mean 57 plies/playout —
  a **~30x wall-clock premium per decision** over random-rollout search at
  equal sims, i.e. ~160x per playout (0.21 ms → ~34 ms with hybrid tails).
- **MPS does not multiplex across processes.** Four worker processes
  sharing the GPU deliver roughly one process's aggregate throughput
  (measured: a 5th consumer sees ~53 ms/forward at batch ~55 vs ~6 ms
  solo). The task-1 batching wins do NOT compound with process-parallel
  datagen — batching across games within one process is the (future) fix.
- Box load during all task-2/4 measurements: the machine was shared with
  other agents (load average 30–96); absolute ms/dec here are upper bounds.

### Deployed teacher v2 and the budget decision (documented per spec)

At the measured contended rates, any config above ~200 ms/dec cannot
produce 600 self-play games (~87k decisions) inside this cycle's compute
cap. Configs considered: N=32/K=30 (479 ms/dec — 2.9 h floor, rejected),
N=16/K=20 (~550 ms/dec contended, rejected), **N=8/K=8, ε=0.1 (deployed)**
— 8 policy plies ≈ the immediate response window after the root action,
which is exactly where exp-01/02 located the interaction-vs-racing signal.
The rest of each rollout is engine-random. This is the sims/budget the
throughput affords; a quieter box or in-process cross-game batching lifts
it.

### P2 results (deployed teacher: psearch-8, K=8, ε=0.1; 100 games each)

Same-day probes on the quieter evening box: psearch-8/K=8 = **65 ms/dec**
(414 playouts/s, 3.3k net obs/s); random-rollout search-8 = **10.4 ms/dec**
(2,585 playouts/s) → playouts/sec ratio ≈ **1:6.2**; equal-wall-clock
opponent N* = 50.

| matchup | psearch-8 win rate | wall check (hero vs villain ms/dec) |
|---|---|---|
| equal sims: vs search-8 | 56.0% [46.2, 65.3] | 65 vs 10 |
| equal wall-clock: vs search-50 | **31.0%** [22.8, 40.6] | 96 vs 59 |

**P2 verdict: REFUTED.** At equal sims, policy rollouts show a positive but
non-significant edge (56.0%, CI spans 50%) — the mechanism is not dead. At
equal wall-clock — the pre-registered, honest comparison — the random-
rollout searcher's ~6x playout advantage wins decisively (psearch took
31.0%, and note the realized wall was *tilted in psearch's favor*, 96 vs
59 ms/dec, making the refutation conservative). Rollout quality did not
come close to buying back the throughput premium on this deck at this
model size.

## Task 4 — Round 1: closing the loop (P3)

**Teacher v2** = psearch-8 (K=8, ε=0.1, R0-student rollouts) — the budget
choice documented above. **Datagen:** 600 games / 68,204 decisions in
**1,507 s** via the pooled driver (45.2 dec/s — vs ~6 dec/s for the
4-process attempt it replaced; the cross-game batched driver is what made
round 1 possible at all). Seat-0 win 42.3%.

**Retrain: fresh, not fine-tuned** (justified: keeps each round's student a
clean function of its teacher's data; R0 showed 10 epochs from scratch
reaches the quick-eval plateau; warm-starting anchors to R0's blind spots
when the loop's gains should flow through the teacher). Hard targets (per
the R0 sweep). An R0+R1 aggregate variant was also trained to exercise the
staleness diagnostic.

| student | quick 200g vs random | judged 400g vs random |
|---|---|---|
| R0 (search-256 teacher) | 88.0% | **87.0%** [83.3, 89.9] |
| R1 fresh (psearch-8 teacher) | 73.5% | **70.0%** [65.3, 74.3] |
| R1 aggregate (R0+R1 data) | 78.5% | — |

### R1 judge + head-to-head

| matchup | R1 win rate |
|---|---|
| vs random (400g) | 70.0% [65.3, 74.3] |
| vs search-8 (200g) | 28.0% [22.2, 34.6] |
| vs search-16 (200g) | 14.5% [10.3, 20.0] |
| vs search-32 (200g) | 17.0% [12.4, 22.8] |
| vs search-64 (200g) | 10.5% [7.0, 15.5] |
| **vs R0 student (400g, head-to-head)** | **25.8%** [21.7, 30.3] |

R1 ladder ≈ N=3–4 (vs R0's ≈7). Behavior: cast_when_able 0.607,
passed_when_able 0.280 — more aggressive than R0 (0.512/0.377), consistent
with noisier, shallower search labels.

Staleness diagnostic (aggregate variant, rounds {0,1}): final val_loss
0.9941 aggregate vs val_loss_fresh 0.9568 — fresh-round loss *lower*, no
stale-label divergence pattern; the diagnostic is armed and produces the
split as designed.

**P3 verdict: REFUTED.** One crank of the loop *degraded* the student:
25.8% head-to-head vs R0 (needed >55%), ladder ≈4 (needed ≥16). Diagnosis:
the loop's label quality collapsed, not its mechanism. R0 distilled argmax-
of-256-playouts labels; the affordable teacher v2 could only argmax 8
noisy policy-rollout scores per action — a worse label source than the
random-rollout teacher it replaced (P2's wall-clock result is the same
fact seen from the other side). Expert iteration on this engine currently
loses to "spend the same wall-clock on more random playouts."

## Prediction verdicts

| prediction | verdict | number |
|---|---|---|
| **P1** ≥10x net-in-loop (2k → ≥20k obs/sec) | **CONFIRMED** | 24,474 obs/sec = 12.0x |
| **P2** policy rollouts win at equal wall-clock (>55%) | **REFUTED** | 31.0% [22.8, 40.6] (equal sims: 56.0%, n.s.) |
| **P3** R1 beats R0 (>55%) and ladder ≥16 | **REFUTED** | 25.8% [21.7, 30.3]; ladder ≈4 |

## Cost ledger (wall-clock at $1.006/hr, exp-00 accounting)

| item | wall |
|---|---:|
| setup, builds, task-1 infra + benchmarks | ~75 min |
| R0 datagen (600g search-256; incl. a reaped first run, 1 shard salvaged) | ~157 min |
| R0 BC (prelim + full sweeps, 8 configs total) | ~9 min |
| R0 judge (400g + 4 rungs + profile) | ~9 min |
| Task-2 infra (RolloutPool, hybrid tails) + probes + aborted P2 configs | ~45 min |
| R1 datagen (600g psearch-8, pooled driver; incl. 2 aborted process-parallel starts) | ~40 min |
| R1 BC (fresh + aggregate) | ~5 min |
| R1 judge + P2 matchups (concurrent) | ~20 min |
| report, docs, tests, commits | ~35 min |
| **session total (measured at close)** | **~5.2 h ≈ $5.23** (cap 8 h) |

(Phases overlapped with coding/waiting; the itemized rows sum above the
wall because heavy phases ran in the background of authoring work.)

Strength-vs-cost ledger points (cumulative project training spend):
R0 student **87.0% vs random / ladder ≈7 at ~$2.9 of datagen+train wall**
(first new-world point); R1 student 70.0% / ladder ≈4 (negative point,
kept for honesty).

## Caveats

- **The teacher is still deaf and strategy-fused** (exp-03 caveat carries):
  policy rollouts change the rollout prior, not the determinization
  blindness — that is the beliefs wave's problem, untouched here.
- **Teacher v2 was budget-thin** (8 sims, 8 policy plies): P3's refutation
  convicts *this cycle's affordable loop*, not expert iteration in
  principle. The equal-sims edge (56.0%, n.s.) says the mechanism deserves
  one retry at N≥64 policy-sims once per-forward dispatch cost drops
  (in-process batching already demonstrated 45 dec/s at N=8; N=64 needs
  either ~8x that or a smaller/faster net).
- **Shared box**: all absolute ms/dec measured under load from concurrent
  agents (load average 17–96); ratios (policy-vs-random cost, win rates)
  are trustworthy, absolute throughput is a lower bound. The P1 baseline
  was reproduced same-day under the same load (3.8k) for honesty.
- **MPS does not multiplex across processes** — discovered mid-cycle, cost
  two aborted datagen starts, fixed by the pooled driver. Future
  net-in-loop datagen must batch in-process.
- **Vectorized eval finish-order bias**: run_vector_games records games as
  they finish under per-seat quotas; if win correlates with game length
  this mildly biases early-stopped tails. Quotas ≫ streams/seat keep the
  effect small; matchup-harness numbers (ladder) are unaffected.
- Exploitability probe: still not run (deferred since exp-03); every
  ladder number remains provisional against it.
- Judge sampling is stochastic-unseeded per the historical protocol;
  repeat matchups drift ~±1 point.

## Next question (C8)

The loop fails on label economics: 0.21 ms random playouts buy more truth
per dollar than 34 ms policy playouts. Two forks, one experiment each:
(a) **make policy playouts cheap** — distill the student into a ~10k-param
fast rollout net (or quantize/compile) and re-run P2's wall-clock test; or
(b) **abandon rollout-policy improvement and gate on goal 4** — train a
value head on the R0 dataset's outcome labels and test *search-with-V vs
V-greedy* (the pre-registered gate), which sidesteps per-ply inference
entirely at leaf-eval time. Given Exit 2's tripwire (two consecutive
sub-2-point rounds), (b) is the priority: if V-guided search also fails,
the wave exits toward model-free.

## Provenance

- Branch: `exp-07-expert-iteration` off `origin/main` (a3bfab2, includes
  rules stage 2).
- Datasets: `.runs/exp07/dataset_r0{,b}` (600g search-256, 80,568
  decisions), `.runs/exp07/dataset_r1` (600g psearch-8, 68,204 decisions);
  shards carry provenance tags (round, teacher spec, git commit).
- Students: `.runs/exp07/student_r0.pt` (=`student_best.pt`),
  `student_r1.pt`, `aggregate/student_r1_agg.pt`; BC logs `bc_*.json`.
- Judge data: `reports/data/exp-07-expert-iteration.json`; P2:
  `reports/data/exp-07-p2.json`.
- Machine: Apple M4 Max (shared), Python 3.12 isolated venv,
  WANDB_MODE=disabled, torch 2.10 (cpu+mps).
