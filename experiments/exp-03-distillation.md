# exp-03: distillation — per dollar, does supervised learning from search beat PPO from scratch?

**Cycle:** C4 (wave/intelligence/01-experiment-loop.md) · **Date:** 2026-07-09 ·
**Deck:** INTERACTIVE_DECK both players ·
**Teacher:** search-64 (flat determinized MC, exp-02: 95.0% vs random, 31 ms/decision) ·
**Student:** fresh default Agent (100,354 params, attention on) ·
**Baseline:** matched-cost PPO (`first_light_shaped_v1`, exp-01 deck rebind, 16 envs).

## Question

Per dollar, does supervised learning from search beat PPO from scratch?

## Pre-registered prediction

**BC-from-search matches or beats matched-cost PPO at ≤1/5 the cost to reach
PPO's final strength.** This is the thesis (goal 5, "search as teacher") in
miniature. Secondary bar, registered in exp-02's next-question: the BC policy
should land in the search-16–64 band — ≥90% vs random, ≥84% vs the C1v2
checkpoints, seat-balanced — else distillation is losing signal that 7 ms of
search finds for free.

## Method

### Task 1 — dataset from the teacher

600 search-64 vs search-64 self-play games (INTERACTIVE_DECK mirror), 4
worker processes, env reseeded per game (`manabot/verify/run_distill_datagen.py`).
At **every** decision by **both** players we record the full encoded
observation dict exactly as the Agent consumes it (all 17 tensors including
validity masks and `action_focus`) plus the searcher's argmax action index.
`skip_trivial` is on, so every surfaced decision has ≥2 valid actions.

- **73,443 decisions** (target ≥50k), mean 122.4 decisions/game, mean 3.08
  valid actions/decision; 14.48M playouts, 0 step-cap hits.
- Seat-0 (on the play) won 47.0% of the 600 mirror games — near-balanced.
- Teacher action mix: spell 28.7%, pass 20.7%, attack 18.8%, land 17.0%,
  target 8.5%, block 6.2%.
- Cost: **762.4 s wall** on 4 workers (0.212 hr), 2,338.8 s engine
  (**0.65 engine-core-hours**, cap was 8).

### Task 2 — behavior cloning

Fresh default Agent (100,354 params — note: *larger* surface than the PPO
recipe's attention-off 50,306-param agent, see caveats), cross-entropy on the
teacher's action through the Agent's own masked logits (invalid actions at
−1e8, so the distribution is over valid actions only). 90/10 train/val split
**by game** (no within-game leakage). Sweep, all billed:

| config | final val loss | final val acc | wall |
|---|---|---|---|
| lr 1e-3 × 10 epochs | 0.9697 | 0.5076 | 216 s |
| lr 3e-4 × 10 epochs | 0.9761 | 0.5037 | 215 s |
| lr 3e-3 × 25 epochs | 0.9649 | 0.5113 | 568 s |
| **lr 1e-3 × 25 epochs** (selected, by val acc) | **0.9642** | **0.5114** | 590 s |

Reference points for the accuracy number: uniform-over-valid = 0.389,
always-index-0 = 0.488. The student only marginally beats the positional
prior on *exact* action matching — but search-64's argmax is itself noisy
(64 sims/action; near-ties resolve essentially at random), so exact-match
accuracy is bounded well below 1 by teacher stochasticity. Strength, not
accuracy, is the judged quantity. A mid-pipeline 200-game probe of the
10-epoch policy (77.5% vs random) motivated the 25-epoch extension; the probe
is billed to BC.

### Cost ledger (all wall-clock at the g5.xlarge rate, $1.006/hr; exp-00 accounting)

| item | wall | $ |
|---|---:|---:|
| dataset generation (600 games, 4 workers) | 762.4 s | $0.213 |
| BC sweep 1 (2 configs × 10 epochs) | 431 s | $0.120 |
| BC sweep 2 (2 configs × 25 epochs) | 1,158 s | $0.324 |
| mid-pipeline probe (200 games) | ~10 s | $0.003 |
| **total_cost_BC** | **2,361 s (0.656 hr)** | **$0.660** |
| matched-cost PPO (below), realized | 2,043.7 s (0.568 hr) | $0.571 |
| bc-fifth variant (§4: 254.1 s datagen share + 75 s train) | 329.1 s | $0.092 |

(Datagen ran 4 workers — the vCPU count of the reference g5.xlarge — so its
wall-clock is what that instance would bill. Engine-core-hours: 0.65 of the
8-hour cap.)

### Task 3 — matched-cost PPO

One fresh run of the exp-01 recipe (`first_light_shaped_v1`, shaping
land 0.03 / creature 0.06 / life-loss 0.01, random opponent), INTERACTIVE_DECK
via the exp-01 deck rebind, 16 envs × 128 steps, seed 1
(`manabot/verify/run_distill_ppo.py`). In-loop periodic eval disabled and
harness evals excluded from the bill: every billed PPO second is learning —
the accounting cut that favors PPO.

**Sizing amendment (documented, not silent):** the pre-registration said
"use 637 SPS to size it." Measured steady-state SPS on today's harness is
**2,472** (65,536-step calibration run; the Rust vector-env rewrite landed
after exp-00's 637 was measured). Sizing by the stale constant would have
given PPO 1.50M steps ≈ $0.17 of compute — a 3.9x undercosting against the
$0.66 target. PPO is instead sized by *measured* SPS to match wall-clock
dollars: 2,361 s × 2,472 SPS → **5,836,800 timesteps** (2,850 updates).
Realized: 2,043.7 s train wall at 2,856 SPS — $0.571, a 13% under-match
against BC's $0.660 (steady-state SPS rose further over the long run).
5.84M steps is 22x the training steps of any prior run on this deck.
Checkpoints persisted every 100 updates plus final (30 files).

### Task 4 — judging

Seat-balanced (alternating play/draw), 400 games per matchup (200/seat),
Wilson 95% CIs, per-seat rates — the exp-00c/exp-01/exp-02 protocol
(`manabot/verify/run_distill_judge.py`, 8 workers). Policies play
stochastically (sampling from masked softmax), exactly as every prior
checkpoint was measured. Behavioral profiles via the exp-01
`capture_evaluation` instrument (400 games vs random, seat-balanced).

## Results

### 1. Headline: BC vs matched-cost PPO (seat-balanced, 400 games, Wilson 95%)

| matchup | overall | on play | on draw |
|---|---|---|---|
| **bc-search64 vs random** | **90.5%** [87.2, 93.0] | 91.5% [86.8, 94.6] | 89.5% [84.5, 93.0] |
| **ppo-matched vs random** | **52.7%** [47.9, 57.6] | 41.5% [34.9, 48.4] | 64.0% [57.1, 70.3] |
| **bc-search64 vs ppo-matched** | **82.0%** [77.9, 85.5] | 66.5% [59.7, 72.7] | 97.5% [94.3, 98.9] |

The BC student is 38 points above PPO against random (non-overlapping CIs by
a wide margin) and beats it 82% head-to-head. PPO at 5.84M steps lands
*within noise of* — and below the best of — exp-01's 262k-step runs
(56.5/47.3/57.5%): 22x the training steps bought the shaped recipe nothing.
PPO's 13% wall-clock shortfall vs the BC bill (below) cannot bridge that.

### 2. BC vs the C1v2 checkpoints (every previously trained policy)

| opponent (its vs-random strength) | BC win rate | on play | on draw |
|---|---|---|---|
| c1v2-s1 (56.5%) | 84.3% [80.4, 87.5] | 82.0% | 86.5% |
| c1v2-s2 (47.3%) | 89.5% [86.1, 92.1] | 87.5% | 91.5% |
| c1v2-s3 (57.5%) | 81.0% [76.9, 84.5] | 83.0% | 79.0% |

Secondary bar from exp-02 (≥90% vs random, ≥84% vs C1v2): vs-random met
(90.5%); vs-C1v2 met for s1/s2, narrowly missed for s3 (81.0%). The student
sits at the bottom edge of the search-16 band, consistent with its ladder
placement below.

### 3. Ladder placement (goal 6 — first real point)

Pre-registered question: largest N in {16, 64} that BC beats — **neither**.
Extended with finer rungs (same protocol, 400 games each):

| vs search-N | BC win rate | on play | on draw |
|---|---|---|---|
| N=4 | 54.7% [49.9, 59.6] | 35.0% | 74.5% |
| N=8 | 50.5% [45.6, 55.4] | 61.0% | 40.0% |
| N=16 | 40.5% [35.8, 45.4] | 30.5% | 50.5% |
| N=64 (its teacher) | 28.0% [23.8, 32.6] | 7.0% | 49.0% |

**Ladder strength ≈ N=8** (interpolated; crosses 50% between 8 and 16). The
student recovers roughly the strength its teacher has at 1/8 of the teacher's
sim budget, at 0.98 ms/decision (measured, single-thread batch-1) instead
of 31. Every previous trained policy
was far below N=16 (search-16 beat them 84–93%, i.e. they sit at ≲N≈1); the
chart's trained-policy frontier moves from ~N≈1 to N≈8 for $0.66.
Exploitability sanity probe (north-star clause) not run this cycle — the
ladder number is provisional until C5's probe.

### 4. Cost to reach PPO's final strength (the ≤1/5 clause)

A deliberately cheap variant: first 200 games of the dataset (24,509
decisions, datagen share 254.1 s) + one config (lr 1e-3 × 10 epochs, 75 s),
no sweep — **total 329 s = $0.092 = 1/6.2 of PPO's realized $0.571**:

| policy | cost | vs random |
|---|---|---|
| bc-fifth (200 games, 10 epochs) | **$0.092** | **71.8%** [67.1, 75.9] (55.0% play / 88.5% draw) |
| ppo-matched | $0.571 | 52.7% [47.9, 57.6] |

Non-overlapping CIs: a sixth of PPO's spend exceeds PPO's final strength by
~19 points. Both seats stay above 50%.

## Prediction verdict

**CONFIRMED, both clauses.**

1. *BC matches or beats matched-cost PPO*: 90.5% vs 52.7% against random;
   82.0% head-to-head. Not close.
2. *≤1/5 the cost to reach PPO's final strength*: reached (and exceeded by
   ~19 points) at 1/6.2 of PPO's realized cost.

The secondary exp-02 bar is met on vs-random (90.5% ≥ 90%) and 2-of-3 on the
C1v2 clause (84.3/89.5/**81.0** vs ≥84) — distillation is losing *some*
signal relative to even search-16 (see ladder), but far less than PPO loses
to sparse-signal RL.

## Behavioral inheritance

capture_evaluation instrument (400 games vs random, seat-balanced), teacher
row from exp-02's fingerprint probe:

| policy | cast_when_able | passed_when_able |
|---|---|---|
| search-64 (teacher) | 0.58 | 0.35 |
| **bc-search64 (student)** | **0.61** | **0.34** |
| ppo-matched | 0.97 | 0.007 |
| C1v2 s1/s2 (exp-01) | 0.88–0.95 | 0.04–0.12 |

**The student inherits the teacher's patience almost exactly** (0.61/0.34 vs
0.58/0.35): it holds castable spells through a third of its opportunities —
the value-of-held-interaction behavior that search discovers and that no
shaped-RL policy here has ever shown. The matched-cost PPO run reproduces
the C1v2 aggro fingerprint (cast 0.97, pass 0.007) — the shaping installs the
same racing prior at 5.84M steps as at 262k. (Profile-instrument win rates —
BC 86.5%, PPO 58.8% — differ a few points from the matchup harness; different
opponent implementation and unseeded policy sampling. The matchup harness is
canonical for all strength claims.)

## Honest caveats

- **The student distills its teacher's blind spots.** The teacher is a deaf,
  naive flat-MC searcher: it maximizes the mean over determinized worlds, so
  it cannot plan on information it doesn't have, cannot value gathering or
  concealing information, and cannot bluff (strategy fusion — it implicitly
  assumes it can act differently in worlds it cannot tell apart; plus
  non-locality, where a determinized world's value depends on the opponent's
  inference elsewhere in the tree). The student's "patience" is a distilled
  average of these value plays, not deception. It also inherits the teacher's
  implicit uniform-random opponent model. An exploitability probe (Exit 1's
  instrument) is the real test; not run this cycle.
- **Architecture asymmetry, pre-registered but real:** the student is the
  default 100,354-param attention-on Agent (per this cycle's spec); the PPO
  recipe (unchanged from exp-01) trains the attention-off 50,306-param agent.
  The comparison is pipeline-vs-pipeline, not architecture-controlled — but
  exp-01's identical recipe at 262k steps and this run's 5.84M steps bracket
  the recipe's ceiling regardless of size.
- **SPS sizing amendment.** Pre-registration said size PPO by 637 SPS; the
  harness now trains at ~2,856 SPS (Rust vector env landed post-exp-00).
  Sized by measured calibration (2,472), realized 2,043.7 s — PPO got $0.571
  of wall-clock against BC's $0.660 (13% under). Direction noted; given PPO's
  flat strength from 262k → 5.84M steps, 13% more wall does not change the
  verdict.
- **Dataset states are search-vs-search self-play states**; the student plays
  its judged games off that distribution (vs random, vs PPO, vs itself-ish
  opponents). Empirically it generalized; no claim beyond these opponents.
- **Per-seat softness on the play vs its teacher** (7.0% on play vs
  search-64, 35.0% vs search-4): the student's on-the-play game is its weak
  half against searchers; seat-balanced numbers mask this (reported per-seat
  throughout).
- **val-accuracy is a weak selection metric** (best 0.5114 vs 0.488
  always-index-0): exact-match accuracy is bounded by teacher argmax noise;
  it still rank-ordered the candidates correctly here (25-epoch > 10-epoch,
  90.5% > ~77.5% probe), but selection-by-strength would need eval games.
- **C1v2 opponents were trained pre-fizzle-fix** (exp-02 note); judged a
  negligible confound, same as exp-02.
- Policy sampling in evaluation is stochastic and unseeded across runs;
  repeat matchups drift ~±1 point (observed 72.5% → 71.8% on bc-fifth).

## Ledger update (strength vs cumulative training cost)

| point | cost | vs random (seat-balanced) | ladder N |
|---|---:|---|---|
| C1v2 best (exp-01) | $1.30 cum | 57.5% | ≲1 |
| search-64 (exp-02) | $0 train | 95.0% | — (is the ladder) |
| **bc-search64 (this cycle)** | **$0.66** | **90.5%** | **≈8** |
| ppo-matched (this cycle) | $0.57 | 52.7% | ≲1 |

## Next question (C5)

**Does the loop close?** Replace the teacher's uniform-random rollouts with
the BC policy (policy rollouts — pulls batched inference, wave goal 1), let
search-with-student generate the next dataset, re-distill, and measure the
ladder rung again. One iteration answers the expert-iteration question: does
student-guided search beat random-rollout search enough that the next student
climbs above N=8? Run the exploitability probe on the frozen BC policy in
the same cycle — the ladder number above is provisional until then.

## Provenance

- Branch: worktree from `search-wave-roadmap` merge (`171f1a0`) + this cycle's commits.
- Dataset: `.runs/exp03/dataset/` (4 shards + manifest, gitignored; summaries here).
- BC: `.runs/exp03/bc_policy_v2.pt`, logs `.runs/exp03/bc_log{,_v2}.json`.
- PPO: `.runs/first-light-exp03-ppo-matched-final/step_5836800.pt`, log `.runs/exp03/ppo_log.json`;
  SPS calibration `.runs/exp03/ppo_spscal_log.json`.
- Judge data: `reports/data/exp-03-distillation.json`.
- C1v2 opponents: `/Users/jack/src/manabot/.runs/first-light-c1-interactive-dev2-s{1,2,3}-final/step_65536.pt`
  (read-only; trained pre-fizzle-fix).
- Machine: Apple M4 Max, Python 3.12, isolated venv, WANDB_MODE=disabled.

## Addendum (coordinating session): the fair-PPO caveat

The matched-cost PPO baseline ran `first_light_shaped_v1` — the recipe exp-04
convicted while this experiment was in flight. Best-practice PPO on this deck
is terminal-only (exp-04: 60.0–75.5% at $0.14/seed). Against THAT reference,
the honest headline is "BC-from-search beats good PPO by ~15+ points at equal
cost and reaches its strength band at well under half the cost" — not the
38-point spread vs the shaped baseline. The 22x-budget shaped run reproducing
the aggro fingerprint exactly (cast 0.97 / pass 0.007) is itself a finding:
the bias is budget-invariant. Directional verdict unchanged; margins repriced.
