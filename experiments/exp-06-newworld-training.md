# exp-06: C6a — new-world training (does the expanded observation/action space hurt?)

**Cycle:** C6 (wave/search/01-experiment-loop.md) · **Date:** 2026-07-09 ·
**Engine:** `origin/rules-stage-2` merged (Stage 1+2: CARD_DIM 37,
PERMANENT_DIM 7, PLAYER_DIM 27, ACTION_TYPE_DIM 14, mid-resolution decision
ActionSpaceKinds) · **Recipe:** first-light dev preset (4 envs, 262k steps,
random opponent), INTERACTIVE_DECK both players, terminal-only reward —
the exp-04 E2a arm re-run on the new engine ·
**Runs:** `c6a-newworld-s{1,2,3}` in `.runs/verify.sqlite` (worktree).

## Question

Same game, bigger representation: does training degrade?

## Pre-registered prediction (verbatim)

> Historical control: exp-04's E2a arm (terminal-only reward, dev preset 262k
> steps, INTERACTIVE_DECK mirror, random opponent) scored 60.0/64.5/75.5%
> seat-balanced vs random on the OLD dims (CARD 29/33-era). PREDICTION:
> new-world seeds land within the 60-75% band → expansion benign. All seeds
> <55% → adverse effect confirmed; first suspect is action-type one-hot
> dilution (7→14 types for actions this deck rarely uses); second is the
> grown card feature vector diluting signal at 100k params.

## Smoke (training path, before any training)

200 random-vs-random games on INTERACTIVE_DECK through the TRAINING path
(manabot `VectorEnv` → Rust vector env with buffer-based observation
encoding and action masking), hero acting uniformly over the valid mask,
`opponent_policy="random"`, terminal-only reward (`scratch/smoke_c6a.py`):

- **200/200 episodes completed, zero crashes, zero non-finite observation
  values, zero reward-policy violations** (reward exactly 0 off-terminal,
  exactly ±1 on-terminal; terminal histogram 101 wins / 99 losses — sane for
  random-vs-random).
- Action-type coverage over the 14-way one-hot: the classic six types
  surfaced (PLAY_LAND, CAST_SPELL, PASS_PRIORITY, DECLARE_ATTACKER,
  DECLARE_BLOCKER, CHOOSE_TARGET). None of the 8 new Stage-2 types
  (ACTIVATE_ABILITY, SCRY_KEEP/BOTTOM, SELECT_CARD, DECLINE_CHOICE,
  PAY_COST, CHOOSE_MODE, TAP_FOR_COST) appear on this deck — expected: the
  deck has no scry/kicker/modal/learn cards. C6a therefore tests dimension
  *dilution*, not the new decision kinds themselves; those get their first
  training exposure in C6b.
- Short PPO run (10,240 steps): finite losses, parameters moved
  (ΔL2 = 0.37), no NaNs. Gradients flow.

**Smoke verdict: training path healthy on the new engine.** No engine or
binding fixes were needed; cargo untouched.

## Results

### Standard-protocol judging (like-for-like with the E2a control)

Seat-balanced 400 games vs random per final checkpoint, stochastic policy,
eval seeds 6001–6003 (`scratch/judge_c6a.py`,
`experiments/data/exp-06-newworld-judging.json`). This is the exact harness the
E2a control was judged with — including its just-discovered single-deal
limitation (next section): each row is 400 stochastic rollouts of **one**
deal, so CIs are deal-conditional.

**C6a — new world (`c6a-newworld-s{1,2,3}`):**

| seed | overall | LB | on play (LB) | on draw (LB) | cast_w_able | passed_w_able | landed_w_able |
|---|---|---|---|---|---|---|---|
| 1 | 69.3% | .646 | 66.0% (.592) | 72.5% (.659) | 0.42 | 0.49 | 0.44 |
| 2 | **77.3%** | .729 | 66.0% (.592) | 88.5% (.833) | 0.16 | 0.76 | 0.64 |
| 3 | 73.5% | .690 | 53.0% (.461) | 94.0% (.898) | 0.15 | 0.79 | 0.49 |

**E2a control — old world (exp-04 addendum, `c2-terminal-s{1,2,3}`):**

| seed | overall (LB) | play / draw | cast_w_able | passed_w_able | landed_w_able |
|---|---|---|---|---|---|
| 1 | 75.5% (.711) | 74.0 / 77.0 | 0.29 | 0.60 | 0.66 |
| 2 | 64.5% (.597) | 62.5 / 66.5 | 0.17 | 0.76 | 0.44 |
| 3 | 60.0% (.551) | 56.0 / 64.0 | 0.15 | 0.78 | 0.69 |

Two seeds inside the pre-registered 60–75% band, one **above** it (77.3%).
No seed near the <55% adverse-effect tripwire. Means 73.4% (new) vs 66.7%
(old) — favorable direction, not significant at n=3. The behavioral profile
reproduces the E2a signature: two of three seeds are patience-shaped
(cast 0.15–0.16, passed 0.76–0.79), seed 1 found a more castive optimum
(0.42) at a similar win rate. Seed 3 is seat-lopsided (53/94), which the
old-world E2a arm didn't show; with n=3 and one deal per judging row this is
noted, not interpreted.

### Fixed-harness judging (per-game deals; see instrument finding)

Same checkpoints, same protocol, after the `Env.reset(seed=...)` fix — each
game now plays a fresh deal (`experiments/data/exp-06-newworld-judging-fixed.json`):

| seed | overall | LB | on play (LB) | on draw (LB) | cast_w_able | passed_w_able | landed_w_able |
|---|---|---|---|---|---|---|---|
| 1 | **77.3%** | .729 | 75.0% (.686) | 79.5% (.734) | 0.41 | 0.50 | 0.43 |
| 2 | 70.3% | .656 | 73.5% (.670) | 67.0% (.602) | 0.17 | 0.74 | 0.60 |
| 3 | 66.5% | .617 | 69.5% (.628) | 63.5% (.566) | 0.10 | 0.84 | 0.50 |

Deal-averaged, all three seeds sit at 66.5–77.3% with Wilson LBs ≥ .617, and
— the instructive part — **seed 3's alarming single-deal seat-lopsidedness
(53/94) dissolves to 69.5/63.5**: what the standard protocol read as seat
specialization was mostly the policy's fit to one particular deal. Per-seat
splits from single-deal judging (including exp-01's "seat-parasitic" seed
and exp-04's E2c seat asymmetries) should be re-read with that in mind.

No fixed-harness numbers exist for the E2a control (its checkpoints live in
another worktree); the like-for-like comparison above is the pre-registered
one.

## Instrument finding: the eval harness was replaying one deal per run

Chasing a baseline discrepancy surfaced by the smoke (training path measured
random-vs-random on-the-play at ~50%, exp-00c's book value was 23.1%)
bottomed out in a pre-existing bug, present in every judged number to date:

- `manabot.env.Env.reset(seed=...)` passed the seed to gymnasium's
  `super().reset()` and **never to the engine**. `managym.Env` exposes no
  `set_seed`, and its `reset()` reuses the constructor seed — so every reset
  replays the identical shuffle. Verified directly: one `Env`, reset with
  seeds 100/200/none → byte-identical deals.
- `capture_evaluation` constructs one `Env(seed=eval_seed)` and calls
  `reset(seed=eval_seed + game_index)` per game. Net effect: **every
  "400-game seat-balanced eval" in exp-00c through exp-04 (and this
  experiment's standard-protocol table) was 400 stochastic rollouts of a
  single deal** (played from both seats). Wilson CIs in all those tables are
  conditional on that deal; cross-seed spread silently includes
  deal-to-deal variance because each judging run used a different
  constructor seed.
- The training path is NOT affected: the Rust `VectorEnv` reseeds each
  episode correctly (verified: per-episode seed strides, and a lockstep
  replay of episode 1 is bit-identical between paths; divergence begins
  exactly at the first auto-reset, where the vector env correctly advances
  the seed and the single env replays its deal).

Evidence chain (all on the new engine, INTERACTIVE_DECK mirror,
uniform-random both seats):

| measurement | path | on-the-play win rate |
|---|---|---|
| 1000 games, "seeds" 42–1041 | single Env (pre-fix) | 36.3% — one deal |
| 1000 games, raw action space | single Env (pre-fix, same deal) | 35.7% |
| 2000 games, seeds 7–2006 | single Env (pre-fix, one deal) | 38.8% |
| 1000 games | Rust VectorEnv (per-episode deals) | 49.1% |
| 1000 games, winner_index cross-check | Rust VectorEnv | 50.1% |
| 1000 games, per-game deals | single Env (post-fix) | **46.7%** [43.6, 49.8] |

Two consequences for the book:

1. **The per-seat random baselines are deal artifacts.** exp-00c's
   "random wins 23.1% on the play / 76.9% on the draw" on INTERACTIVE_DECK
   (and the 93.4% on-the-play figure on STANDARD_DECK behind amendment A1)
   are properties of single deals, not of the game. Deal-averaged, random
   mirror play on INTERACTIVE_DECK is near seat-parity (~47–50% on the play,
   pooled ≈48.6% over 3000 games). Seat-balancing evals (A1) remains correct
   practice; quoting those per-seat numbers as game facts does not.
   (Old-engine deal-averaged values were never measured, so the engine-shift
   component, if any, is unknown.)
2. **Historical judged win rates carry an unquantified deal-selection
   term.** Rankings within a report (same harness, same era) are less
   affected than absolute numbers; cross-report comparisons — including this
   experiment's headline table — inherit it. The fixed-harness table above
   is the first deal-averaged judging row in the book.

Fix (minimal, Python-only): `manabot/env/env.py` rebuilds the engine when
`reset(seed=...)` receives a new seed (exactly what the Python `VectorEnv`
wrapper already did). Regression tests in
`tests/env/test_env.py::TestResetSeed`. No Rust changes; the pyo3 surface is
untouched.

## Param / throughput deltas (cost-ledger denominators)

Parameter count (`Agent` on default `ObservationSpaceHypers`, new dims
player 27 / card 37 / permanent 7 / action 15-with-validity):

| config | old (exp-00) | new | Δ |
|---|---:|---:|---:|
| Default (attention on) | 100,354 | **101,506** | +1,152 (+1.1%) |
| Attention off (the config these runs train) | 50,306 | **51,458** | +1,152 (+2.3%) |

All growth is in the embedding layers (player +64, card +512, perm +128,
action +448 weights); attention/action/value heads are dim-independent.

Training SPS, exp-03's calibration recipe reproduced exactly
(`manabot.verify.run_distill_ppo`, 16 envs × 128 steps, 65,536 steps,
INTERACTIVE_DECK, in-loop eval disabled, same M4 Max):

| measurement | old | new |
|---|---:|---:|
| 16-env calibration (exp-03 protocol) | 2,472 | **3,013** (+22%) |
| dev-preset shape (4 envs), per-chunk steady state | n/a (unmeasured) | 1,439–1,759 |

Throughput did not regress with the wider observation — it improved; the
Stage-2 engine work (or ambient machine variance across sessions) more than
covers the extra encode width. Cost ledger should use **3,013 SPS** and
**101,506 / 51,458 params** as current denominators.

## Verdict

**Expansion benign — prediction confirmed** (with one seed landing above the
band, the favorable side). The training path digests the Stage 1+2
observation/action space without crashes, NaNs, or reward violations;
terminal-only PPO on the same deck reaches 69.3–77.3% seat-balanced vs
random (66.5–77.3% deal-averaged, Wilson LBs ≥ .617) against the old
world's 60.0–75.5%, reproducing the E2a behavioral signature. Neither suspected dilution mechanism (action-type one-hot 7→14,
card feature growth) shows any adverse signal at this budget. Param count
+1.1%, throughput +22%.

The unregistered yield of the cycle is the instrument finding above: the
judging harness had been evaluating one deal per run since exp-00. Fixed and
regression-tested; deal-averaged judging is now the default going forward.

## Next question (C6b)

**Pool expansion on the real matchup:** train on UR Lessons vs GW Allies
(the wave/rules two-deck slice) once rules stage 3 lands — first training
exposure of the new decision kinds (scry / look-and-select / pay-or-not /
modal / learn / waterbend) that C6a's deck never surfaces, on an asymmetric
matchup where deck identity, not just seat, differentiates policies. Queued
on rules stage 3.

## Provenance

- Engine: `origin/rules-stage-2` (a171fa1) merged into the worktree branch;
  cp312 extension rebuilt release-profile; cargo untouched.
- Runs: `.runs/verify.sqlite`, labels `c6a-newworld-s{1,2,3}` (run_ids 1–3),
  seeds 1–3, dev preset, random opponent, terminal-only reward via the
  exp-01/exp-04 `fl.STANDARD_DECK`/`fl.FIRST_LIGHT_REWARD` rebind
  (`scratch/run_c6a.py`). Training walls 176.7 / 190.3 / 198.2 s.
- Checkpoints: `.runs/first-light-c6a-newworld-s{n}-final/step_65536.pt`
  (chunk-local counters, as in exp-01/exp-04).
- Judging: `scratch/judge_c6a.py`; standard-protocol rows in
  `experiments/data/exp-06-newworld-judging.json` (pre-fix harness, eval seeds
  6001–6003), fixed-harness rows in
  `experiments/data/exp-06-newworld-judging-fixed.json` (same seeds, per-game
  deals).
- SPS calibration: run_id 4 (`c6a-sps-cal`), log `.runs/exp06/sps_cal.json`.
- Smoke: `scratch/smoke_c6a.py`.
- Tests: full suite green including the new
  `tests/env/test_env.py::TestResetSeed`.
