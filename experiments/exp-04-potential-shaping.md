# exp-04: E2c — potential-based shaping (does dense signal need the aggro fingerprint?)

**Cycle:** C2 (wave/intelligence/01-experiment-loop.md) · **Date:** 2026-07-09 ·
**Recipe:** first-light dev preset (4 envs, 262k steps, random opponent),
INTERACTIVE_DECK both players — identical to E2b except the reward ·
**Runs:** `c2-potential-s{1,2,3}` in `.runs/verify.sqlite` (worktree).

## Question

Does potential-based shaping learn without the aggro fingerprint?

Ng, Harada & Russell (1999): shaping of the form γ·Φ(s′) − Φ(s) is provably
policy-invariant. The current shaping (pay-per-land/creature/damage) is not —
exp-01 showed it training agents into cast-everything failure (cast_when_able
0.88–0.95, one seed net-harmful at 47.3%).

## Pre-registered prediction

> E2c reaches ≥90% of E2b's win-vs-random while holding instants
> significantly longer. Fingerprint proxy: cast_when_able materially below
> E2b's 0.88–0.95 while win rate ≥ E2b's.

E2b reference (exp-01 attempt 2, seat-balanced 400g): 56.5% / 47.3% / 57.5%,
cast_when_able 0.88 / 0.95 / 0.50, passed_when_able 0.12 / 0.04 / 0.41.

## Method

### Potential definition

Hero-perspective board-state potential:

    Φ(s) = w_land · (hero bf lands − villain bf lands)
         + w_creature · (hero bf creatures − villain bf creatures)
         + w_life · (hero life − villain life) / 20

Shaping term added to every hero step reward: **γ·Φ(s′) − Φ(s)** with
γ = potential_gamma = 0.99 = train.gamma (matching the training discount is
required for policy invariance).

**Terminal handling (critical):** Φ(terminal) is treated as 0. The terminal
step's shaping is exactly −Φ(s_last), added *on top of* the ±1 win/lose
terminal reward. Without this the potential does not telescope out and
invariance breaks. In the tensorized path this also avoids a mechanical bug:
on done steps the observation buffers already hold the *next* episode's reset
observation, which must not leak into the shaping. Truncated episodes are
treated the same as terminated ones (Φ = 0), consistent with the existing
terminal masking.

### Weights

    potential_land_weight     = 0.03
    potential_creature_weight = 0.06
    potential_life_weight     = 0.2   (life diff is /20-normalized → 0.01 per life point)

Chosen so per-event shaping magnitudes match E2b's per-event payments
exactly: playing a land moves Φ by +0.03, a creature by +0.06, a point of
opponent life by +0.01 — the same 0.01–0.06 per-step scale as
`FIRST_LIGHT_REWARD` (land 0.03, creature 0.06, life 0.01). This makes
E2b-vs-E2c a controlled comparison of shaping *form* (path-dependent payment
vs potential difference), not of signal scale. The differences from E2b's
signal: Φ is *symmetric* (villain's board subtracts — the villain resolving a
creature is negative signal), *refundable* (your creature dying takes the
0.06 back; Pyroclasm'ing your own board is immediately negative), and
*telescoping* (no net reward for a round trip). Worst-case |Φ| on this deck
is ≈1 (≈0.3 lands + ≈0.4 creatures + ≈0.4 life at extreme board states), so
the terminal −Φ(s_last) correction is comparable to but does not dominate the
±1 terminal reward; the (1−γ)·Φ per-step leak is ≤ ~0.01.

### Implementation

- `manabot/infra/hypers.py` — `RewardHypers.potential_enabled` (default off),
  `potential_gamma`, `potential_{land,creature,life}_weight`.
- `manabot/env/vector_env.py` — `_compute_potential_shaping` /
  `_potential`; shaping added after the terminal win/lose overwrite so the
  −Φ(s_last) correction survives on done steps. Training path.
- `manabot/env/match.py` — same term in the single-env `Reward.compute`
  (eval path; eval discards rewards but the paths now agree).
- `tests/env/test_potential_shaping.py` — sign conventions (hero positive,
  villain negative, non-battlefield zones excluded), terminal −Φ(s_prev)
  handling, and the telescoping property end-to-end: with γ=1 the per-episode
  shaped return equals the unshaped return exactly (Φ(s₀)=0 from the
  symmetric start) while per-step rewards differ.

E2b's pay-per-event weights are all zeroed in the E2c reward config; the only
dense signal is the potential difference.

### Judging

Each final checkpoint: seat-balanced 400-game eval vs random
(`capture_evaluation(..., seat_balanced=True)`, stochastic policy, eval seeds
4001–4003). E2b checkpoints re-judged with the identical harness (eval seeds
2001–2003, as in exp-01) to add `landed_when_able` to the reference row and
confirm reproduction.

## Results

Seat-balanced, 400 games (200/seat), stochastic policy, vs random. Per-seat
baselines on this deck (exp-00c): a random policy wins 23.1% on the play,
76.9% on the draw.

### E2c — potential-based shaping (`c2-potential-s{1,2,3}`)

| seed | overall | LB | on play (LB) | on draw (LB) | cast_w_able | passed_w_able | landed_w_able |
|---|---|---|---|---|---|---|---|
| 1 | 65.2% | .605 | 42.0% (.354) | 88.5% (.833) | **0.35** | 0.50 | 0.87 |
| 2 | **75.0%** | .705 | 99.0% (.964) | 51.0% (.441) | **0.38** | 0.54 | 0.49 |
| 3 | 68.2% | .635 | 77.0% (.707) | 59.5% (.526) | **0.38** | 0.52 | 0.50 |

### E2b — pay-per-event shaping (re-judged, `c1v2-s{1,2,3}` checkpoints)

Same harness, eval seeds 2001–2003; exp-01's original numbers in parentheses.

| seed | overall | LB | on play | on draw | cast_w_able | passed_w_able | landed_w_able |
|---|---|---|---|---|---|---|---|
| 1 | 55.7% (56.5%) | .509 | 65.5% | 46.0% | 0.88 | 0.11 | 0.56 |
| 2 | 43.2% (47.3%) | .385 | 61.5% | 25.0% | 0.95 | 0.05 | 0.58 |
| 3 | 58.3% (57.5%) | .534 | 18.5% | 98.0% | 0.50 | 0.41 | 0.52 |

The re-judge reproduces exp-01 within noise and confirms the reference row.

### Reading

1. **Every E2c seed beats every E2b seed outright.** Worst E2c overall
   (65.2%, LB .605) is above best E2b overall (58.3%, LB .534) with
   non-overlapping Wilson intervals. Means: 69.5% vs 52.4%. Same recipe, same
   deck, same step budget — the only change is the shaping form.
2. **The aggro fingerprint is gone, uniformly.** cast_when_able 0.35–0.38
   across all three seeds (E2b: 0.88–0.95 in the two aggro seeds);
   passed_when_able 0.50–0.54 (E2b aggro seeds: 0.04–0.12). E2c seeds pass
   priority with spells in hand roughly half the time and win more doing it.
   The behavioral profile is also far more *consistent* across seeds than
   E2b's (which spanned cast 0.50–0.95).
3. **Seed 3 is the first policy ever to pass the exp-00c gate** (overall LB
   .635 > .55; play LB .707 and draw LB .526 both > .50). Seeds 1 and 2 fail
   only the per-seat clause, in opposite directions (s1 weak on the play at
   42%, s2 weak on the draw at 51%): the shaping form fixed the strategic
   prior, not seat specialization. Notably s2's failure seat is still at
   random-level-ish (51% vs random's 76.9% on the draw) while its play seat
   is 99% — a new flavor of seat asymmetry worth watching, though unlike
   E2b's seed 3 it stays well above 50% overall in both directions of the
   metric that matters.
4. Anti-correlated with E2b, landed_when_able is *high* where the win rate is
   carried by the draw seat (s1: 0.87) — consistent with a develop-and-defend
   posture rather than the racing posture the old shaping paid for.

## Prediction verdict

**Confirmed, with margin.** Pre-registered bar: ≥90% of E2b's win rate while
holding instants significantly longer (cast_when_able materially below
0.88–0.95 at win rate ≥ E2b). Measured: 112–129% of E2b's *best* seed, with
cast_when_able at 0.35–0.38 — under half the aggro seeds' rate and below even
E2b's non-aggro seed (0.50). Dense signal no longer costs a strategic prior.
Per the loop doc, goal 5's "delete shaping" becomes "replace with Φ", and aux
heads stay dead.

## Caveats

- **cast_when_able is a proxy, not a trick meter.** A low value is consistent
  with holding instants for value *or* with generic passivity. The
  passed-then-won pattern and the win-rate gap argue for the former, but a
  direct metric (Bolt cast during combat / Counterspell cast with a spell on
  the stack) would settle it; worth adding before this number is quoted as
  "instant-holding."
- **Policy invariance is a statement about optimal policies.** At a 262k-step
  budget, shaping still changes learning dynamics — that is its entire
  purpose. The claim supported here is that the potential form does not
  *install* the cast-everything prior, not that finite-budget training is
  unbiased in every respect.
- **Truncated episodes are treated as terminal** (Φ = 0) in both paths,
  a minor invariance deviation; rollout health logged zero truncated episodes
  in these runs.
- **Seat asymmetry persists per seed** (finding 3). The gate's per-seat
  clause still fails 2/3 seeds; whatever causes seat specialization is not
  the shaping form.
- **One weight scale tested.** Weights were deliberately matched to E2b's
  per-event magnitudes to isolate the form; no sensitivity sweep.
- **Comparison against terminal-only (E2a) is pending its report.** Judging
  numbers observed in this workspace's shared scratchpad for the parallel
  E2a runs (75.5% / 64.5% / 60.0% overall, cast 0.15–0.29) suggest
  terminal-only *also* escapes the fingerprint and lands near E2c's range on
  this budget. If those numbers hold up, the honest bar for keeping any dense
  shaping is "beats terminal-only," not "beats the old shaping" — E2c's mean
  (69.5%) vs that E2a mean (66.7%) is not obviously a significant gap.
- C3's ladder is unaffected: search-at-16 (91.7% vs random) still beats every
  trained policy, including these.

## Ledger

New chart point: **~69.5% mean (65.2–75.0%) seat-balanced vs random** at
+3 dev runs ≈ $0.45 (g5 accounting rate), cumulative training spend ≈ $1.75.

## Next question

Is dense signal worth anything at all? E2c beats E2b decisively, but the
observed E2a (terminal-only) numbers land close to E2c's. The next cheap
experiment is a properly powered E2a-vs-E2c comparison (more seeds or longer
runs, same judging harness) to decide whether Φ earns its complexity before
it is promoted into the default recipe — then C4 (distillation vs RL) as
scheduled, using whichever reward wins as the PPO baseline.

## Provenance

- Runs: `.runs/verify.sqlite` (worktree `worktree-agent-a9873295eb69e8e95`),
  labels `c2-potential-s{1,2,3}`, seeds 1–3, dev preset, random opponent.
- Checkpoints: `.runs/first-light-c2-potential-s{n}-final/step_65536.pt`
  (chunk-local step counters, as in exp-01).
- Judging: `scratch/judge_c2_potential.py`, 400 games seat-balanced,
  eval seeds 4001–4003 (E2c) and 2001–2003 (E2b re-judge);
  raw rows in `reports/data/exp-04-potential-judging.json` and
  `reports/data/exp-04-e2b-rejudge.json`.
- Runner: `scratch/run_c2_potential.py` (INTERACTIVE_DECK rebind + potential
  reward config).
- Tests: `tests/env/test_potential_shaping.py` (8 tests); full suite 189
  passed (tests/gui excluded: missing `httpx2` in the experiment venv,
  unrelated). No Rust touched; cargo test not required.

## Addendum (coordinating session): E2a terminal-only, full results

Seat-balanced 400g vs random, INTERACTIVE_DECK, labels `c2-terminal-s{1,2,3}`:

| seed | overall (LB) | play / draw | cast_w_able | passed_w_able | landed_w_able | gate |
|---|---|---|---|---|---|---|
| 1 | 75.5% (.711) | 74.0 / 77.0 | 0.29 | 0.60 | 0.66 | PASS |
| 2 | 64.5% (.597) | 62.5 / 66.5 | 0.17 | 0.76 | 0.44 | PASS |
| 3 | 60.0% (.551) | 56.0 / 64.0 | 0.15 | 0.78 | 0.69 | marginal (per-seat) |

The pre-registered E2a prediction (pass-collapse reproduces) is **refuted**.
Under first-light's diagnostics these behavioral profiles read as collapse
(passed_when_able 0.60–0.78); they coexist with the best trained win rates to
date. On this deck heavy passing is patience. Note E2a's seats are balanced
where E2c seeds 1–2 are strongly seat-lopsided (42/88.5, 99/51) — terminal-only
produced the more seat-robust policies.

Open questions carried forward: (1) powered E2a-vs-E2c comparison (≥10 seeds
or head-to-head) before Φ enters any recipe; (2) the discriminating run for
*why* first-light's terminal-only collapsed — same recipe on STANDARD_DECK,
current code, labels `c2-vanilla-terminal-s{1,2,3}`, in flight at time of
writing. Registered prediction (60/40): no collapse — pointing to code-era
artifact over deck mechanism.

## Addendum 2: the discriminating run — first-light's pass-collapse was an artifact

`c2-vanilla-terminal-s{1,2,3}`: terminal-only reward, STANDARD_DECK (the
original first-light deck), current code, seat-balanced 400g judging:

| seed | overall (LB) | cast_w_able | passed_w_able | landed_w_able |
|---|---|---|---|---|
| 1 | 64.0% (.592) | 0.999 | 0.31 | 0.35 |
| 2 | 66.3% (.615) | 0.987 | 0.40 | 0.27 |
| 3 | 60.8% (.559) | 0.987 | 0.37 | 0.29 |

**No pass-collapse on either deck.** The registered 60/40 prediction (no
collapse → code-era/measurement artifact, not deck mechanism) resolved on the
60 side. First-light's founding observation — "pure terminal reward fails...
produces pass-collapse" (wave/archive/first-light/README.md, unsourced) — fails
direct replication on its own deck with current code. Candidate causes of the
original observation: the then-unfixed PPO bugs (first-light goal 1), and/or
hero-on-play single-seed 50-game evals reading behavioral drift as failure.

The pair of E2a results is the cleanest demonstration of shaping's cost
available: the SAME terminal-only reward produces cast_when_able ≈ 0.99 on the
vanilla deck (casting is correct there) and 0.15–0.29 on the interactive deck
(holding is correct there) — the un-shaped policy adapts its behavior to the
deck. Pay-per-event shaping forced vanilla-deck behavior onto the interactive
deck. The reward didn't need to encode strategy; it needed to stop doing so.

Also noteworthy: vanilla winners play FEWER lands than random (0.27–0.35 vs
~0.44 baseline) — surplus land plays are dead actions once the curve is met.
Under first-light's diagnostics, declining landed_when_able was the definition
of failure; even the failure signature was backwards.
