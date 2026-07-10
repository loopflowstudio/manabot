# Exp-10 — The Goal-4 Gate: search-with-V vs V-greedy (wave/intelligence C10)

Date: 2026-07-10
Machine: Apple M4 Max (shared with other agents throughout; load average
100-300 during the morning measurement window — all absolute ms/dec are
upper bounds, ratios and win rates are trustworthy).
Deck: INTERACTIVE_DECK mirror, both seats. Seat-balanced everywhere (A1).
Accounting: exp-00 cost basis, $1.006/hr wall-clock. Cap: 6 h — **exceeded;
see the cost ledger for the honest breakdown.**

**Reading order note.** This experiment was run twice. The first battery was
played on the broken deal instrument (every game in a run on the same
opening deal — the exp-06 `Env.reset(seed=)` bug, which this worktree still
carried); the second battery, after merging the fix, is deal-diverse and is
**the record**. Deviation 3 tells the story; the deal-narrow numbers are
kept in the side-by-side table there because several of them dissolve
instructively.

## Why this gate, now

Exp-07 refuted the policy-rollout crank on label economics: an affordable
policy-rollout teacher (8 playouts) labels worse than 256 random playouts,
because policy playouts cost ~6x per decision (~160x per playout). The escape
route the wave pre-registered (goal 4, now under `wave/intelligence/`) is a
VALUE HEAD at rollout depth 0 — one forward pass replacing an entire playout
of signal — IF search on top of V still improves on V. The gate is not V
accuracy:

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
(3 of 8 shards, 27,423 decisions, 121.9 decisions/game) in time to train V:
the first 4-worker run was reaped by the session harness at ~45 min with 3
shards durable; the relaunched run died silently overnight (empty log, no
shards written) across a machine-sleep window; a third, supervised
small-shard run was killed by a coordinating-session intervention because
the compute cap was already blown. Root causes, honestly: (a) long-running
background jobs in this harness need external supervision, which the first
two runs lacked; (b) **the completion monitors watched only for success
artifacts (shard files), not for worker liveness/progress**, so ~10
wall-clock hours passed with 3 shards on disk and nobody noticing; (c) the
box was shared, sometimes at load average 300, and slept overnight mid-run.

Per the loop's redesign-don't-extend rule, the cycle closed with what was on
disk: **225 games / 27,423 decisions**. Consequences: V trained on 37% of
the pre-registered data; economics cells at 200 games instead of 400. The
gate itself ran at the full pre-registered 400 games.

The report originally noted here that seat 0 won 68.9% of the training
games (155/225) and read it as a play advantage of search-256 self-play in
the stage-3 world. Deviation 3 corrects that reading: those 225 games were
played on **three opening deals**, so 68.9% is a 3-deal seat asymmetry, not
a game fact.

## Deviation 3: the deal instrument bug — the first battery was single-deal

Mid-experiment, the coordinating session flagged that this worktree predated
the exp-06 seed fix (`d4a5ebe`, amendment A5 in the cycle log:
`Env.reset(seed=)` historically never reached the engine). A direct
instrument check confirmed the worst case: **five resets across three
different seeds produced the identical opening deal** — same-hash hands,
both seats.
Only the *constructor* seed reached the engine, and the runners construct
one `Env` per worker. Concretely:

- **Training data**: each datagen worker wrote one shard from one `Env`, so
  each shard's games all share one opening deal. The 225 training games
  (shards 01-03) contain **3 distinct deals**; the full 600 games eventually
  on disk (7 shards) contain 7. Verified by hashing both opening hands at
  each game's first recorded decision across all 600 games.
- **The entire morning judging battery** (gate, all matchups, both
  assessments, the degeneracy probe) was deal-narrow: one deal per worker
  process, 1-2 deals per run.

The fix was inherited by merging `origin/main` (which also brought the
stage-3b observation change `a9f1f91`: 13 effective-keyword flags on
permanents, hexproof on cards, permanent cap 30→40 — a second world break
inside one experiment). Post-merge verification: 40/40 distinct deals over
`reset(seed=1..40)`, and 120/120 distinct deals through the exact
`play_games` path and seed arithmetic the gate uses (`Env(seed=0)`,
`reset(seed=game_index)`); same-seed resets reproduce the same deal.

The checkpoints were ported to the stage-3b encoding by appending
**zero-weight input columns** for the new features and remapping the moved
validity column — unlike the Deviation-1 case, these features are appended
at known offsets, and a zero column makes the adapted network compute
exactly the original function on the features it was trained on (the value
logit shifts only through the changed mean-pool denominator from the raised
permanent cap; ordering is preserved). `student_bc_adapted.pt` and
`value_full_adapted.pt` are those ports; the adapted V was smoke-tested
against a live state before any run.

Every judging cell was then re-run deal-diverse. Side by side:

| cell | deal-narrow (invalid) | **deal-diverse (the record)** |
|---|---|---|
| **THE GATE: value-search-64 vs V-greedy (400g)** | 57.5% [52.6, 62.3] | **60.25% [55.4, 64.9]** |
| gate per-seat (play/draw) | 19.0% / 96.0% | 48.5% / 72.0% |
| V assessment: Spearman V ~ rollout gt | 0.646 (830 states) | **0.485 (825 states)** |
| V-greedy vs random (200g) | 69.5% [62.8, 75.5] | **84.0% [78.3, 88.4]** |
| V-greedy vs student_bc (200g) | 67.5% [60.7, 73.6] | **53.0% [46.1, 59.8]** |
| student_bc vs random (200g) | 86.5% [81.1, 90.6] | **78.0% [71.8, 83.2]** |
| value-search-64 vs search-16 (200g) | 28.5% [22.7, 35.1] | **38.5% [32.0, 45.4]** |
| value-search-64 vs search-64 (200g) | 22.5% [17.3, 28.8] | **23.5% [18.2, 29.8]** |
| degeneracy probe (agreement w/ V-greedy) | 74.8% (1,582 dec) | **79.5% (1,398 dec)** |

Three morning "findings" dissolve under deal diversity: the violent 19/96
gate seat split (→ 48.5/72), the non-transitivity story (V-greedy "decisively
beating" its init student 67.5% while scoring worse vs random — deal-diverse
they are statistical peers at 53.0%), and the student's "frontier
reproduction" (86.5% → 78.0%). One finding **strengthens**: the gate itself,
from marginal 57.5% to 60.25% with the CI lower bound (55.4%) clear of the
pre-registered 55% bar. All numbers below are the deal-diverse record unless
explicitly marked deal-narrow.

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
**78.0% [71.8, 83.2] vs random (200g, deal-diverse)**. That is below
exp-07's 600-game R0 frontier (87.0%, and exp-11 re-validated its ported
sibling at 86.5% on the fixed harness) — expected for a student trained on
37% of the data and, per Deviation 3, on only **three opening deals**.

Value variants (10 epochs, lr 1e-3, batch 512, val split by game):

| variant | val BCE | val Brier | val acc (outcome) |
|---|---:|---:|---:|
| **full fine-tune from BC init (selected)** | **0.5512** | **0.1762** | **0.759** |
| frozen encoder, value head only | 0.6243 | 0.2162 | 0.666 |

Fitting only the value head on top of BC features costs ~9 points of
held-out outcome accuracy — the policy-distillation encoder does not already
linearly contain the win-probability signal. Train BCE 0.330 vs val 0.551
shows real overfitting room at 225 games; more data was the plan. Note the
val split is by *game*, not by *deal*: with 3 deals in the corpus, held-out
games share deals with training games, so 0.759 overstates generalization —
the deal-diverse assessment below is the honest generalization number.

## Task 2 — Assessing V (wave README protocol)

825 fresh held-out states from 60 games of BC-student self-play (deal-diverse,
disjoint from training games by construction), ground truth per state = mean
over legal actions of 64 fresh uniformly-random playouts per action
(`Env.flat_mc_scores(16,4)` — the unbiased P(win|s) instrument, ≥64 playouts
per action). V is read at the same state from the acting player's
perspective.

| metric | deal-diverse (record) | deal-narrow |
|---|---:|---:|
| **Spearman V ~ rollout gt** | **0.485** | 0.646 |
| Pearson | 0.498 | 0.638 |
| Spearman V ~ actual game outcome | 0.278 | 0.362 |
| mean bias (V − gt) | −0.167 | −0.160 |
| MAE | 0.288 | 0.258 |

**P1's Spearman threshold (≥0.6) fails on deal-diverse states: 0.485.** The
morning 0.646 was measured on states from the same narrow deal family V
trained on; averaged over fresh deals, a third of the ordering signal
evaporates. The deal-narrow instrument itself was high-variance *by deal*:
its two sub-batches (different constructor seeds → different single deals)
scored Spearman 0.437 and 0.747. Calibration is monotone but compressed and
globally pessimistic (mean V 0.38 vs mean gt 0.55; in V's most pessimistic
bin, V<0.1, n=286, the random-rollout truth is 0.44). Part of the level gap
is instrument mismatch rather than pure error — V estimates P(win | both
continue as search-256), gt measures P(win | both continue uniformly random)
— but ordering, which is what search consumes, has no such excuse.

### Per-bucket bias (the aggro-bias tripwire) — reported prominently

Buckets: board advantage sign (sum of P+T on battlefield, mine − theirs) x
holding interaction (Lightning Bolt / Counterspell in hand). Deal-diverse:

| bucket | n | mean V | mean gt | bias | MAE | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| ahead / holding | 101 | 0.482 | 0.621 | −0.139 | 0.301 | 0.479 |
| ahead / no-instants | 243 | 0.565 | 0.697 | −0.131 | 0.258 | 0.550 |
| even / holding | 131 | 0.313 | 0.515 | −0.202 | 0.320 | 0.319 |
| even / no-instants | 136 | 0.336 | 0.515 | −0.179 | 0.324 | **0.152** |
| behind / holding | 109 | 0.164 | 0.389 | −0.224 | 0.285 | **0.122** |
| behind / no-instants | 105 | 0.220 | 0.374 | −0.154 | 0.263 | 0.394 |

The tripwire fires, but not with P1's predicted sign pattern. V is
pessimistic *everywhere* relative to random-rollout truth — there is no
optimism in board-ahead states — and ordering degrades from usable at the
extremes (ahead: 0.48-0.55) to near-noise in the middle of the game
(even/no-instants 0.15, behind/holding 0.12, even/holding 0.32). The
deal-narrow assessment had localized the collapse specifically to
even-board/holding-interaction states (Spearman 0.23 there vs 0.53-0.70
elsewhere) — the sharper deal-diverse statement is that **V's ordering is
weak everywhere the game is not already decided**, interaction states
included. Both batteries agree on the headline: the value function is
close-to-uninformative precisely where decisions matter, another appearance
of the project's structural bias — cheap approximations cannot represent
plans, and a V distilled from determinized-search outcomes inherits it.

## Task 3 — V-greedy

`VGreedyPlayer`: argmax over legal actions of V at the state after applying
the action on one determinized clone (hero perspective, non-hero-actor
leaves flipped 1−V). Judged (200g each, seat-balanced, Wilson 95%,
deal-diverse):

| matchup | V-greedy win rate |
|---|---|
| vs random | **84.0% [78.3, 88.4]** |
| vs student_bc | 53.0% [46.1, 59.8] |

One-step V-argmax is a real policy — decisively above random, ~6 points
above its init student against random (84.0 vs 78.0, overlapping CIs) and a
statistical coin-flip against it head-to-head. The morning battery's
dramatic non-transitivity (V-greedy 67.5% over the student head-to-head
while scoring 17 points *below* it vs random) was a single-deal artifact and
is withdrawn. Cost: 2.6-5.8 ms/decision on CPU depending on box load.

## Task 4 — The gate

Config: `ValueSearchPlayer`, N=64 determinized worlds x 1-step-then-V
(depth 0, rollouts 1) — the pre-registered primary. Degeneracy check before
running: at N=64/d0, value-search agrees with V-greedy on only **79.5%** of
1,398 nontrivial decisions (mean score gap when disagreeing 0.130), so
1-step-then-V with world-averaging is a genuinely distinct policy and no
depth extension was needed. What N=64 buys over V-greedy is exactly the
hidden-information average: 64 determinizations of the opponent's hand and
both libraries, with common random numbers across actions, versus V-greedy's
single sample.

**THE GATE (400g, seat-balanced, Wilson 95%, deal-diverse):**

| matchup | win rate | on play | on draw |
|---|---|---|---|
| **value-search-64 vs V-greedy** | **60.25% [55.4, 64.9]** | 48.5% | 72.0% |

**GATE VERDICT: PASS.** The wave's gate condition — search-with-V beats
V-greedy, >50% — is met with the CI clear of 50%, and P2's stricter
pre-registered bar (>55%) is also cleared, point estimate *and* CI lower
bound (55.4%). Search on top of this V is a policy improvement operator —
with only 37% of the intended training data behind V, and that data spanning
only 3 opening deals. That last fact makes the pass more interesting, not
less: on fresh deals V is far off its training distribution (Spearman 0.485,
collapsing in mid-game buckets), and averaging V over 64 determinizations
plus one step of lookahead is worth *more* there than it was on-distribution
(60.25% vs the deal-narrow 57.5%). Search is compensating for V's noise,
which is exactly the property the gate was designed to detect.

The per-seat split (48.5% on the play, 72.0% on the draw) still leans
draw-side, but mildly; the morning battery's violent 19/96 split was a
single-deal artifact (documented in Deviation 3, kept for the record).

### Economics (P3)

Measured decision costs in this world, from the deal-diverse runs (shared
box; treat as upper bounds, ratios trustworthy):

| player | ms/decision | context |
|---|---:|---|
| V-greedy | 2.6 | gate villain, CPU |
| search-16 (random rollouts) | 21.0 | ladder villain, 2 workers (14.7 in the uncontended deal-narrow run) |
| value-search-64 (MPS) | 25.6 | equal-sims hero, 1 worker (30.6 in the gate; 70.2 CPU-contended at 2 workers) |
| search-64 (random rollouts) | 72.3 | equal-sims villain, 1 worker |

The speed half of P3 is real but modest: at N=64, replacing every random
playout with a V forward is ~2.8x cheaper per decision (25.6 vs 72.3
ms/dec), nowhere near the ~120x playout-length ratio the prediction leaned
on — random playouts run inside the Rust engine at engine speed while leaf
values cross the Python/MPS boundary per batch. That 2.8x puts
value-search-64's honest equal-wall-clock rung at roughly **search-16**
(25.6 vs 21 ms/dec), not search-64.

Head-to-head at the honest comparison points (200g each, deal-diverse):

| matchup | value-search-64 win rate |
|---|---|
| vs search-16 (~wall-clock parity) | **38.5% [32.0, 45.4]** |
| vs search-64 (villain spends ~2.8x the hero's wall-clock) | **23.5% [18.2, 29.8]** |

**P3 VERDICT: REFUTED.** At wall-clock parity value-search loses 38.5-61.5
to random-rollout search-16, and giving the random-rollout side a 2.8x
budget advantage (search-64) only widens it to 23.5-76.5. There is no
measured N at which V-at-leaves beats random playouts at equal wall-clock in
this world. The two facts compose into one sentence: **search improves V
(the gate passes), but V is not a substitute for rollouts (economics fail)**
— the speedup per leaf is real, and it is entirely eaten by V's quality
(Task 2: ordering near-noise in undecided positions), not by the search
wrapped around it.

### Ladder placement

Anchors, deal-diverse: value-search-64 sits **below search-16** on the
ladder (38.5% against it) at matched wall-clock;
V-greedy and the BC student sit in the same band as each other (84.0% and
78.0% vs random; 53.0% head-to-head), both far below any searcher. For
calibration, exp-02's ladder (deal-narrow era, different world) had
search-16 at 91.7% vs random; no policy or value construct in this
experiment threatens the random-rollout ladder at any rung.

## Prediction verdicts

| prediction | verdict | number |
|---|---|---|
| **P1** Spearman ≥0.6 + per-bucket bias (aggro pattern) | **REFUTED on the threshold, deal-diverse** (deal-narrow 0.646 met it, but on V's own deal family); bias measurable but wrong pattern | Spearman 0.485; pessimistic in *all* buckets (−0.13 to −0.22, no board-ahead optimism); ordering near-noise in undecided mid-game buckets (0.12-0.32) |
| **P2 (THE GATE)** search-with-V > V-greedy >55% | **CONFIRMED** | **60.25% [55.4, 64.9]** over 400g deal-diverse; CI clear of both the 50% gate bar and the 55% pre-registered bar |
| **P3** value-search beats random-rollout search at equal wall-clock at some N | **REFUTED** | 38.5% [32.0, 45.4] vs search-16 at ~wall-clock parity; 23.5% [18.2, 29.8] vs search-64 (which spends ~2.8x more); leaf speedup real (~2.8x/leaf-set) but eaten by V quality |

## Cost ledger

Exp-00 basis, $1.006/hr wall-clock. The 6 h cap was exceeded; the honest
breakdown:

| item | wall-clock |
|---|---:|
| datagen, 3 attempts (0.75 h productive; the rest a dead overnight run and supervision gaps — Deviation 2) | ~10 h elapsed, ~1.5 h compute |
| training: BC + 2 value variants + smoke | ~0.6 h |
| deal-narrow battery (gate 400g, 5 matchups, 2 assessments, probe) — invalidated | ~1.3 h |
| deal-diverse battery (gate 400g, 5 matchups, assessment, probe) | ~0.9 h |
| equal-sims re-run + deal verification (successor session) | ~0.6 h |
| **total booked (elapsed, incl. dead time)** | **~13.5 h ≈ $13.6** |
| of which productive compute | ~5 h ≈ $5 |

Two structural lessons are in Deviation 2 (monitor liveness, not artifacts)
and Deviation 3 (protect the instrument: a worktree that predates an
instrument fix silently re-imports the bug; ~1.3 h of measurement was spent
on a battery that had to be discarded).

## Caveats

### Deal diversity (the instrument bug of record) — verified

Per exp-06/A5, `Env.reset(seed=)` historically never reached the engine.
Status of every ingredient of this experiment, verified by hashing both
opening hands per game:

- **Training shards (`dataset_v0`, `prelim`)**: NOT diverse — generated
  pre-fix on the pooled driver path, one constructor-seeded `Env` per
  worker. 225 training games = **3 distinct deals** (600 games on disk = 7).
  This contaminates *training*, and cannot be fixed without regenerating
  data (out of budget; see next-question).
- **Deal-narrow battery**: NOT diverse (1-2 deals per run); all its numbers
  demoted to the Deviation-3 table.
- **Deal-diverse battery (the record)**: verified diverse twice — 40/40
  distinct deals through `Env.reset(seed=1..40)` post-merge, and **120/120
  distinct deals through the exact `play_games` code path and seed
  arithmetic the gate used** (constructor seed 0, `reset(seed=game_index)`),
  with same-seed reproducibility confirmed. The assessment states (60
  behavior self-play games) go through the same per-game reseed path.

Consequence for interpretation: the gate (60.25%) is a claim about a V
trained on 3 deals and evaluated on ~400 fresh ones. It passed *despite*
that handicap. The 68.9% seat-0 training self-play figure is a 3-deal
artifact and is not quotable as a game fact.

### Others

- **37% of intended data, 3 deals**: V saw 27,423 decisions from 225 games
  on 3 opening deals. Every V-quality number (Spearman 0.485, bucket
  collapse, economics losses) is a floor for the *recipe*, not a ceiling
  for the *approach* — but the gate's pass means more data would be spent
  answering a question (can V replace rollouts?) that Task 2 and P3 already
  answer negatively at this data scale.
- **Single training seed** (loop discipline #5): one BC init, one V
  fine-tune. The gate CI quantifies eval noise of this checkpoint, not
  method variance across seeds.
- **Adapted checkpoints**: the stage-3b port (Deviation 3) preserves the
  trained function exactly on old features (zero-weight new columns), but
  the judged players therefore *ignore* keyword visibility that natively
  trained stage-3b agents would see. Fair within this experiment (all
  ported players share the blindness; searchers don't use nets); not
  comparable against future natively-trained checkpoints.
- **Shared box**: absolute ms/dec are upper bounds; all cost claims are
  ratios measured under comparable contention except where noted.
- **World-relative**: all numbers are stage-3b world (post `a9f1f91`);
  exp-02/07 ladder anchors quoted for orientation only are from earlier
  worlds and are not directly comparable (WORLDS.md discipline).

## Next question (C11)

**Recommendation: do not run a value-crank C11** (more data, bigger V,
deeper value-search). The gate passed, so the operator exists; but P3's
refutation plus Task 2's diagnosis (ordering near-noise exactly in undecided
positions, trained from determinized-search outcomes) says the binding
constraint is *what V can represent and what it was trained on*, not how
much data it got. Feeding the crank now would repeat exp-07's mistake in
value clothing: optimizing a component before diagnosing which deficit —
planning, beliefs, or information-set consistency — actually binds.

The wave has already been redesigned around exactly this question
(`wave/intelligence/README.md`, diagnose-then-treat): **C11 should be D1,
the 2x2 ablation** ({hidden, full information} x {random, oracle/exact
continuation}) on competency scenarios and micro-format matchups, with D2
(the exactly solvable microgame) behind it. Exp-10 feeds D1 directly: if the
full-info column with exact continuation still fails, no V was ever going to
save determinized search; if hidden-info with oracle beliefs recovers it,
the treatment is belief-shaped (02-beliefs-design.md), and a *belief-
conditioned* value head — not this scalar rung-1 V — is the version of
goal 4 worth training. Deal-diverse regeneration of the training corpus is a
prerequisite for whichever treatment D1 selects, not a cycle of its own.

## Provenance

- **Code** (this branch): `manabot/sim/value.py` (train_value, VGreedyPlayer,
  ValueSearchPlayer, assessment + agreement instruments),
  `manabot/sim/flat_mc.py` (play_games seat-balanced driver),
  `managym/src/agent/rollout_pool.rs` + vector-env bindings (leaf batching),
  runner `scripts/exp10_value_gate.py` (subcommands bc / train-value /
  assess / agree / match), tests `tests/sim/test_value.py`.
- **Checkpoints** (gitignored, `.runs/exp10/`): `prelim/bc.pt` → re-saved as
  `student_bc.pt`; `value_full.pt` (selected; provenance embedded: shards=
  `prelim/`, init=`prelim/bc.pt`, freeze_encoder=False); `value_frozen.pt`
  (rejected variant); `student_bc_adapted.pt` / `value_full_adapted.pt`
  (stage-3b zero-column ports used by the deal-diverse battery).
- **Data**: `experiments/data/exp-10-value-gate.json` — consolidated: both
  batteries' matchup metrics, both assessments (incl. buckets), degeneracy
  probes, deal-diversity verification counts, training curves. Raw per-run
  JSONs and logs in `.runs/exp10/` (deal-narrow at top level, deal-diverse
  under `dealfix/`); assessment state dumps as `.npz` alongside.
- **Training corpus**: `.runs/exp10/dataset_v0/` shards 00-06 (600g, 7
  deals); V trained on `prelim/` = shards 01-03 (225g, 3 deals). Provenance
  strings embedded per shard (teacher search-256, round 0).
- **Instrument checks**: deal-hash verifications reported here were run
  2026-07-10 (pre-fix Env probe: 1 deal / 5 seeds; post-fix: 40/40; gate
  path: 120/120; shard hashing: 3/225, 7/600). Deal-narrow battery ran
  pre-merge at `39d0a1a`; deal-diverse battery ran post-merge (merge commit
  `197d1e5`, includes seed fix `d4a5ebe` and stage-3b `a9f1f91`).
