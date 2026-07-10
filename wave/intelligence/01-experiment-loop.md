# The Experiment Loop

The wave's goals (README) are structure. This is the executable plan: a loop of
cheap, pre-registered experiments, each one answering a question that changes
what we do next. Infrastructure is pulled by experiments, never pushed ahead of
them.

## North star

**Impressive intelligence for minimal training cost**, operationalized as a
chart:

- **y — ladder strength**: the largest N such that the policy alone (no search
  at inference) beats determinized search-at-N rollouts, interpolated. Sanity
  check: exploitability probe (a from-scratch agent trained against the frozen
  policy must not exceed ~65% win rate, or the ladder number is a lie).
- **x — cumulative training cost**: dollars (local runs priced at market
  GPU-hour rates for honesty) and GPU-hours, log scale.

Every cycle adds a point or improves the instrument that measures one.

**"Wow" milestone, defined in advance:** policy-alone beats search-at-1000 on
an interactive deck, total training spend < $100. Pluribus-shaped on purpose —
in imperfect-information games, algorithmic choices historically beat compute,
and the budget cap is the claim, not a limitation.

## Protocol (per cycle)

1. **Question** — one sentence.
2. **Prediction** — a number, written *before* the run, recorded in the report.
   Unregistered beliefs rot into folklore (see: "~200 steps/game", the
   auto-pass claim).
3. **Cost cap** — set in advance. Exceeding it means the experiment is
   redesigned, not extended.
4. **Run** — through the verify harness; provenance in the store.
5. **Report** — `experiments/exp-NN-<name>.md`: prediction vs. result, belief
   update, and the next cycle's question. The report is not done until it names
   the next question.
6. **Chart** — update the strength-vs-cost ledger.

Global rules: every number gets a CI; every claim in a doc links a report; kill
criteria written before running; one cycle in flight.

## Cycles

### C0 — Calibrate the instrument

**Q:** What is the real decision horizon, and what does a training step cost?

No one has measured either. All credit-assignment arguments rest on an
unsourced "~200 steps"; strength-per-dollar has no denominator yet.

- **E0a** Decision profile: expose `skip_trivial_count` via PyO3; log surfaced
  decisions/game by `ActionSpaceKind` (priority / attacker / blocker / target)
  and the collapse ratio. (Wave goal 2.)
- **E0b** Cost basis: parameter count, FLOPs/step, measured $/1M steps on
  local hardware and on the g5.xlarge spec.
- **E0c** Baseline matrix: random-vs-random, untrained-vs-random,
  untrained-vs-passive with CIs, on the current deck — the last clean numbers
  before C1 changes the game.

**Prediction:** surfaced decisions/game (both players) lands in 40–120, well
under the folklore 200; the hero's share under 60. If so, the effective GAE
horizon (~17 steps at γλ=0.9405) is within ~3x of the real horizon and the
"plumbing, not strategy" story weakens — worth knowing before betting on it.
**Cost cap:** zero GPU-hours; a day of coding.

### C1 — Does the game contain strategy?

**Q:** When the deck can interact, does anything first-light concluded survive?

Add `DrawCards` and mass-removal effects to the engine (the only missing
mechanics — `state/ability.rs:20-36` has four `Effect` variants); register a
draw spell and a wipe; build an interactive deck (creatures + Bolt +
Counterspell + draw + wipe) for both players. Re-run the first-light dev
preset *unchanged*. (Wave goal 0.)

**Predictions:** untrained-vs-random win rate drops below 65% (dynamic range
restored from the current 82%-vs-ceiling); the shaped agent shows a measurable
aggro fingerprint (near-zero instant-holding: casts instants at first legal
opportunity ≥80% of the time); surfaced decisions/game rises ≥1.5x vs. E0a
(response windows open — `skip_trivial` cannot absorb them).
**Cost cap:** one dev-preset run (~262k steps) + the Rust work. Local.

**RESULT (2026-07-09):** see `experiments/exp-01-c1-training.md` (index: `experiments/README.md`).

### C2 — Is bias-free dense signal possible?

**Q:** Does potential-based shaping learn without the aggro fingerprint?

Ng, Harada & Russell (1999): shaping of the form Φ(s′)−Φ(s) is provably
policy-invariant. Current shaping (pay-per-land, pay-per-damage) is not, which
is exactly why it installs a strategic prior. Three dev runs on the C1 deck:

- **E2a** terminal-only — expect pass-collapse to reproduce (first time on an
  interactive deck).
- **E2b** current shaping — expect learning *with* the aggro fingerprint.
- **E2c** potential-based Φ over board state — the experiment. Expect learning
  *without* the fingerprint.

**Prediction:** E2c reaches ≥90% of E2b's win-vs-random while holding instants
significantly longer than E2b (fingerprint metric from C1). If E2c works,
dense signal no longer costs a strategic prior — goal 5's "delete shaping"
becomes "replace with Φ," and aux heads stay dead.
**Cost cap:** 3 dev runs. Local.

**RESULT (2026-07-09):** see `experiments/exp-04-potential-shaping.md` (index: `experiments/README.md`).

### C3 — How much intelligence is free?

**Q:** What does zero-training intelligence look like on this engine?

Flat determinized Monte Carlo with **random rollouts**: sample worlds
consistent with the observation (opponent hand from remaining deck), roll out
to terminal, pick the best action. No network, no GPU, no training — pure
engine throughput. Play search-at-{16, 64, 256} against every policy produced
by C1/C2 and against each other. (Wave goal 3, first half.)

**Predictions:** search-at-64 with random rollouts beats every trained policy
to date (>60% win rate); search-at-N strength is monotone in N over this
range. This is the first point on the north-star chart: **strength at $0
training cost.** Every future training run is judged against it.
**Cost cap:** CPU only; engine-time budget ~1 hr per matchup pair.
**Kill criterion:** if search-at-256 does *not* beat the trained policies,
determinization or the rollout policy is broken — stop and diagnose before any
distillation work.

**RESULT (2026-07-09):** see `experiments/exp-02-flat-mc.md` (index: `experiments/README.md`).

### C4 — Is distillation cheaper than RL?

**Q:** Per dollar, does supervised learning from search beat PPO from scratch?

Generate a dataset of C3 search decisions; behavior-clone a fresh policy on
it; compare against PPO trained to *matched total cost* (search generation +
BC training vs. PPO wall-clock, same hardware pricing).

**Prediction:** BC-from-search matches or beats matched-cost PPO at ≤1/5 the
cost to reach PPO's final strength. This experiment is the thesis in
miniature — if it fails, "search as teacher" (goal 5) needs rethinking before
scaling.
**Cost cap:** dataset generation ≤ 24 engine-hours; BC is minutes.

**RESULT (2026-07-09):** see `experiments/exp-03-distillation.md` (index: `experiments/README.md`).

### C9 — Can the pilot play control? (2026-07-09)

**Q:** Is exp-08's UR-at-22% a deck property (H1) or a pilot property (H2 —
flat MC with random rollouts cannot play control: strategy fusion never holds
interaction for value; random rollouts burn inherited counterspells on the
first target)?

Two instruments (`manabot/verify/competency.py`, engine state-injection
surface `managym/src/flow/scenario.rs`):

1. **Competency scenarios** — five constructed positions with documented
   known-correct lines (counter-the-bomb, hold-the-wipe, bolt-the-threat,
   race-vs-block, hold-up-quench), scored per decision against scripted
   villains, ≥100 runs × {random, search-16/64/256}, Wilson CIs. This is a
   permanent instrument: every future policy gets a tactics score.
2. **Micro-format mirrors** — MICRO_AGGRO vs MICRO_CONTROL (≤6 names each),
   seat-balanced 300 games/cell, mirrors ≈50% sanity, cross-matchup at
   N ∈ {16, 64, 256} with behavioral probes (what counters countered, what
   bolts targeted, instant-holding rate).

**Predictions** (registered in `experiments/exp-09-control-competency.md` before
the runs): under H2 the scenario correct-line rates are flat in N (Δ < 0.15
from N=16 to N=256), never exceed 0.50, and random ≥ search-16 on
hold-the-wipe; control-vs-aggro is flat-to-declining in N (~0.35) while
counter_first_window_rate stays > 0.70 and instant_holding_rate < 0.15.
**Cost cap:** CPU only, ≤ 4 workers (shared machine), ~6 core-hours.

**RESULT (2026-07-09): H2 confirmed at the decision level; exp-08's 22% is
a pilot artifact, not a deck measurement.** All three registered scenario
claims held, mostly at ceiling: max correct-line rate 0.39 at any N,
Δ(256−16) ≤ 0.02 on every scenario, and uniform random *beats* every
search strength on hold-the-wipe (0.23 vs 0.00 — search casts Pyroclasm on
turn 1 in 300/300 runs), race-vs-block, and hold-up-quench; the bolt is
burned before the key threat even appears in 100% of S3 runs (0/400 ever
killed the Lieutenant). The micro win-rate claim was refuted,
instructively: control-vs-aggro goes 0.037 (random) → 0.393 (N=16) →
0.630 (N=64) → 0.530 (N=256) — search rescues the *win rate* while the
behavior probes show it never plays control (instant-holding flat ~0.40 and
below random's 0.57 at every N; a quarter of Counterspells burned on 1-MV
goblins at N=256; what rises with N is only within-decision target quality,
0.34 → 0.69). Mirrors sanity at 0.490/0.517. N buys discrimination inside
the present decision, none across turns. The scenario suite
(`manabot/verify/competency.py`, ~4 min for 2,000 scored runs) is now a
standing gate: any pilot/policy that cannot lift S2/S5 off zero is not a
control player, whatever the ladder says. Feeds Exit 1 (belief-based /
root-level information-set handling) and the C5 policy-rollout question.
See `experiments/exp-09-control-competency.md`.

### C5+ — The loop proper

Policy-rollout search (first time batched inference — wave goal 1 — is
*pulled*: policy rollouts are blocked on it), then expert iteration:
search-with-current-policy generates targets → distill → stronger policy →
stronger search. Each iteration adds a chart point. The goal-4 gate
(search-with-V beats V-greedy) sits between C5 and any value-guided search.
Ladder strength and the exploitability check ride every iteration.

### C6 — Does the expanded world change the answers?

**Q:** The rules waves grow the observation/action space under every policy
we train; at what point does representation growth itself become a training
cost?

- **C6a** Same game, bigger representation: rules Stage 1+2 dims (CARD_DIM
  37, PERMANENT_DIM 7, PLAYER_DIM 27, ACTION_TYPE_DIM 14, mid-resolution
  decision kinds) on the C1 deck, exp-04's E2a recipe unchanged
  (terminal-only, dev preset, 3 seeds). Pre-registered: seeds land in E2a's
  60–75% band → benign; all <55% → adverse (suspects: action-type one-hot
  dilution, card-feature dilution).
- **C6b** Pool expansion on the real matchup: UR Lessons vs GW Allies (the
  two-deck slice), first training exposure of the Stage-2 decision kinds
  (scry / look-and-select / pay-or-not / modal / learn / waterbend) and of
  an asymmetric matchup. Queued on rules stage 3.

**RESULT (2026-07-09, C6a):** benign — **confirmed**. Training-path smoke
clean (200 random-vs-random games through the vector env, zero
crashes/NaNs/reward violations); 3 seeds judged seat-balanced 400g vs
random: **69.3 / 77.3 / 73.5%** (E2a control 75.5 / 64.5 / 60.0%;
deal-averaged re-judge 77.3 / 70.3 / 66.5%, LBs ≥ .617), same
patience-shaped behavioral signature, no seed near the adverse tripwire.
Params 100,354 → 101,506 (+1.1%); training SPS 2,472 → 3,013 (+22%) on the
exp-03 calibration recipe. Unregistered yield: **the judging harness had
been evaluating one deal per run since exp-00** — `Env.reset(seed=...)`
never reached the engine, so every historical 400-game eval was 400
stochastic rollouts of a single deal, and exp-00c's per-seat random
baselines (23.1% on-play interactive, 93.4% on-play standard) are deal
artifacts; deal-averaged, random mirror play is near seat-parity (~48.6%
on-play, 3000 games). Fixed (Python-only) + regression-tested; judging is
deal-averaged from now on. See `reports/exp-06-newworld-training.md`.

### C7 — Expert iteration lands in the new world

**Q:** Does one full crank of the closed loop (distill → student becomes
rollout policy → stronger teacher → re-distill) climb the ladder — and does
batched inference finally unblock it?

First cycle on the post-stage-2 world (CARD_DIM 37; all prior checkpoints
dimensionally dead). Pre-registered predictions (verbatim in
`experiments/exp-07-expert-iteration.md`): P1 batched inference ≥10x (2k → ≥20k
obs/sec); P2 policy-rollout search beats random-rollout search at equal
wall-clock (>55%); P3 the R1 student beats the R0 student head-to-head
(>55%) and places ≥ N=16.

**RESULT (2026-07-09):** see `experiments/exp-07-expert-iteration.md` (index: `experiments/README.md`).

## Protocol amendments

Amendments are allowed; silent amendments are not. Each is dated and lands
*before* the experiment it affects runs.

### A1 — Seat balancing (2026-07-09, after C0)

C0 measured random-vs-random on STANDARD_DECK at **93.4% for the on-the-play
player**. Every historical hero-on-the-play "vs random" win rate (including
first-light's 82% baseline and 100% final) is seat-contaminated. Henceforth:
all evaluations are seat-balanced (50/50) and report per-seat win rates
separately. C1's original "untrained-vs-random < 65%" prediction is untestable
as stated and is withdrawn.

### A2 — Multi-init baselines (2026-07-09, after C0)

Three fresh untrained inits scored 13% / 68% / 89% vs random (non-overlapping
CIs). Single-init untrained baselines are meaningless. Henceforth: untrained
baselines use ≥3 inits (5 preferred), reported as a spread.

### A3 — C1 environment predictions, restated (2026-07-09, before the C1
training run; original total-decisions prediction likely refuted)

Registered before the seat-balanced interactive-deck measurement runs:

1. Seat-balanced random-vs-random on INTERACTIVE_DECK: on-the-play advantage
   drops below 80% (interaction punishes pure racing; was 93.4% on
   STANDARD_DECK).
2. Priority share of hero decisions rises above 50% (was ~38%; combat
   declarations were 58% of all surfaced decisions on STANDARD_DECK).
3. The original "total decisions/game rises ≥1.5x" prediction appears headed
   for refutation — C1 validation measured ~171 mean vs 194 on STANDARD_DECK.
   If confirmed refuted, the interesting quantity is the *mix shift*
   (combat-declaration spam replaced by priority decisions), not the total.

### A4 — skip_trivial folklore correction (2026-07-09, after C0)

The collapse ratio is **~12%**, not "nearly all off-turn windows." Zero
single-action leaks in ~150k decisions (the mechanism is sound), but the
volume story told in this wave's own strategy section overstated it. Combat
declaration, not priority passing, dominates surfaced decisions.

### A5 — Deal-averaged evaluation (2026-07-09, after C6a)

C6a found that `Env.reset(seed=...)` never reached the engine: every
evaluation run since exp-00 played all its games on a single deal
(`reports/exp-06-newworld-training.md`). Henceforth: evaluations vary the
deal per game (the fixed harness does this by default), and per-seat
win-rate splits from single-deal-era reports are not quotable as game facts
— including exp-00c's 23.1%/76.9% interactive-deck seat baselines and the
93.4% STANDARD_DECK figure cited in A1 (seat *balancing* remains required;
deal-averaged random mirror play measures near seat-parity, ~48.6% on-play
over 3000 games). Wilson CIs from that era are conditional on the run's
deal.

## Pre-registered exits

Written 2026-07-09, while genuinely agnostic. The point of writing them now is
that after months of investment, nobody is agnostic — these are the
solo-researcher's substitute for an advisor saying "drop it." Each exit names
its tripwire, the interpretation, and the pivot. Measured on the interactive
(C1) deck unless stated.

### Exit 1 — PIMC has plateaued → go belief-based

**Tripwire (both required):**
- two consecutive 10x increases in rollout count N yield < 2 points of
  win-rate gain on the ladder, **and**
- an exploiter trained against the frozen search-derived policy reaches ≥ 65%
  win rate.

Strength ceiling *plus* exploitability is the signature of strategy fusion
binding — determinization, not compute, is the constraint. Corroborating
evidence: bluff-rate ≈ 0, instant-holding ≈ 0, and the exploiter's wins
concentrated in represent/trick lines.

**Pivot:** belief-conditioned value function over public state (ReBeL-shaped,
scoped down). Explicitly not the fix: more rollouts, bigger trees. Full design
captured in **[wave/intelligence/02-beliefs-design.md](02-beliefs-design.md)** (dormant,
trigger-armed on this exit): PBS, update rule, ε-floor, canonical action IDs,
range-conditioned value head, off-model handling, pre-registered predictions.

### Exit 2 — Search isn't paying → model-free game-theoretic bet

**Tripwire (either):**
- C3's kill criterion fires (search-at-256 loses to trained policies) and
  diagnosis clears the determinizer and rollout policy of bugs, **or**
- two consecutive C5 expert-iteration rounds gain < 2 points on the ladder
  while matched-compute model-free PPO equals the distilled policy.

Interpretation: the game's effective horizon and stochasticity make lookahead
worthless at feasible N — the Go/poker intuition doesn't transfer.

**Pivot:** DeepNash-style model-free (R-NaD / NFSP) with population play. The
batched-inference investment carries over unchanged.

### Exit 3 — The game is degenerate → expand the pool minimally

**Tripwire:** on the C1 deck, search-at-16 ≈ search-at-256 (nothing to find by
looking deeper), or a ~20-line scripted heuristic matches search-at-256.

**Pivot:** add the minimum cards/mechanics that break the degeneracy; re-run
C1. Budget at most two such expansions before questioning the small-pool
approach itself.

### Exit 4 — Wrap the environment paper

**Tripwire (any):**
- 2027-04-01 arrives without a citable search-value result, or
- cumulative spend exceeds $2k without a new chart point, or
- two consecutive quarters with no completed cycle.

**Pivot:** write the environment paper with honest partial results and ship it.
This is the floor outcome and it is a *good* outcome — the env is citable
regardless of how the science went.

### Release bar (decided 2026-07-09)

The environment is publicized **when the first citable result exists** — the
wow milestone or the search-value curve — not before, and *not* gated on rules
completeness. Completeness is a tarpit with no finish line; a result is a
finish line. Quiet availability (public repo, no promotion) is fine earlier;
promotion rides with the result so the env and its headline demo land as one
unit.

## Goal ↔ cycle map

| Wave goal | Cycle |
| --- | --- |
| 0 real deck | C1 |
| 1 batched inference | **landed at C7** (2.0k → 24.5k obs/sec) |
| 2 decision profile | C0 |
| 3 determinized search | C3 (flat), C7 (policy rollouts — wall-clock negative) |
| 4 gate: assess V | C8 entry condition |
| 5 search as teacher | C4 (static, positive), C7 (iterated, negative) |
| 6 headline metric | the chart, every cycle |
