# exp-02: flat determinized Monte Carlo — how much intelligence is free?

**Cycle:** C3 (wave/intelligence/01-experiment-loop.md) · **Date:** 2026-07-09 ·
**Deck:** INTERACTIVE_DECK both players ·
**Training cost:** $0 (no network, no GPU; pure engine throughput).
**Total measurement cost:** 77.9M playouts, 4.5 CPU-core-hours, 42.5 min
wall on 8 workers of an M-series laptop.

## Question

What does zero-training intelligence look like on this engine? Flat
determinized Monte Carlo with uniformly-random rollouts, strength dialed by
sims-per-action N, played against random and against every trained policy
produced so far.

## Pre-registered predictions

1. **search-at-64 beats every trained policy to date** (>60% win rate,
   seat-balanced). The trained bar: C1v2 final checkpoints at 56.5% / 47.3% /
   57.5% seat-balanced vs random (exp-01; all three failed the exp-00c gate).
2. **Strength is monotone in N** over {16, 64, 256}.

**Kill criterion:** if search-at-256 does not beat the trained policies, the
determinizer or rollout policy is broken — stop and diagnose.

## Method

### Determinization

Decklists are known to both players (both run INTERACTIVE_DECK). Hidden
information is therefore exactly: the opponent's hand, and the order of both
libraries. One sampled world (`Game::determinize`, managym/src/flow/search.rs):

- the opponent's hand is replaced by a uniform draw of |hand| cards from
  their unseen pool — hand ∪ library (their graveyard / exile / battlefield /
  stack cards are public and never enter the pool);
- the remainder becomes their library, shuffled;
- the searcher's own library is reshuffled (contents known, order not).

All public state, including the current action space, is preserved (cargo
tests in managym/tests/search_tests.rs assert this, plus clone independence,
seed determinism, and playout termination).

### Search

At each hero decision point (`managym.Env.flat_mc_scores`, hot loop entirely
in Rust): for each legal action, over W determinized worlds × R random
playouts (N = W·R sims per action, R = 4), apply the action in a clone of the
world, play both sides uniformly-random-legal to terminal, score win = 1 /
loss = 0 / draw-or-cap = 0.5. Worlds are shared across actions (common random
numbers). Argmax. Playout step cap 2000 — **never hit: 0 cap-hits in 77.9M
playouts**; no draws either.

### Evaluation

Seat-balanced (hero alternates play/draw seats), 300 games per matchup
(150/seat), Wilson 95% intervals, per-seat rates — same protocol as
exp-00c/exp-01, run via `manabot/verify/run_flat_mc.py` (8 worker processes,
env reseeded per game). Trained opponents: the three C1v2 final checkpoints
(step_65536), loaded with their saved hypers, stochastic action selection —
identical to how their exp-01 numbers were measured. Random opponent: uniform
over valid encoded actions, identical to every historical vs-random baseline.

## Results

### 1. N-scaling vs random

| policy | overall | on play | on draw | ms/decision (mean/median) |
|---|---|---|---|---|
| search-16 | 91.7% [88.0, 94.3] | 84.7% [78.0, 89.6] | 98.7% [95.3, 99.6] | 7 / 6 |
| search-64 | 95.0% [91.9, 96.9] | 98.7% [95.3, 99.6] | 91.3% [85.7, 94.9] | 31 / 25 |
| search-256 | 99.0% [97.1, 99.7] | 98.7% [95.3, 99.6] | 99.3% [96.3, 99.9] | 113 / 92 |

Monotone: 91.7 → 95.0 → 99.0. Reference: random-vs-random on this deck wins
23.1% on the play (exp-00c); search-16 already lifts the disadvantaged seat
to 84.7%, and by N=64 the seat asymmetry is essentially erased.

### 2. Search vs trained policies (C1v2 finals; their vs-random strength 56.5% / 47.3% / 57.5%)

| | vs c1v2-s1 | vs c1v2-s2 | vs c1v2-s3 |
|---|---|---|---|
| search-16 | 92.7% [89.1, 95.1] | 89.0% [85.0, 92.1] | 84.0% [79.4, 87.7] |
| search-64 | 94.0% [90.7, 96.2] | 95.7% [92.7, 97.5] | 95.3% [92.3, 97.2] |
| search-256 | 98.7% [96.6, 99.5] | 98.0% [95.7, 99.1] | 97.0% [94.4, 98.4] |

Per-seat rates for all nine matchups are in
`reports/data/exp-02-flat-mc.json`; no matchup has a seat below 80.7%, and
search-256 is ≥96.7% in every seat of every matchup.

### 3. Monotonicity head-to-head

search-16 vs search-256 (search-16 as hero): **23.3%** [18.9, 28.4] — i.e.
search-256 wins 76.7% [71.6, 81.1] directly (30.7% for s16 on the play, 16.0%
on the draw). Strength is monotone not just against fixed opponents but in
direct play.

## Prediction verdicts

1. **CONFIRMED, understated.** search-64 beats every trained policy 94.0–95.7%
   (predicted >60%). Even search-16 — 7 ms and 16 sims per action — beats all
   three at 84.0–92.7%.
2. **CONFIRMED.** Monotone vs random (91.7/95.0/99.0), monotone against every
   checkpoint, and 256-over-16 head-to-head at 76.7%.
3. **Kill criterion: does not fire.** search-256 beats everything ≥97%.

Exit-3 tripwire (game degeneracy: search-16 ≈ search-256) also does not fire —
there is real depth between N=16 and N=256. The curve shape is monotone and
not yet saturating in log N against adaptive opposition (76.7% head-to-head
across a 16x sim gap), though it is compressing against random (ceiling).

## Economics (the $0 chart point)

| N | ms/decision (mean) | sims/decision (mean) | decisions/game | search cost/game |
|---|---|---|---|---|
| 16 | 7–8 | 44–48 | ~58–62 | ~0.45 s |
| 64 | 29–31 | 180–194 | ~50–60 | ~1.7 s |
| 256 | 113–163 | 741–784 | ~51–55 | ~6.5 s |

(sims/decision = N × mean legal actions ≈ N × 3; mean playout 0.21 ms ≈ ~38
engine steps — playouts from mid-game are short.)

The entire 13-matchup, 3,900-game experiment cost **4.5 CPU-core-hours, $0
GPU**. For the north-star chart: at $0 training cost, the engine already
contains a ~99%-vs-random, ~97–99%-vs-best-trained player at ~113 ms/decision
on one laptop core — and a 92%-vs-random one at 7 ms/decision. Every C1
training dollar spent so far ($1.30 cumulative) produced policies that
search-16 beats 84–93%. The ladder strength of every trained policy to date
is **below N=16** (no policy survives even the weakest searcher measured).

## Behavioral note: the searcher holds spells

Fingerprint probe (search vs random, seat-balanced; hero priority decisions
where a spell was castable):

| policy | cast_when_able | passed with spell available |
|---|---|---|
| search-16 | 0.64 | 0.30 |
| search-64 | 0.58 | 0.35 |
| search-256 | 0.52 | 0.41 |
| C1v2 seeds s1/s2 (exp-01) | 0.88–0.95 | 0.04–0.12 |

The searcher casts *less* eagerly as N grows — the opposite of the shaped
agents' race fingerprint. Deeper search increasingly declines to cast at the
first opportunity: holding Bolt/Counterspell mana through the opponent's turn
is frequently the argmax even under random rollouts, because worlds where the
held card answers a threat score well.

## Honest caveats

- **Strategy fusion.** This searcher maximizes the *average* over
  determinized worlds — it cannot make information-dependent plans, cannot
  value concealing information, and cannot bluff or represent a card it does
  not hold. Its "held instants" above are value plays visible in rollout
  averages, not deception; no behavioral evidence of information-hiding play
  was observed (none is representable). Against opponents that exploit
  information — none exist in this pool yet — PIMC's known pathologies
  (strategy fusion, non-locality) would bind. Exit 1's exploitability probe
  remains the instrument for that; nothing here measures it.
- **Random-rollout evaluation is deck-relative.** Uniform-random playouts
  happen to be informative on a 60-card aggro-interaction mirror; on decks
  where random play almost never executes the win condition (combo, heavy
  permission), flat MC with random rollouts could be arbitrarily weak. The
  N-scaling curve is a property of this deck as much as of the algorithm.
- **The trained opponents are weak.** Beating C1v2 checkpoints (best: 57.5%
  vs random, 0/3 passed the exp-00c gate) is a low bar; they were simply
  every trained policy in existence. The interesting comparison is C4:
  matched-cost distillation vs PPO.
- **Rollout opponent model is uniform-random for both sides.** The searcher
  implicitly assumes a random opponent; at these opponent strengths that is
  not yet a liability, but against strong opposition the rollout policy will
  misprice lines (classic PIMC bias). C5's policy rollouts are the fix.
- **Engine fizzle bugfix (below) means the eval environment ≠ the C1v2
  training environment** — judged a negligible confound but noted for
  provenance.

## Engine bugfix found on the way (pre-existing)

Random playouts immediately exposed a spell-fizzle bug
(`resolve_spell_object`, managym/src/flow/resolution.rs): a targeted spell
whose target was illegal at resolution called `counter_spell`, but the stack
object had already been popped, so the call no-opped and **the card was
stranded in the stack zone forever** — never reaching the graveyard, visible
as a phantom stack entry in observations for the rest of the game. Fixed per
CR 608.2b (a spell with no legal targets on resolution is put into its
owner's graveyard); the engine's `assert_stack_consistent` debug assertion
now holds under fully randomized play (it did not before). The C1v2
checkpoints were trained with the bug present; policies fizzle rarely, so
their measured strength is judged unaffected, but it is a real environment
change.

## Infrastructure added

- managym: `Game::determinize` / `Game::random_playout` / `Game::reseed`
  (src/flow/search.rs), `ZoneManager::resample_hidden` (src/state/zone.rs),
  `Env::fork` + `Env::flat_mc_scores` (src/agent/env.rs), PyO3 surface
  `clone_env` / `determinize` / `random_playout` / `flat_mc_scores` /
  `action_count` / `winner_index` / `is_game_over` / `current_agent_index`;
  8 cargo tests (tests/search_tests.rs).
- manabot: `manabot/sim/flat_mc.py` (FlatMCPlayer + seat-balanced two-player
  matchup loop + Wilson intervals), `manabot/verify/run_flat_mc.py`
  (multiprocess matrix runner, resumable JSON), `tests/sim/test_flat_mc.py`
  (11 tests).
- Data: `reports/data/exp-02-flat-mc.json` (per-matchup metrics with per-seat
  rates and CIs, search timing/sim counts, specs, wall/engine seconds).

## Next question (C4)

**Per dollar, does supervised learning from search beat PPO from scratch?**
Generate a dataset of search-at-256 decisions (~6.5 s of engine time per
game, so ~1k games / ~55k decisions ≈ 2 core-hours), behavior-clone a fresh
policy, and compare against PPO trained to matched total cost. The bar this
experiment sets: the BC policy should land in the search-16-to-64 band
(≥84% vs the C1v2 checkpoints, ≥90% vs random, seat-balanced) — anything
less means distillation is losing signal that 7 ms of search finds for free.
