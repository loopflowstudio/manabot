# exp-00c — Seat-balanced baselines (C0.5: E0c-v2 + C1 environment measurements)

Date: 2026-07-09

## Why this exists

C0 (`reports/exp-00-decision-profile.md`) found that random-vs-random on
STANDARD_DECK is **93.4%** for the on-the-play player, and that untrained
inits vary 13%–89% vs random — so every historical hero-on-the-play "vs
random" number is seat-contaminated, and single-init untrained baselines are
meaningless. Protocol amendments A1 (seat balancing) and A2 (multi-init
baselines) in `wave/search/01-experiment-loop.md` mandate the fix; amendment
A3 pre-registers the interactive-deck environment predictions scored below.

## Pre-registered predictions (quoted from amendment A3, registered before these runs)

> 1. Seat-balanced random-vs-random on INTERACTIVE_DECK: on-the-play
>    advantage drops below 80% (interaction punishes pure racing; was 93.4%
>    on STANDARD_DECK).
> 2. Priority share of hero decisions rises above 50% (was ~38%; combat
>    declarations were 58% of all surfaced decisions on STANDARD_DECK).
> 3. The original "total decisions/game rises ≥1.5x" prediction appears
>    headed for refutation — C1 validation measured ~171 mean vs 194 on
>    STANDARD_DECK. If confirmed refuted, the interesting quantity is the
>    *mix shift* (combat-declaration spam replaced by priority decisions),
>    not the total.

## Method

**Seat mechanism.** The engine hardcodes the starting player:
`managym/src/flow/setup.rs` constructs `TurnState::new(PlayerId(0))`, so the
first `PlayerConfig` passed to `Game::new` is always on the play. The engine
also implements the CR 103.7.1 first-turn draw skip keyed to `PlayerId(0)`
(`managym/src/flow/tick.rs:170-173`), so the player in seat 1 sees one extra
card. Seat balancing is therefore done at the Python eval level by swapping
which config is handed to the engine as player 0 — `Match.swapped()`
(`manabot/env/match.py`) — rather than adding a starting-player option to the
engine, which would also have had to touch the draw-skip rule. Observation
identity is unaffected: observations are perspective-relative (the `agent`
side is always the player to act), and policies are routed by
`raw_obs.agent.player_index`.

**Harness.** `manabot/verify/decision_profile.py` now takes
`--seat-balanced` (default on; even-indexed games put the hero in seat 0, on
the play; odd-indexed games in seat 1, on the draw), `--deck
{standard,interactive}`, and `--untrained-inits/--baseline-inits` for
multi-init baselines with logged torch seeds. Summaries report the hero win
rate overall, per seat, and the role-agnostic on-the-play win rate, all with
Wilson 95% intervals. The training-eval path
(`manabot/verify/util.py::run_evaluation`/`capture_evaluation`) gained the
same `seat_balanced` option with `win_rate_on_play` / `win_rate_on_draw`
metrics, for use by the C1 training run. Tests:
`tests/verify/test_seat_balance.py`, including an asymmetric-deck check that
the swap actually reaches the engine (the creature deck wins from whichever
seat it occupies, so the winning *player index* must flip with the seat).

**Policies** as in C0: `random` = uniform over valid encoded actions,
`untrained` = fresh `Agent` init (attention off, verify defaults, stochastic
sampling), `passive` = always pass. `skip_trivial=true` throughout. Decks:
STANDARD_DECK (12 Mountain, 12 Forest, 18 Llanowar Elves, 18 Grey Ogre; 36
creatures, zero interaction) and INTERACTIVE_DECK (12 Island, 12 Mountain, 6
Grey Ogre, 6 Wind Drake, 4 Man-o'-War, 4 Raging Goblin, 6 Lightning Bolt, 4
Counterspell, 3 Ancestral Recall, 3 Pyroclasm; 20 creatures + 16 interaction
spells), same deck both sides. Zero GPU-hours; zero aborted games anywhere.

## Results — Task 2: E0c-v2 on STANDARD_DECK (seat-balanced)

### random-vs-random, 1000 games (500 per seat)

| Metric | Value | Wilson 95% CI |
| --- | ---: | --- |
| Hero win rate (overall) | **50.3%** (503/1000) | [47.2%, 53.4%] |
| Hero win rate, seat 0 (on the play) | 94.4% (472/500) | [92.0%, 96.1%] |
| Hero win rate, seat 1 (on the draw) | 6.2% (31/500) | [4.4%, 8.7%] |
| On-the-play win rate (either role) | 94.1% (941/1000) | [92.5%, 95.4%] |

Sanity confirmed: seat-balanced random-vs-random is 50.3% ≈ 50%, and the raw
seat advantage replicates C0's 93.4% at 94.1% [92.5%, 95.4%].

### untrained-vs-random, 5 fresh inits × 400 games (200 per seat)

| Init (torch seed) | Overall [Wilson 95%] | Seat 0 (play) | Seat 1 (draw) | On-the-play wins (either role) |
| --- | --- | ---: | ---: | ---: |
| 100000 | 48.7% [43.9%, 53.6%] | 16.0% | 81.5% | 17.2% |
| 101000 | 50.0% [45.1%, 54.9%] | 1.5% | 98.5% | 1.5% |
| 102000 | 49.0% [44.1%, 53.9%] | 9.5% | 88.5% | 10.5% |
| 103000 | 51.7% [46.9%, 56.6%] | 20.0% | 83.5% | 18.2% |
| 104000 | 50.2% [45.4%, 55.1%] | 100.0% | 0.5% | 99.8% |

**Spread: overall 48.7%–51.7% (3.0 points; every CI covers 50%). Per-seat:
1.5%–100% on the play, 0.5%–98.5% on the draw.** Two findings:

1. **The C0 "13%–89% init spread" was a seat-by-init interaction, not a
   skill spread.** Once seats are balanced, all five inits are statistically
   indistinguishable from 50% — and from each other. Untrained-vs-random,
   measured correctly, is a coin flip for every init.
2. **The seat advantage is a property of the policy *pair*, direction
   included.** Under random-vs-random the on-the-play player wins 94%; put an
   untrained net in the game and for four of five inits the on-the-play
   player *loses* (on-the-play wins 1.5%–18.2% regardless of which role sits
   there), while the fifth init (104000) amplifies the racing advantage to
   99.8%. Untrained inits are biased policies whose game dynamics decide
   which seat's structural edge (attacking first vs. the extra card on the
   draw) dominates. A per-seat report is therefore not optional decoration:
   the overall 50% conceals a deterministic seat lottery.

### untrained-vs-passive, 3 fresh inits × 400 games (200 per seat)

| Init (torch seed) | Overall [Wilson 95%] | Seat 0 (play) | Seat 1 (draw) |
| --- | --- | ---: | ---: |
| 200000 | 100.0% [99.0%, 100%] | 100.0% | 100.0% |
| 201000 | 100.0% [99.0%, 100%] | 100.0% | 100.0% |
| 202000 | 100.0% [99.0%, 100%] | 100.0% | 100.0% |

Unchanged by seat and by init: any policy that ever attacks beats a policy
that never acts. The passive baseline is a floor detector, not a skill
measure.

## Results — Task 3: INTERACTIVE_DECK environment measurements

### Seat-balanced random-vs-random, 1000 games (500 per seat)

| Metric | Value | Wilson 95% CI |
| --- | ---: | --- |
| Hero win rate (overall) | 48.9% (489/1000) | [45.8%, 52.0%] |
| Hero win rate, seat 0 (on the play) | 22.0% (110/500) | [18.6%, 25.8%] |
| Hero win rate, seat 1 (on the draw) | 75.8% (379/500) | [71.9%, 79.3%] |
| **On-the-play win rate (either role)** | **23.1%** (231/1000) | [20.6%, 25.8%] |

### Decision profile by ActionSpaceKind, random-vs-random (per game, hero side; villain is symmetric)

| Metric | STANDARD (1000 games) | INTERACTIVE (1000 games) | Change |
| --- | ---: | ---: | ---: |
| Surfaced decisions (both players) | 194.1 [192.0, 196.1] | 116.8 [114.1, 119.5] | **0.60x** |
| Surfaced decisions (hero) | 97.2 [95.9, 98.4] | 58.3 [56.9, 59.8] | 0.60x |
| Hero priority | 40.7 [40.3, 41.2] | 41.4 [40.4, 42.4] | 1.02x |
| Hero declare_attacker | 36.4 [35.6, 37.3] | 9.0 [8.6, 9.3] | 0.25x |
| Hero declare_blocker | 20.0 [19.7, 20.3] | 2.1 [1.9, 2.2] | 0.10x |
| Hero choose_target | 0.0 | 5.9 [5.8, 6.0] | new |
| **Priority share of hero decisions** | **41.9%** | **71.0%** | +29.1 pts |
| Combat share of hero decisions | 58.0% | 19.0% | −39.0 pts |
| Skipped (trivial, collapsed) | 27.4 [26.8, 27.9] | 17.5 [16.9, 18.0] | 0.64x |
| Collapse ratio | 0.122 [0.121, 0.124] | 0.126 [0.125, 0.128] | ~flat |
| Turns | 40.8 [40.3, 41.2] | 32.7 [31.9, 33.5] | 0.80x |

Zero surfaced decisions with ≤1 valid action in ~311k recorded decisions
across both decks (the `skip_trivial` mechanism remains sound).

### Supplementary (not pre-registered): untrained-vs-random on INTERACTIVE_DECK, 3 inits × 400 games

Run to give the C1 threshold recommendation a baseline on the deck C1
actually trains on.

| Init (torch seed) | Overall [Wilson 95%] | Seat 0 (play) | Seat 1 (draw) |
| --- | --- | ---: | ---: |
| 100000 | 48.2% [43.4%, 53.1%] | 66.0% | 30.5% |
| 101000 | 51.5% [46.6%, 56.4%] | 70.5% | 32.5% |
| 102000 | 49.7% [44.9%, 54.6%] | 49.5% | 50.0% |

Same collapse to ~50% overall. The seat interaction is milder and again
policy-pair-dependent — two of three untrained inits *reverse* the deck's
random-vs-random seat advantage (winning on the play where random loses).
Across all eight inits on both decks, seat-balanced untrained-vs-random spans
**48.2%–51.7%**, every Wilson CI covering 50%.

## Prediction verdicts

**1. On-the-play advantage drops below 80% on INTERACTIVE_DECK — CONFIRMED,
and overshot: the advantage inverted.** Measured 23.1% [20.6%, 25.8%] vs
94.1% on STANDARD_DECK. The prediction's direction and mechanism were right —
interaction punishes pure racing — but the size was wrong: being on the play
is now a large *disadvantage* under random play. Mechanism: with Bolt,
Pyroclasm, and blockers stopping the race, the structural edge that remains
is the draw seat's extra card (the engine implements the CR 103.7.1
first-turn draw skip), and INTERACTIVE_DECK — 40% non-creature spells plus
Ancestral Recall — converts card count into wins far more directly than a
creature pile does. On the play you spend the game a card down in a deck
that's about cards.

**2. Priority share of hero decisions rises above 50% — CONFIRMED.** 71.0%
on INTERACTIVE_DECK vs 41.9% on STANDARD_DECK (seat-balanced; C0's ~38% was
the hero-on-the-play figure). Priority *count* is flat (40.7 → 41.4); the
share rises because combat collapsed.

**3. Original C1 "total decisions ≥1.5x" — REFUTED.** 116.8 [114.1, 119.5]
vs 194.1 [192.0, 196.1]: total decisions *fell* 40% (0.60x). The by-kind
breakdown explains why: combat declarations are per-creature, and the
interactive deck fields fewer creatures (20 vs 36 in the decklist) that die
faster (removal, Pyroclasm, actual blocking) — declare_attacker dropped 36.4
→ 9.0 and declare_blocker 20.0 → 2.1, a loss of ~45 decisions per hero-game.
The predicted flood of new response windows never materialized as *surfaced*
decisions: priority stayed flat and choose_target added only 5.9, because
`skip_trivial` still collapses response windows where the player holds no
instant (collapse ratio unchanged at ~0.13). C1-validation's 171 mean was
measured hero-on-the-play with a different game mix; the seat-balanced
number is lower still. The environment story of C1 is a **mix shift, not a
volume shift**: combat-declaration spam (58% of decisions) was replaced by
priority decisions (71%), exactly the "interesting quantity" amendment A3
anticipated — and the decision horizon *shortened* (58 hero decisions/game),
which slightly eases credit assignment relative to C0's analysis.

## Implications for the C1 training run

1. **The clean null is 50%, for any init.** Seat-balanced
   untrained-vs-random is statistically 50% across all 8 inits × 2 decks
   measured (range 48.2%–51.7%, max Wilson upper bound 56.6%). Seat
   balancing didn't just remove the seat bias — it removed the init lottery
   from the overall number. "Untrained" and "random" are equally matched
   once neither owns a seat.
2. **Recommended C1 training-run success criterion** (replacing the
   withdrawn "untrained-vs-random drops below 65%"): the trained agent's
   **seat-balanced win rate vs random on INTERACTIVE_DECK, ≥400 games (200
   per seat), must have a Wilson 95% lower bound above 55%** — clear of
   every untrained upper bound we measured (≤56.6%, so in practice a point
   estimate ≳60%) — **and a win rate above 50% in each seat separately.**
   The per-seat clause is load-bearing: untrained init 104000 hit 50.2%
   overall while winning 100%/0.5% by seat, so an overall number alone can
   be satisfied by a seat-parasitic policy that learned nothing but the
   race. Both clauses together are unreachable by any untrained init we
   measured (best per-seat pair: 70.5%/32.5%).
3. **Report per-seat win rates in every future eval** (amendment A1 stands),
   and treat the on-the-play rate as a diagnostic of the *policy pair*, not
   a constant of the deck: it ranged from 1.5% to 99.8% on the same deck
   depending on who was playing.
4. **Untrained-vs-passive carries no information** for C1 (100% from both
   seats for every init); keep it only as a smoke test.

## Provenance

- Base: merge of `origin/c0-decision-profile` (decision-profile
  instrumentation) and `origin/c1-interactive-deck` (INTERACTIVE_DECK,
  DrawCards/MassDamage) into `main` at `f67b670` — merge commit `f019e7a` on
  branch `worktree-agent-aacc5f44347d2038e`.
- Code: `manabot/env/match.py` (`Match.swapped`),
  `manabot/verify/decision_profile.py` (seat balancing, per-seat summaries,
  deck + multi-init CLI), `manabot/verify/util.py` (seat-balanced
  `run_evaluation`/`capture_evaluation`),
  `tests/verify/test_seat_balance.py`.
- Machine: Apple M4 Max, macOS. Python 3.12 in an isolated venv
  (`.venv-exp`), managym built from source; `WANDB_MODE=disabled`.
- Checks: `cargo test` 99 passed, 0 failed (10 lib + 10 engine + 79 rules);
  `pytest tests/ -q` 179 passed, 0 failed (gui tests needed `httpx`
  installed into the fresh venv — environment, not code).
- Seeds: game seeds `base + game_index`; bases 0 (rvr), 100000 + 1000·i
  (untrained-vs-random init i), 200000 + 1000·i (untrained-vs-passive init
  i). Untrained torch init seed = matchup base seed (logged per row above).
  Even game index → hero seat 0, odd → seat 1.
- Raw per-game records: `scratch/exp00c-standard.json`,
  `scratch/exp00c-interactive.json`,
  `scratch/exp00c-interactive-untrained.json`.
- Reproduce:
  `python -m manabot.verify.decision_profile --games-random 1000
  --games-untrained 400 --games-baseline 400 --untrained-inits 5
  --baseline-inits 3 --deck standard --seed 0 --out
  scratch/exp00c-standard.json`;
  `python -m manabot.verify.decision_profile --games-random 1000
  --games-untrained 0 --games-baseline 0 --deck interactive --seed 0 --out
  scratch/exp00c-interactive.json`;
  `python -m manabot.verify.decision_profile --games-random 0
  --games-untrained 400 --games-baseline 0 --untrained-inits 3 --deck
  interactive --seed 0 --out scratch/exp00c-interactive-untrained.json`.

## Next question

C1's training run can now be scored cleanly. The environment side is done:
interaction inverted the seat advantage and shifted the decision mix to
priority. The open question is the original C1 one — **does anything
first-light concluded survive on a deck where the game is about cards and
instants instead of racing?** — measured against the criterion above, plus
the aggro-fingerprint metric (instant-holding) that shaping is predicted to
install.
