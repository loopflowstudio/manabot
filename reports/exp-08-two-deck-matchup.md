# exp-08 — The first A-vs-B matchup table: UR Lessons vs GW Allies

Date: 2026-07-09 (Milestone 1, wave/rules Stage 4)

## Pre-registered prediction

Written and committed before any games were run (the only prior data point:
the Stage-3 smoke, 200 games NOT seat-balanced, where GW as villain/on-the-
draw won 142/200).

**I pick GW Allies, and I predict search narrows the gap but does not close
it.** Card-level reasoning: under uniform-random play, UR's spell suite is
mostly *conditional* value — counterspells (Divide by Zero, It'll Quench Ya!)
fired at nothing or at the wrong stack, scry/learn card selection wasted by
random picks, kicker paid at random. GW's cards are *unconditionally* board-
positive: bodies with ETB counters, token makers, an anthem, earthbend
inevitability — a random pilot still deploys them and attacks into a random
defense. Random-vs-random attrition should therefore heavily favor GW.
Search reclaims part of UR's conditional value (countering the anthem,
aiming Igneous Inspiration/kickered Firebending Lesson as removal, bounce as
tempo), but flat MC with random rollouts still evaluates positions through
random-play continuations, which under-price UR's control lines and
over-price GW's board inevitability.

Numbers (UR win rate, seat-balanced):

| Cell | Predicted UR win rate |
| --- | ---: |
| random (UR) vs random (GW) | **0.32** (GW 68%) |
| search-16 vs search-16 | 0.38 |
| search-64 vs search-64 | **0.42** (gap narrows with N) |
| search-64 (UR) vs random (GW) | 0.85 |
| random (UR) vs search-64 (GW) | 0.03 (GW 97%) |

Secondary predictions: seat (on-play) advantage persists in every cell but is
smaller than the deck effect; search-vs-search games run longer in turns than
random-vs-random (search stabilizes boards; random games end by attrition).

## Method

**Harness.** `manabot/verify/run_two_deck_matchup.py` — the exp-02
multiprocess runner pattern over `manabot/sim/flat_mc.py::play_games`, with
per-game decision-profile instrumentation added to the shared loop (turns,
steps, surfaced decisions by ActionSpaceKind split per side — the exp-00
instrumentation, extended to the Stage-2/3 decision kinds: scry,
look_and_select, pay_or_not, modal, discard_then_draw, waterbend).

**Setup.** Hero side is ALWAYS the UR Lessons player (blessed 41-card list,
`manabot.verify.util.UR_LESSONS_DECK`); villain is always GW Allies (40).
Seat-balanced: the UR player takes seat 0 (on the play) in even-indexed
games and seat 1 (on the draw) in odd — so the hero win rate IS the UR
per-deck win rate, and per-seat splits come for free. 400 games per cell,
8 worker processes, base seed 0 (per-cell offset), env reseeded per game.
Search = flat determinized MC (`managym.Env.flat_mc_scores`), N sims per
legal action as W worlds x 4 rollouts, random-rollout evaluation, argmax.
Win-rate CIs are Wilson 95%. Encoded action capacity was raised to
`max_actions = 32` first — at 20, real-deck priority windows truncated and
the uniform-random player never saw the tail of the action list.

Note on the eval-stack observation change shipped in the same wave:
observation encoding gained effective-keyword visibility and larger
permanent/action capacities (CARD_DIM 38, PERMANENT_DIM 24, 40 permanent
slots, 32 action slots); search itself reads raw engine state and is
unaffected, the random player samples over valid encoded actions.

## Results

2000 games total (5 cells x 400), **zero aborted / drawn / step-capped games,
zero engine errors**. Raw per-cell aggregates in
`reports/data/exp-08-two-deck-matchup.json`.

### Win rates (hero = UR Lessons side)

| Matchup (UR vs GW) | UR win rate [Wilson 95%] | UR on play | UR on draw | On-play win (any deck) | Turns mean/med/p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| random vs random | **0.345** [0.300, 0.393] | 0.420 [0.354, 0.489] | 0.270 [0.213, 0.335] | 0.575 [0.526, 0.623] | 24.7 / 22 / 44 |
| search-16 vs search-16 | **0.223** [0.184, 0.266] | 0.275 [0.218, 0.341] | 0.170 [0.124, 0.228] | 0.552 [0.504, 0.600] | 18.9 / 17 / 33 |
| search-64 vs search-64 | **0.237** [0.198, 0.282] | 0.185 [0.137, 0.245] | 0.290 [0.232, 0.356] | 0.448 [0.400, 0.496] | 20.4 / 20 / 29 |
| search-64 (UR) vs random (GW) | **0.860** [0.823, 0.891] | 0.810 [0.750, 0.858] | 0.910 [0.862, 0.942] | 0.450 [0.402, 0.499] | 24.2 / 21 / 44 |
| random (UR) vs search-64 (GW) | **0.007** [0.003, 0.022] | 0.005 [0.001, 0.028] | 0.010 [0.003, 0.036] | 0.497 [0.449, 0.546] | 16.1 / 15 / 24 |

### Game length and decision profile

| Matchup | UR decisions/game (mean, p95) | GW decisions/game (mean, p95) | Steps mean | UR ms/dec | GW ms/dec |
| --- | ---: | ---: | ---: | ---: | ---: |
| random vs random | 67.3, 140 | 118.4, 434 | 186 | — | — |
| search-16 vs search-16 | 44.5, 88 | 58.1, 176 | 103 | 28 | 38 |
| search-64 vs search-64 | 53.5, 93 | 62.6, 131 | 116 | 174 | 191 |
| search-64 (UR) vs random (GW) | 67.5, 126 | 115.7, 420 | 183 | 170 | — |
| random (UR) vs search-64 (GW) | 36.8, 63 | 40.5, 71 | 77 | — | 155 |

(ms/dec measured with 8 concurrent worker processes — treat as relative.)

Decision-kind mix, mean surfaced decisions per game (random-vs-random cell):

- UR: priority 44.9, declare_attacker 10.3, declare_blocker 4.2,
  choose_target 3.5, **discard_then_draw (learn) 2.7, pay_or_not 0.9,
  look_and_select 0.7**
- GW: priority 54.9, **waterbend 30.2**, declare_attacker 17.8,
  look_and_select 5.1, choose_target 4.7, **scry 2.8**, declare_blocker 2.6,
  pay_or_not 0.3

Every Stage-2/3 decision kind except modal (no modal card in the blessed 40)
surfaces in every cell. Random play drowns in GW's waterbend spaces (30/game
— tap-to-pay menus after randomly activating Water Tribe Rallier); search
prunes that to ~8/game, which is most of why search games run ~40% fewer
steps.

## Prediction vs result

| Cell | Predicted UR | Actual UR | Verdict |
| --- | ---: | ---: | --- |
| random vs random | 0.32 | 0.345 [0.300, 0.393] | ✓ inside the CI |
| search-16 mirror | 0.38 | 0.223 | ✗ wrong direction |
| search-64 mirror | 0.42 | 0.237 | ✗ **search widens the gap** |
| search-64 (UR) vs random (GW) | 0.85 | 0.860 [0.823, 0.891] | ✓ dead on |
| random (UR) vs search-64 (GW) | 0.03 | 0.007 | ~ right ballpark, GW even stronger |

The deck call (GW favored everywhere) and both asymmetric cells were right;
the headline claim — that search would *narrow* the deck gap by reclaiming
UR's conditional value — was wrong. Search **widens** it (UR 34.5% → ~23%)
and the widening is flat in N (16 vs 64 statistically indistinguishable:
0.223 vs 0.237, overlapping CIs). Reading: flat MC with random-rollout
evaluation is exactly the wrong lens for a control deck. It sharpens GW's
proactive plan (search-GW attacks 14x/game vs random-GW's blundering 17.8x
that includes bad attacks; it stops wasting waterbend menus) while UR's
counterspells and card selection only pay off through multi-turn
coordination that one-ply search over random continuations cannot credit —
random rollouts price a held-up Divide by Zero at roughly zero. Both
secondary predictions also failed, instructively: search-vs-search games are
*shorter*, not longer (18.9–20.4 turns vs 24.7 — search closes games,
doesn't stall them), and the on-play seat advantage does *not* persist under
search — at search-64 the on-play win rate (any deck) drops to 0.448
[0.400, 0.496] and UR does better on the *draw* (0.290 vs 0.185): with
competent piloting the extra card appears to outweigh the tempo of moving
first in this matchup.

Milestone-1 headline: **GW Allies is the better deck — 65.5% seat-balanced
against uniform random, ~77% in search mirrors — and the gap is a deck
property, not a seat artifact (seat effects are an order of magnitude
smaller and vanish under search).** Skill dominates deck: search-64 gives
the weaker deck an 86% win rate against a random GW pilot.
