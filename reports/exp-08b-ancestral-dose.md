# exp-08b: deck-tuning probe — 4× Ancestral Recall in UR Lessons

**Date:** 2026-07-09 · **Run by:** coordinating session (not a worktree agent)
· **World:** rules-stage-4 (Milestone-1 engine, PR #47 branch) · **Harness:**
`manabot.sim.flat_mc.play_games`, single process, seed 101 · **Deck delta:**
UR_LESSONS_DECK + 4 Ancestral Recall, − 2 First-Time Flyer, − 2 Accumulate
Wisdom (41 cards preserved) · **Opponent:** GW_ALLIES_DECK unchanged ·
**Pilot:** search-64 both seats.

## Question

exp-08 measured UR Lessons at 23.7% vs GW Allies under search-64 mirror
piloting. Owner: "if the UR deck isn't a favorite something is wrong." Probe:
does maximizing the deck's best card (Ancestral Recall, previously absent
from the 40) move the matchup?

## Pre-registered prediction (recorded in the run script before execution)

> UR search-64 mirror win rate rises from 0.237 to 0.30–0.36 (four copies of
> the best draw spell ever printed is a bigger swap than one card) — better,
> but the gap is structural (threat density vs reactive tools), so one card
> upgrade won't close it. <0.26 = card tuning is noise at this gap; >0.35 =
> the gap is card-quality-driven and deck tuning is higher-leverage than
> believed.

## Result (400 games, 200/seat, seat-balanced)

| metric | value |
|---|---|
| UR win rate | **0.7825** [0.739, 0.820] |
| on the play | 0.990 |
| on the draw | 0.575 |
| mean turns | 15.5 (exp-08 baseline ~19.9) |
| baseline (3 fewer Recalls, exp-08) | 0.237 |

**Prediction refuted upward, decisively.** The matchup did not improve — it
inverted: a 55-point swing from a four-card swap. The >0.35 clause fires:
at this pool size, matchup tables are card-quality-dominated, not structural.

## Reading (with the exp-09 caveat)

- The tuned deck wins as a *race* deck: games shorten, on-play converts 99%.
  Ancestral Recall is proactive — it routes around both pilot pathologies
  (strategy fusion, rollout bias tax only reactive play), so this result says
  nothing about whether the pilot can play control (exp-09 answered that
  separately: it cannot).
- First card-power measurement the platform has produced: 4× Ancestral
  Recall warps a 40-card matchup by 55 points / 99% seat conversion — the
  empirical rediscovery of why the card has been restricted since 1994.
- Follow-up registered, not yet run: singleton Recall (cube-realistic
  dose). Coordinating-session guess: UR lands 0.30–0.38.

## Provenance

Run script (with the pre-registered prediction in its docstring) and raw log
in the session scratchpad; deck delta asserted card-count-preserving at
runtime. Engine world: rules-stage-4 worktree at Milestone-1 completion
(cargo 194 green at that ref). Single seed batch (seed 101, game_offset 0);
no multi-seed replication yet — treat the point estimate, not the third
decimal, as the finding.
