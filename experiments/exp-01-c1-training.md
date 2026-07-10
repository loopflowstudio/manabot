# exp-01: C1 training — does the first-light recipe survive an interactive deck?

**Cycle:** C1 (wave/intelligence/01-experiment-loop.md) · **Date:** 2026-07-09 ·
**Recipe:** `first_light_shaped_v1` unchanged (dev preset: 4 envs, 262k steps,
random opponent, shaping land=0.03 / creature=0.06 / life_loss=0.01) ·
**Deck:** INTERACTIVE_DECK both players (UR: Bolt 6, Counterspell 4,
Ancestral Recall 3, Pyroclasm 3, 20 creatures, 24 lands) ·
**Cost:** 6 training runs ≈ $0.85 at the g5 accounting rate.

## Question

When the deck can interact, does anything first-light concluded survive?

## Prediction status

The original prediction ("untrained-vs-random < 65%") was withdrawn by
amendment A1 (seat contamination). The judging criterion used instead is the
exp-00c threshold, registered before these runs were judged:

> Seat-balanced win rate vs random, 400 games (200/seat): Wilson 95% lower
> bound **> 55% overall** AND **> 50% in each seat**.

Reference points (exp-00c): untrained seat-balanced ≈ 50% (every init 48–52%);
random-vs-random on-the-play on this deck = **23.1%** (the seat advantage
*inverts* — the draw seat's extra card dominates once interaction kills the
race).

## Attempt 1 (methodology broken, findings real)

Three seeds, legacy hero-on-the-play 50-game evals. Two flaws discovered by
running it: evals seat-contaminated, and **no checkpoints existed** —
`Trainer.save()` returned early when wandb was disabled, so every policy was
unrecoverable (fixed in `train.py`; local save now unconditional).

Final win rates (hero on play, 50 games — wide CIs): 48% / 18% / 84%. All
three runs recommended "stay in first-light."

The keeper finding is **seed 2**: it trained to `cast_when_able` 0.96,
`passed_when_able` 0.03 — cast everything at first legal opportunity — and its
win rate *fell* from 44% (untrained) to 18%. On the vanilla deck this behavior
profile was the *success* criterion (first-light gated on `cast_when_able ≥
0.70`). On a deck where Pyroclasm can hit your own board and Counterspell is
worthless cast proactively, the same behavior is a failure mode.
**`cast_when_able` flipped sign when the game acquired strategy.** It was
never measuring skill; it was measuring compliance with the shaping, which
coincided with good play only in a game too simple to punish it.

## Attempt 2 (judged)

Same recipe, three seeds, checkpoints persisted, finals judged seat-balanced
(400 games each, fresh eval seeds):

| seed | overall | LB | on play | on draw | cast_w_able | passed_w_able | verdict |
|---|---|---|---|---|---|---|---|
| 1 | 56.5% | .516 | 66.5% | 46.5% | 0.88 | 0.12 | **FAIL** |
| 2 | 47.3% | .424 | 64.5% | **30.0%** | 0.95 | 0.04 | **FAIL** |
| 3 | 57.5% | .526 | **20.0%** | **95.0%** | 0.50 | 0.41 | **FAIL** |

Baselines for the seat columns: a random policy wins 23.1% on the play, 76.9%
on the draw.

## Reading

1. **The recipe does not transfer.** 0/3 seeds pass. Best overall is 57.5%
   against a 50% untrained baseline — after 262k steps of the recipe that
   reached 100% (seat-inflated) on the vanilla deck. Dynamic range is
   restored: the instrument can now distinguish policies, and it says the
   shaped recipe barely beats noise here.
2. **Seeds 1–2 show the aggro/deploy fingerprint** (cast 0.88–0.95, pass
   0.04–0.12) and both *underperform random on the draw* — they race from the
   disadvantaged seat and squander the advantaged one. Seed 2 is net-negative
   overall: the shaping actively taught it to lose.
3. **Seed 3 is seat-parasitic in the opposite direction**: 95% on the draw
   (above random's 76.9%) but 20% on the play (*below* random's 23.1%). It
   learned to exploit the structurally advantaged seat and nothing else. The
   per-seat threshold clause — added by exp-00c precisely for this — caught
   it; the overall number (57.5%) alone would have looked like mild progress.
4. First-light finding 2 ("reward shaping is required") is now bounded:
   whatever shaping is required, *this* shaping is somewhere between useless
   and harmful on a deck with interaction.

## Ledger

Strength-vs-cost chart, current point: **~57% seat-balanced vs random** at
~$1.30 cumulative training spend (attempts 1+2). Zero-training reference
(search-at-N, cycle C3) not yet measured.

## Next question

Two candidate roads, per the loop doc: **C2** (potential-based shaping — is
bias-free dense signal possible?) and **C3** (flat determinized search — how
much intelligence is free?). C3 first is now clearly right: it needs no
training, its result calibrates the chart everything else is judged against,
and C2's three runs are more interpretable once the free-intelligence
reference exists.

## Provenance

Runs in `.runs/verify.sqlite`: labels `c1-interactive-dev-s{1..3}` (attempt
1), `c1-interactive-dev2-s{1..3}` (attempt 2). Checkpoints:
`.runs/first-light-c1-interactive-dev2-s{n}-*/step_*.pt` (note: step counters
are chunk-local; finals live in `-final/` dirs). Judging evals: 400 games,
seat-balanced, seeds 2001–2003. Branch: search-wave-roadmap (PR #37) at the
checkpoint-persistence fix.
