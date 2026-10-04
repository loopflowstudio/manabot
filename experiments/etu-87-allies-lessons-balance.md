# ETU-87 — Which cards decide Allies versus Lessons?

**Status: exploratory.** No prediction was committed before these runs, so by
the discipline in [README](README.md) the numbers below are a map for choosing
the next decklists, not evidence for a claim. Runs: 2026-09-29 to 2026-10-04.

## Question

With the same player in both seats, GW Allies beats UR Lessons about 70–30.
Which cards carry that gap, and what is the smallest decklist change that
brings the matchup near even?

## Method

`python -m manabot.study.allies_lessons.balance` plays each candidate pair of
lists with Demo Search-64 (64 simulations, 4 rollouts per world) in both
seats. Every deal seed is played once in each deck order, so games = 2 × deals.
The rate is UR Lessons' share of wins; intervals are 95%. Candidates that
share a seed share their deals. Every game finished (`unfinished: 0`).

Two worlds appear. **w3** is the corrected Learn/Lesson world. **w4** adds
Gran-Gran, Proft's Eidetic Memory, Combustion Technique and the hand-size
rule (PR #196). Rates are comparable within a world and a seed, not across.

Result rows are in
[`data/etu87-allies-lessons-balance-20260930/`](data/etu87-allies-lessons-balance-20260930/);
each row carries the full lists it played. Candidate files are in
[`study/`](study/).

## Results

### The gap is in the decks, not the pilot

The 5,800-game play study (w3, seeds from 97000, configuration in
[`study/allies-lessons-10x-study.json`](study/allies-lessons-10x-study.json))
gives UR Lessons the same share under either pilot:

| Pilot in both seats | Games | UR Lessons wins | 95% interval |
|---|---|---|---|
| Demo Search-64 | 1,200 | 30% (354) | 27% to 32% |
| Random | 4,000 | 30% (1,183) | 28% to 31% |

Search-64 beat Random in 559 of 600 games (93%), so the pilot is far from
random and the deck gap is unchanged. The record and report for this run are
under `.runs/study/allies-lessons-10x/` on the machine that ran it and are not
committed (77 MB).

### Cutting Allies cards for lands (w3)

| GW Allies change | File | Seed | Games | UR Lessons wins | 95% interval |
|---|---|---|---|---|---|
| Authored lists | `gw-weaken-10x` | 96000 | 1,000 | 30.7% | 28% to 34% |
| Cub ×2 to lands | `gw-weaken-10x` | 96000 | 1,000 | 32.4% | 30% to 35% |
| Voyager to land | `gw-weaken-10x` | 96000 | 1,000 | 32.5% | 30% to 35% |
| Cub ×2 and Voyager to lands | `gw-weaken-10x` | 96000 | 1,000 | 34.5% | 32% to 38% |
| Cub ×2 to Healer | `gw-weaken-10x` | 96000 | 1,000 | 29.6% | 27% to 33% |
| Voyager to Healer | `gw-weaken-10x` | 96000 | 1,000 | 31.9% | 29% to 35% |
| Cub ×2 and Voyager to Healer ×2 and Deserters | `gw-weaken-10x` | 96000 | 1,000 | 34.0% | 31% to 37% |
| 6 to lands: the three above, White Lotus Reinforcements ×2, Suki | `gw-lands` | 96000 | 300 | 59.7% | 54% to 65% |
| 10 to lands: plus Earth King's Lieutenant ×2, Kyoshi Warriors ×2 | `gw-lands` | 96000 | 300 | 94.0% | 91% to 96% |
| 14 to lands: plus Allies at Last ×2, Fancy Footwork ×2 | `gw-lands` | 96000 | 300 | 94.3% | 91% to 96% |

Badgermole Cub and South Pole Voyager, the first suspects, are worth about
four points together. The next three cards are worth about 25.

### White Lotus Reinforcements and Suki (w3)

| GW Allies change | File | Seed | Games | UR Lessons wins | 95% interval |
|---|---|---|---|---|---|
| One White Lotus Reinforcements to land | `gw-targeted` | 96000 | 300 | 31.0% | 26% to 36% |
| Suki to land | `gw-targeted` | 96000 | 300 | 33.3% | 28% to 39% |
| White Lotus Reinforcements ×2 to Healer ×2 | `gw-targeted` | 96000 | 300 | 33.3% | 28% to 39% |
| White Lotus Reinforcements ×2 and Suki to lands | `gw-targeted` | 96000 | 300 | 46.0% | 40% to 52% |
| **White Lotus Reinforcements ×2 and Suki to lands** | `gw-confirm` | 98000 | 1,000 | **47.7%** | 45% to 51% |
| White Lotus Reinforcements ×2 and Suki to Healer ×2 and Deserters | `gw-confirm` | 98000 | 1,000 | 38.5% | 36% to 42% |

Cutting the two lords and Suki together for lands is the smallest change
found that brings w3 near even, and it held on fresh deals. No single cut
moves the rate outside the authored interval. Replacing the three with other
creatures instead of lands gives back about nine of the seventeen points, so
part of the effect is Allies having fewer bodies, not only losing these three.

### The revised Lessons list (w4)

Jack's revised UR Lessons list, as changes to the authored main deck:
-2 It'll Quench Ya!, -2 First-Time Flyer, -1 Pop Quiz, -1 Igneous
Inspiration, +2 Combustion Technique, +1 Proft's Eidetic Memory,
+1 Gran-Gran, +1 Accumulate Wisdom.

| Lists | File | Seed | Games | UR Lessons wins | 95% interval |
|---|---|---|---|---|---|
| Authored Lessons vs authored Allies | `w4-lessons` | 96000 | 300 | 28.3% | 24% to 34% |
| Revised Lessons vs authored Allies | `w4-lessons` | 96000 | 300 | 41.3% | 36% to 47% |

The revised list is worth about 13 points and leaves Lessons short of even
on its own.

### Both levers together (w4)

Candidates: [`study/w4-combined-2026-10-04.json`](study/w4-combined-2026-10-04.json),
150 deals from seed 96000, the same deals as the two rows above.

**Not yet run.** The first attempt (2026-09-30) crashed at startup on a card
name containing a comma, fixed since. The second (2026-10-04) was stopped on a
contended machine before any candidate finished. The prediction below was
committed before any result existed and stands for the rerun:

| Lists | Predicted UR Lessons wins |
|---|---|
| Revised Lessons vs Suki to land | 38% to 49% |
| Revised Lessons vs one lord to land | 37% to 48% |
| Revised Lessons vs lords and Suki to lands | 50% to 62% |
| Authored Lessons vs lords and Suki to lands | 39% to 51% |

The two levers are read as roughly additive: 13 points from the list and 17
from the cut on a 28% baseline. If revised Lessons against the lord-and-Suki
cut lands below 45%, the levers are not complementary and the additive
reading is wrong.

```bash
uv run python -m manabot.study.allies_lessons.balance \
    --candidates experiments/study/w4-combined-2026-10-04.json \
    --deals 150 --seed 96000 --out .runs/study/w4-combined.jsonl
```

## Reading

- The matchup gap is a property of the lists. A random pilot reproduces it.
- Allies' strength sits in its creature suite, and inside that in the lord
  (White Lotus Reinforcements) together with Suki. Allies at Last and Fancy
  Footwork add nothing once the creatures are gone.
- The two levers found are an Allies cut (lords and Suki) and a stronger
  Lessons list (the three w4 cards). Whether they add is unmeasured.

## What could make this wrong

- **Pilot dependence.** Search-64 is a weak, uniform-rollout pilot. Jack
  noted that UR Lessons card strength is contextual; a pilot that cannot
  sequence Lessons undervalues that deck, and a stronger pilot may move the
  crossing point. The Random mirror agreeing at 30% argues the gap is not
  only a Search-64 artifact, but it says nothing about strong play.
  Discriminator: repeat the confirmed pair with a trained manabot in both
  seats.
- **Small samples mislead.** At 100 games, "Cub ×2 and Voyager to lands"
  read 50% (`gw-weaken`, seed 95000); at 1,000 games it is 34.5%. Treat
  300-game rows as screening and confirm on fresh deals before acting.
- **Selection.** The lord-and-Suki cut was chosen after seeing the six-card
  result on seed 96000. The seed-98000 confirmation is the only row free of
  that selection.
- **Land swaps change two things.** Turning spells into lands both removes
  the card and raises Allies' land count from 17 to 20, so "to lands" rows
  overstate a card's own worth. The Healer rows separate the two for the
  lord-and-Suki cut only.
- **World drift.** The Allies cuts were measured on w3 and the Lessons list
  on w4; the w4 authored baseline (28.3%) is close to w3's (30.7%) but the
  hand-size rule differs.

## Not run

A removal sweep over UR Lessons cards was stopped after three candidates
(`balance-sweep`, 100 games each) on Jack's note that contextual cards make
single-card removals uninformative for that deck.
