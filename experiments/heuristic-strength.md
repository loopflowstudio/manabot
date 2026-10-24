# Heuristic strength against a strong checkpoint, and what search buys

**Completed 2026-10-24 on the Mac mini (Apple M1, six game workers).**

Jack Heart asked (2026-10-24) how much the `heuristic` anchor could gain as UR
Lessons against a checkpoint strong against it, without slower decisions; then
whether deck knowledge helps; then what search on top of it costs per unit of
strength.

| | v1 heuristic | this change |
|---|---:|---:|
| UR vs the checkpoint's GW, 300 held-out games | 18.3% | **62.0%** |
| All ten Elemental decks vs the checkpoint's GW and UB seats, 400 games | 10.8% | **34.3%** |
| GW vs the checkpoint's UR, 150 games | 8.7% | 10.7% |
| Per decision, single process, 20 fixed games | 27.0 µs | **23.4 µs** |

No deck scored lower in any of the 20 grid cells. Search over the improved
heuristic adds about 6 points for about 3,000 times the decision time (below).

## Setup

- Opponent: MTG-183 seed 18301, update 1,750 (`mtg183-elemental-long-w2`, raw
  checkpoint SHA-256 `a4dcd67155bee615…`), world w13, pack `elemental-world-v2`.
  It samples its policy (seeded per decision).
- [`heuristic_strength`](heuristic_strength.py) plays each deal twice, once
  with each deck on the play. Iteration deals are 0–49; held-out deals are
  1000–1149; the grid uses 4000–4009. Deal and sampling seeds are fixed, so
  revisions meet identical games.
- v1 is `manabot/arena/heuristic.py` at `6c2f289a2`.

## What changed, in order

Each step is 100 games on the iteration deals, so a step is ±5 points.

| Step | Wins |
|---|---:|
| v1 | 20 |
| Read the Elemental world's card programs (Lightning Strike hit its own face 21 times in 100 games; Combustion Technique and Octopus Form had no rules) | 22 |
| Instant timing: nothing in its own upkeep or draw step; draw and selection instants at the opponent's end step; Learn, discard and select rules | 33 |
| Let its own spell resolve before casting another (no second removal at a doomed target) | 35 |
| Blocks planned once per combat, one blocker per attacker; hold defenders when the swing back is lethal (38 without the hold) | 41 |
| Crack fetch lands (Fabled Passage produced no mana before) | 45 |
| Hold burn for a kill or lethal | 47 |
| Pump unblocked attackers with self-pump abilities (Fire Nation Cadets) | 51 |
| Rank blocks by quality across all pairs; chump only what lethal needs | 53 |
| Learn reads sideboard names, so it actually takes removal against creatures | **57** |

Tried and left out:

- Holding removal for the opponent's attack or end step: 50, against 51.
- An opponent-trick estimate from the authored lists (item 3): the deck is
  inferred from cards shown, and the chance of a pump instant in hand is
  hypergeometric over the unseen main deck. It changed no game at a 50%
  threshold and one at 25%, against GW and GR: those lists hold two or three
  pump instants, so the estimate rarely clears any useful threshold.

Decision time fell because each native observation field is now read once per
decision. Every read converts a whole native list into new Python objects, and
v1 read some lists many times. Most priority prompts return after one pass over
the offers. Target sides come from the board, not a fresh read per offer. The
speed refactor replayed all 100 iteration games decision for decision.

## Search over the heuristic

The search is flat determinized Monte Carlo. At each decision with a real
choice (a cast or ability on offer, an attack, a block, a target), every
offered action is played in W copies of the game. Each copy resamples the
opponent's hidden cards from the searcher's view, and the heuristic plays
both seats to the end. Paired against W = 0 on the 300 held-out games, with
bootstrap intervals over deals:

| Worlds W | Wins | vs W = 0 | Heuristic time per decision | per game |
|---:|---:|---|---:|---:|
| 0 | 62.0% | — | 45 µs | 0.01 s |
| 4 | 64.7% | +2.7 [−3.0, +8.7] | 94 ms | 9.0 s |
| 16 | 68.3% | +6.3 [0.0, +12.7] | 346 ms | 32 s |

On the 50 iteration games, W = 1, 2, 4 and 8 scored 72%, 62%, 70% and 68%
against 68% without search, and W = 16 scored 78%. Times are wall clock with
six concurrent games.

The gain is real but expensive: about six points for 3,000 times the time.
The rollouts model the checkpoint as the heuristic, which the checkpoint is
not, and they are played to the end of the game.

## Limits

- One checkpoint, one training seed, and one primary matchup. Gains elsewhere
  come from the 20-game grid cells.
- Changing the anchor breaks comparability with earlier heuristic curves. The
  ladder and MTG-165/181 numbers were measured against v1. MTG-183's follower
  runs from its pinned checkout and is unaffected.
- Mana is still approximated by untapped lands, and colours are ignored.
