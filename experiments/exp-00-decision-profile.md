# exp-00 — Decision profile and baseline matrix (cycle C0: E0a + E0c)

Date: 2026-07-09

## Pre-registered prediction

Quoted verbatim from `wave/search/01-experiment-loop.md` (C0, written before any
measurement):

> **Prediction:** surfaced decisions/game (both players) lands in 40–120, well
> under the folklore 200; the hero's share under 60. If so, the effective GAE
> horizon (~17 steps at γλ=0.9405) is within ~3x of the real horizon and the
> "plumbing, not strategy" story weakens — worth knowing before betting on it.
> **Cost cap:** zero GPU-hours; a day of coding.

## Method

**Instrumentation (E0a).** `skip_trivial_count` — the engine counter that was
write-only since its introduction — is now exposed end to end:
`Env::skip_trivial_count()` and `VectorEnv::skip_trivial_counts()` in Rust
(`managym/src/agent/env.rs`, `managym/src/agent/vector_env.rs`), through the
PyO3 bindings (`managym/src/python/bindings.rs`,
`managym/src/python/vector_env_bindings.rs`), and through the Python wrappers
(`manabot/env/env.py`, `manabot/env/vector_env.py`). The counter resets on
game reset, so a read at terminal gives the per-game collapse count. The
current `ActionSpaceKind` was already readable from Python as
`obs.action_space.action_space_type`; no change needed.

**Harness.** `manabot/verify/decision_profile.py` plays full games through the
manabot `Env` (both players surfaced to Python) and records, per game:
surfaced decisions split by ActionSpaceKind and by player, total surfaced
decisions, `skip_trivial_count`, collapse ratio = skipped / (skipped +
surfaced), game length in turns, and the winner. Terminal `GameOver` spaces
are not counted as decisions.

**Setup.** `STANDARD_DECK` both sides (12 Mountain, 12 Forest, 18 Llanowar
Elves, 18 Grey Ogre). Hero is player 0 and — important below — **always on
the play** (`managym/src/flow/setup.rs`: `TurnState::new(PlayerId(0))`).
Policies: `random` = uniform over valid encoded actions; `untrained` = fresh
`Agent` init (attention off, verify defaults), stochastic sampling;
`passive` = always pass priority. Decision-count CIs are normal-approximation
95% CIs on the mean; win rates carry Wilson 95% intervals. Runs: 500 games
random-vs-random, 200 untrained-vs-random, 200 untrained-vs-passive, all with
`skip_trivial=true` (the training default). Zero GPU-hours — under the cost
cap.

## Results — E0a decision profile

All values are per game. Format: mean [95% CI], median, p95.

### random-vs-random (500 games, 0 aborted)

| Metric | Mean [95% CI] | Median | p95 |
| --- | ---: | ---: | ---: |
| Surfaced decisions (both) | 194.1 [191.2, 197.1] | 196 | 246 |
| Surfaced decisions (hero) | 106.0 [104.4, 107.7] | 107 | 135 |
| Surfaced decisions (villain) | 88.1 [86.6, 89.6] | 87 | 117 |
| Skipped (trivial, collapsed) | 27.7 [26.9, 28.4] | 27 | 42 |
| Collapse ratio | 0.124 [0.121, 0.126] | 0.124 | 0.164 |
| Turns (total, both players) | 40.8 [40.2, 41.5] | 41 | 51 |
| Hero priority | 40.0 [39.3, 40.8] | 39.5 | 54 |
| Hero declare_attacker | 45.1 [44.3, 46.0] | 46 | 60 |
| Hero declare_blocker | 20.8 [20.3, 21.3] | 21 | 30 |
| Hero choose_target | 0.0 | 0 | 0 |
| Villain priority | 41.3 [40.7, 42.0] | 41 | 53 |
| Villain declare_attacker | 27.9 [27.1, 28.7] | 27 | 45 |
| Villain declare_blocker | 18.9 [18.5, 19.3] | 19 | 27 |
| Villain choose_target | 0.0 | 0 | 0 |

### untrained-vs-random (200 games, 0 aborted)

| Metric | Mean [95% CI] | Median | p95 |
| --- | ---: | ---: | ---: |
| Surfaced decisions (both) | 192.4 [186.7, 198.2] | 192 | 255 |
| Surfaced decisions (hero) | 90.7 [87.9, 93.5] | 89 | 125 |
| Surfaced decisions (villain) | 101.8 [98.3, 105.2] | 105 | 140 |
| Skipped (trivial, collapsed) | 27.1 [25.7, 28.6] | 27 | 45 |
| Collapse ratio | 0.121 [0.117, 0.125] | 0.122 | 0.164 |
| Turns (total, both players) | 44.1 [42.8, 45.5] | 44 | 58 |
| Hero priority | 44.9 [43.5, 46.2] | 45 | 60 |
| Hero declare_attacker | 27.7 [26.4, 29.1] | 27 | 43 |
| Hero declare_blocker | 18.1 [17.4, 18.8] | 18 | 27 |
| Hero choose_target | 0.0 | 0 | 0 |
| Villain priority | 44.6 [43.1, 46.0] | 44 | 64 |
| Villain declare_attacker | 40.9 [39.1, 42.7] | 42 | 60 |
| Villain declare_blocker | 16.3 [15.5, 17.0] | 16 | 25 |
| Villain choose_target | 0.0 | 0 | 0 |

### untrained-vs-passive (200 games, 0 aborted)

| Metric | Mean [95% CI] | Median | p95 |
| --- | ---: | ---: | ---: |
| Surfaced decisions (both) | 73.6 [71.9, 75.3] | 73 | 93 |
| Surfaced decisions (hero) | 54.2 [52.7, 55.6] | 54 | 71 |
| Surfaced decisions (villain) | 19.5 [19.2, 19.8] | 20 | 22 |
| Skipped (trivial, collapsed) | 20.4 [19.5, 21.3] | 20 | 33 |
| Collapse ratio | 0.212 [0.208, 0.216] | 0.213 | 0.265 |
| Turns (total, both players) | 20.5 [20.2, 20.8] | 21 | 23 |
| Hero priority | 23.0 [22.5, 23.4] | 23 | 29 |
| Hero declare_attacker | 31.2 [30.1, 32.3] | 31 | 45 |
| Hero declare_blocker | 0.0 | 0 | 0 |
| Villain priority | 19.5 [19.2, 19.8] | 20 | 22 |
| Villain declare_attacker / blocker / target | 0.0 | 0 | 0 |

**Sanity checks.** Zero surfaced decisions with ≤1 valid action across all
~150,000 recorded decisions — `skip_trivial` collapses every single-action
space; nothing trivial leaks through. Zero `choose_target` decisions —
STANDARD_DECK contains no targeted effects.

## Results — E0c baseline matrix

Hero is player 0 and is always on the play in every matchup.

| Matchup | Games | Hero wins | Win rate | Wilson 95% CI |
| --- | ---: | ---: | ---: | --- |
| random-vs-random | 500 | 467 | 93.4% | [90.9%, 95.3%] |
| untrained-vs-random | 200 | 26 | 13.0% | [9.0%, 18.4%] |
| untrained-vs-passive | 200 | 200 | 100.0% | [98.1%, 100%] |

**Robustness check on "untrained".** Two additional fresh inits (100 games
each vs random): torch seed 555 → 68.0% [58.3%, 76.3%]; torch seed 777 →
89.0% [81.4%, 93.7%]. Together with the main run's 13.0% [9.0%, 18.4%], the
three inits produce non-overlapping intervals spanning 13–89%. The
"untrained-vs-random" number is dominated by initialization variance, not
sampling noise: a single-init untrained baseline is not a baseline.

## Prediction vs. result

**Refuted, both clauses.**

- Surfaced decisions/game (both players): predicted 40–120; measured **194.1
  [191.2, 197.1]** (random-vs-random) and **192.4 [186.7, 198.2]**
  (untrained-vs-random). Above the predicted range by ~60%, and the folklore
  "~200 steps/game" turns out to be *approximately correct* for two-sided play
  on this deck — unsourced, but not wrong.
- Hero's share: predicted < 60; measured **106.0 [104.4, 107.7]** (random)
  and **90.7 [87.9, 93.5]** (untrained). Roughly double the prediction.

The one configuration that lands inside the predicted range is
untrained-vs-passive (73.6 total, 54.2 hero) — the profile is strongly
opponent-dependent, so any future decision-horizon claim must name the
matchup.

## Implications

**The "plumbing, not strategy" credit-assignment story survives.** The
pre-registered conditional was: *if* decisions landed in 40–120 with hero
under 60, the ~17-step effective GAE horizon (1/(1−γλ) = 16.8 at γλ =
0.99 × 0.95 = 0.9405) would be within ~3x of the real horizon and the story
would weaken. The opposite happened. Hero surfaces 91–106 decisions/game
against an active opponent — **5.4–6.3x the GAE horizon**. The terminal win
signal decays by (γλ)^N: it retains ~35% of its weight 17 steps back, ~0.4%
at 91 steps, and ~0.15% at 106 steps. Without an accurate value function to
bootstrap through, decisions in the first half of a game are effectively
invisible to the terminal reward. Dense/shaped signal (or a good V) is not a
nice-to-have on this horizon; it is the mechanism.

**The auto-pass claim is dead.** `skip_trivial` collapses only **12.4%** of
decision points under two-sided random play (21.2% vs a passive opponent) —
not the majority that folklore assumed. The engine's decision profile is
dominated by combat micro-decisions: declare_attacker + declare_blocker are
~58% of surfaced decisions under random-vs-random (one decision per eligible
creature per combat). If per-creature combat declaration were batched or
otherwise collapsed, the surfaced horizon would shrink far more than any
skip_trivial tuning.

**Seat advantage contaminates every win-rate baseline.** Random-vs-random is
**93.4%** for the player on the play. On a deck with no interaction, whoever
attacks first races ahead and random blocking cannot stabilize; the game is
close to a deterministic race. Consequences: (a) "win rate vs random" has a
ceiling/floor structure set by seat, not skill — all historical
hero-perspective numbers (e.g. first-light's 82%-vs-random) were measured
from the on-the-play seat and are inflated by it; (b) C1's pre-registered
threshold ("untrained-vs-random drops below 65%") is untestable as stated
unless seat is randomized or reported per side.

**"Untrained" is not "random."** A fresh network is a *biased* policy — its
init-dependent action preferences put it anywhere from 13% to 89% vs random
across three seeds, versus 93.4% for an actually-random hero in the same
seat. Untrained-policy baselines need an ensemble over inits, and
random-vs-random — not untrained-vs-random — is the meaningful "no-skill"
reference point.

**E0b (cost basis) remains open** — this report covers E0a and E0c only.

## Provenance

- Instrumentation + harness code: commit `b884038` on branch
  `worktree-agent-a3313d12dd35c2ac6` (main merged at `a95648a`). Measurement
  runs executed on a tree identical to that commit.
- Machine: Apple M4 Max, macOS 26.0.1. Python 3.12.12 in an isolated venv
  (`.venv-exp`), managym built from source via maturin; wandb disabled.
  `cargo test`: 95 passed (10 lib + 10 engine + 75 rules), 0 failed.
  `pytest tests/ -x -q`: passed.
- Sample sizes: 500 (random-vs-random), 200 (untrained-vs-random), 200
  (untrained-vs-passive), plus 2 × 100 init-robustness games.
- Seeds: game seeds `seed + game_index` with base 0 / 100000 / 200000 per
  matchup; untrained init = torch seed equal to the matchup base seed.
- Raw per-game records: `scratch/exp00-decision-profile.json`.
- Reproduce: `python -m manabot.verify.decision_profile --games-random 500
  --games-untrained 200 --games-baseline 200 --seed 0 --out
  scratch/exp00-decision-profile.json`

## Next question

**How much of any measured win rate is seat, not skill — and what does the
baseline matrix look like once the seat is randomized?** Concretely: add
seat randomization (or alternation) to the eval harness and re-derive the
random-vs-random and untrained ensembles per side, *before* C1 changes the
deck — otherwise C1's "drops below 65%" prediction is being scored against a
93%/7% seat-split baseline and can pass or fail on seat assignment alone.
Secondary follow-up for C1's design: surfaced-decision growth (the ≥1.5x
prediction) should be measured per kind, since combat declaration — not
priority windows — is what dominates the current profile.
