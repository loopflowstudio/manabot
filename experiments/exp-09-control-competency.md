# exp-09 — Control competency: can the pilot play reactive Magic?

Date: 2026-07-09 (wave/search C9)

## Question

exp-08 measured UR Lessons (the control deck) at ~22% against GW Allies under
search mirrors, and search *widened* the deck gap relative to random-vs-random.
Two hypotheses survive that result:

- **H1 — deck**: GW/go-wide is structurally favored; the pilot is fine.
- **H2 — pilot**: flat determinized MC with uniformly-random rollouts cannot
  play control. Two named mechanisms: *strategy fusion* (the one-ply search
  never credits holding interaction for a future branch, because the random
  continuation doesn't execute the plan), and *rollout pathology* (random
  continuations burn inherited counterspells on the first target, so a held
  counter is priced at ~zero).

This experiment builds the instrument that discriminates them: score the
*decisions*, not just the win rates.

## Instruments

**1. Competency scenarios** (`manabot/verify/competency.py`, engine surface
`managym/src/flow/scenario.rs`). Exact mid-game positions injected into the
engine (hands, battlefields, life totals), a deterministic scripted villain,
and a tracker that scores each run against a documented known-correct line.
Hero is always seat 0. Run-to-run variance comes only from the agent's own
randomness (search world/rollout seeds, library shuffle). N ≥ 100 runs per
cell, Wilson 95% CIs. Agents: random (floor), search-16/64/256 (the exp-02/
exp-08 pilot at three strengths).

**2. Micro-format mirrors.** Two minimal 40-card decks:

- `MICRO_AGGRO` (5 names): 9 Mountain, 9 Island, 8 Gray Ogre,
  7 Raging Goblin, 7 Wind Drake.
- `MICRO_CONTROL` (6 names): 10 Island, 8 Mountain, 6 Wind Drake,
  6 Counterspell, 7 Lightning Bolt, 3 Ancestral Recall.

Seat-balanced, 300 games/cell: both mirrors at search-64 (sanity: ≈50%), and
control-vs-aggro at random / search-16 / search-64 / search-256, with
per-decision behavioral probes on the control side: what each Counterspell
countered (name, mana value, first-window rate), what each Bolt targeted
(biggest-threat rate among multi-creature choices, face share), and an
instant-holding rate (main-phase windows where casting the creature would
drop open mana below UU while holding a counter).

## The scenarios and their correct lines

**S1 COUNTER-THE-BOMB.** Hero: 12 life, 2 untapped Islands + 2 Wind Drakes,
hand = [Counterspell], library all Islands. Villain: 20 life, 6 Mountains,
hand = [Gray Ogre, Shivan Dragon], deck 30 Mountain / 5 Ogre / 5 Dragon.
Script: cast Ogre T1, Dragon T2 (7 mana by then), attack always. The clock
math: hero's 2 drakes deal 4/turn → villain dead end of hero-T6 at the
earliest. If the Dragon resolves it deals 5/turn from villain-T3 → hero (12,
taking 2/turn from the ogre already) dies on villain-T4/T5, *before* the
drakes finish. If the Dragon is countered, the ogre's 2/turn kills the hero
on villain-T6 at the earliest — after the hero's T6 lethal. Countering the
Ogre instead leaves the Dragon → same loss. **Correct: decline the Ogre,
counter the Dragon.** Score: bomb-countered rate. The information set
supports the line without seeing the villain's hand: the deck is 5 Dragon /
5 Ogre, so the only counterspell should be saved for the expensive half.

**S2 HOLD-THE-WIPE.** Hero: 20 life, 2 Mountains, hand = [Pyroclasm], library
all Mountains. Villain: 20 life, 2 ready Gray Ogres + 6 Mountains, hand =
[2x Gray Ogre], deck 26 Mountain / 14 Ogre. Script: cast every Ogre when
able, attack always. Wiping on T1 kills 2 ogres and saves the T1 attack
(4 damage); the villain redeploys 2 from hand immediately. Waiting one turn
wipes 4. EV gap: +2 ogres killed (6 mana of board) and the villain's hand
emptied, for -4 life from 20 → 16 with a post-wipe residual clock near zero
either way. At 16+ life the extra life has ~no marginal value; the two extra
kills remove the entire rebuild. **Correct: pass T1, wipe ≥4 on T2.** Score:
premature-wipe rate (cast wiping ≤2).

**S3 BOLT-THE-THREAT.** Hero: 14 life, 2 Wind Drakes + 3 lands, hand =
[Lightning Bolt], library all lands. Villain: 20 life, a 1/1 Invasion
Reinforcements decoy + 4 lands, hand = [Earth King's Lieutenant, White Lotus
Reinforcements, Invasion Reinforcements]. Script casts the Lieutenant T1
(a 1/1 that puts a +1/+1 counter on each other Ally on entry and grows with
every later Ally). On villain-T2 the script casts WLR (+token) + IR: the
Lieutenant becomes 4/4 — out of bolt range forever, with a pumped board.
**Correct: hold the bolt past the decoy, kill the Lieutenant in its 1/1
window (villain-T1 to hero-T2).** Score: Lieutenant-killed rate; buckets for
decoy / face / own-creature / never.

**S4 RACE-VS-BLOCK.** Hero: 10 life, 2 Wind Drakes + 2 Gray Ogres (all
ready), empty hand, library all lands. Villain: 17 life, 3 ready Gray Ogres
+ 3 Mountains, empty hand, library all Mountains (full information modulo
empty hands). Script: attack all, never block. Race line (attack 4 = 8/turn):
villain dies on hero-T3, but attackers tap and the villain's 6/turn kills the
hero during villain-T2 — one turn too late. Block line (attack drakes only,
ogres block): T1 trades both hero ogres for two villain ogres taking 2; the
remaining ogre deals 2/turn while the drakes' 4/turn finishes on hero-T5 with
the hero at 2 life. **Correct: attack with exactly the two drakes, hold both
ogres.** Score: raced rate (any ogre in the first attack batch).

**S5 HOLD-UP-QUENCH.** Hero: 20 life, 2 Islands, hand = [It'll Quench Ya!,
Otter-Penguin], library all Islands. Villain: 20 life, 6 Forests, hand =
[Craw Wurm], deck 33 Forest / 7 Wurm. Script casts the Wurm T1, tapping 6 of
at most 7 lands — it cannot pay the {2} Quench tax. A resolved 6/4 dominates
this board forever (nothing in the hero deck answers it); the 2/1 penguin
trades with nothing and the hero can deploy it on T2+ anyway (all-Island
library keeps making land drops). **Correct: pass the main phase holding
1U; quench the tapped-out Wurm.** Score: held-main rate and
counter-completion (held AND quenched).

## Pre-registered predictions (written 2026-07-09, before any runs)

Reasoning from **H2**: the pilot's correct-line rate should be *flat in N*
(more simulations sharpen the same broken evaluation) and low wherever the
correct line requires declining immediate value; random can even *beat*
search where the correct line is "do nothing yet" (random sometimes does
nothing by accident, search systematically prefers the visible action).

Scenario correct-line rates:

| Scenario | random | search-16 | search-64 | search-256 |
| --- | ---: | ---: | ---: | ---: |
| S1 counter-the-bomb | 0.05 | 0.10 | 0.15 | 0.15 |
| S2 hold-the-wipe | 0.40 | 0.15 | 0.20 | 0.25 |
| S3 bolt-the-threat | 0.10 | 0.25 | 0.30 | 0.35 |
| S4 race-vs-block | 0.06 | 0.10 | 0.15 | 0.20 |
| S5 hold-up-quench | 0.25 | 0.20 | 0.25 | 0.30 |

Headline registered claims: (a) no scenario exceeds 0.50 correct at any N;
(b) the N=256 vs N=16 delta is < 0.15 on every scenario (flatness); (c) on
S2, random ≥ search-16 (the "do-nothing" anomaly).

Micro-format (hero win rate, seat-balanced, 300 games):

| Cell | Prediction |
| --- | ---: |
| aggro mirror @ search-64 | 0.50 ± CI |
| control mirror @ search-64 | 0.50 ± CI |
| control vs aggro @ random | 0.30 |
| control vs aggro @ search-16 | 0.35 |
| control vs aggro @ search-64 | 0.35 |
| control vs aggro @ search-256 | 0.33 |

Registered claim: control-vs-aggro is flat-to-declining in N (search does not
rescue the control deck), while the behavioral stats stay pathological at
every N: counter_first_window_rate > 0.70, countered-MV mean ≤ 2.4 (counters
burned on the cheapest early spells), instant_holding_rate < 0.15,
bolt_face_share > 0.30, bolt_biggest_rate ≈ 0.5 (coin-flip targeting).

Decision rule, registered in advance:

- **H2 confirmed** if the scenario rates are flat-low in N *and* the micro
  behavioral stats stay pathological while control-vs-aggro fails to improve
  with N. (The deck can still also be worse — H1 and H2 are not exclusive —
  but the *pilot* is then established as incapable of control lines, and
  exp-08's matchup number says nothing about the decks.)
- **H1 (pilot exonerated)** if scenario rates climb toward 1.0 with N and the
  behavior stats normalize (counters held past bait, wipes delayed): then
  22% was a fair measurement of the deck.

## Results

Scenario suite: 2,000 scored runs (5 scenarios × 4 agents × 100), zero
engine errors, ~4 minutes of wall clock at 4 workers. Raw data:
`reports/data/exp-09-scenarios.json`.

### Scenario table

Correct-line rate per agent, Wilson 95% CIs. **Predicted** values are the
pre-registered numbers from the section above.

| Scenario | random | search-16 | search-64 | search-256 |
| --- | ---: | ---: | ---: | ---: |
| S1 counter-the-bomb | **0.34** [.26,.44] (pred .05) | **0.39** [.30,.49] (.10) | **0.30** [.22,.40] (.15) | **0.39** [.30,.49] (.15) |
| S2 hold-the-wipe | **0.23** [.16,.32] (.40) | **0.00** [0,.04] (.15) | **0.00** [0,.04] (.20) | **0.00** [0,.04] (.25) |
| S3 bolt-the-threat | **0.00** [0,.04] (.10) | **0.00** [0,.04] (.25) | **0.00** [0,.04] (.30) | **0.00** [0,.04] (.35) |
| S4 race-vs-block | **0.04** [.02,.10] (.06) | **0.02** [.01,.07] (.10) | **0.01** [0,.05] (.15) | **0.00** [0,.04] (.20) |
| S5 hold-up-quench | **0.14** [.09,.22] (.25) | **0.01** [0,.05] (.20) | **0.08** [.04,.15] (.25) | **0.03** [.01,.09] (.30) |

All three registered headline claims hold, mostly by wide margins:

- **(a) No scenario exceeds 0.50 at any N.** Max observed: 0.39.
- **(b) Flat in N** (registered: Δ(256−16) < 0.15 everywhere). Observed
  Δ: S1 0.00, S2 0.00, S3 0.00, S4 −0.02, S5 +0.02. More simulations do
  not buy any tactical competence.
- **(c) Random ≥ search-16 on S2.** 0.23 vs 0.00 — and the anomaly is not
  S2-specific: **uniform random beats every search strength outright on S2,
  S4 and S5**, and matches it on S1. Search never beats random anywhere.

What the decision-level detail shows (the mechanism, per scenario):

- **S1**: search *always* spends the Counterspell (never-countered = 0/300
  across all N) and puts it on the bait Gray Ogre 61–70% of the time. Its
  30–39% "correct" runs are indistinguishable from the random floor (34%,
  of which 28% never cast at all). Exactly the "burns inherited
  counterspells on the first target" mechanism — the search's own root
  decision inherits the same pathology because every held-counter branch is
  evaluated by rollouts that waste it.
- **S2**: search casts Pyroclasm on turn 1, wiping exactly the 2 visible
  creatures, in **300/300 runs at every N** (mean cast turn 1.0, wiped 2.0).
  Random waits by accident often enough to reach 23%. Strategy fusion in its
  purest form: the "cast now" branch shows two guaranteed kills in every
  determinized world; the "wait" branch's better wipe never happens in
  random continuations.
- **S3**: the Lightning Bolt was cast in 100% of runs at every strength and
  the Earth King's Lieutenant died in **zero of 400 runs** (all agents).
  Search burns the bolt on turn 1 before the key creature is even castable:
  search-256 put 71% into the 1/1 decoy and 29% into the villain's face.
  There is no "hold removal for the real threat" anywhere in this pilot.
- **S4**: search races into a losing clock 92–97% of runs (random: 72%) —
  N=256 races *more* than N=16 holds. The rollout villain is uniformly
  random, not an attacker, so racing looks safe in every world; blocking
  value never materializes in random continuations.
- **S5**: search taps out on the Otter-Penguin 75–87% of runs; the full
  correct line (hold + quench the tapped-out Craw Wurm) happens ≤8%.
  Random, which passes the main phase 54% of the time by construction,
  scores 0.14 — again above every search cell.

The absolute levels of several predictions were wrong (S1's random floor was
far higher than predicted — random spends the counter somewhere and there
are only two spells to hit; S3's search numbers were far lower — turn-1
bolt-burning was even stronger than the mechanism argument suggested), but
every registered directional/structural claim (a–c) held.

### Micro-format table

300 games/cell, seat-balanced, both sides piloted by the same agent. Raw
data: `reports/data/exp-09-micro.json`. (A probe accounting bug —
`bolt_biggest` counted single-creature boards while its denominator did not
— was found and fixed mid-run; all probe-affected cells were re-run with
identical seeds and reproduced identical win rates, so the win-rate columns
double as a determinism check.)

| Cell | Hero win [Wilson 95%] | Predicted | On play / on draw | Turns |
| --- | ---: | ---: | ---: | ---: |
| aggro mirror @ s64 | **0.490** [.434,.546] | 0.50 ✓ | 0.27 / 0.71 | 24.4 |
| control mirror @ s64 | **0.517** [.460,.573] | 0.50 ✓ | 0.61 / 0.43 | 33.5 |
| control vs aggro @ random | **0.037** [.021,.064] | 0.30 ✗ | 0.03 / 0.05 | 26.1 |
| control vs aggro @ s16 | **0.393** [.340,.450] | 0.35 ~ | 0.17 / 0.61 | 21.6 |
| control vs aggro @ s64 | **0.630** [.574,.683] | 0.35 ✗ | 0.61 / 0.65 | 20.3 |
| control vs aggro @ s256 | **0.530** [.473,.586] | 0.33 ✗ | 0.47 / 0.59 | 22.9 |

Both mirror sanity checks pass. The registered "flat-to-declining in N"
win-rate claim is **refuted**: the control deck goes 3.7% under random
piloting (near-unplayable — its card quality is conditional on choices),
climbs steeply to 63.0% at N=64, then *declines* to 53.0% at N=256
(Δ = −0.10, CIs barely touching). Search rescues the control deck's *win
rate* in this micro format — but not, as the behavior below shows, by
playing control. Side notes: the aggro mirror shows a strong on-the-draw
advantage under search (0.71 vs 0.27), consistent with exp-08's finding
that the extra card beats tempo once piloting is competent.

### Behavioral stats (control side, control-vs-aggro)

| Metric | random | s16 | s64 | s256 | registered claim |
| --- | ---: | ---: | ---: | ---: | --- |
| counter casts / windows | 0.57 | 0.74 | 0.55 | 0.58 | — |
| counter at first-ever window | 0.21 | 0.28 | 0.28 | 0.29 | > 0.70 ✗ (level), pathology real (see below) |
| countered MV mean | 1.93 | 2.14 | 2.27 | 2.29 | ≤ 2.4 ✓ |
| countered MV ≤ 2 share | 0.55 | 0.44 | 0.38 | 0.36 | — |
| counters spent on 1-MV Raging Goblin | 17% | 30% | 24% | 27% | — |
| instant-holding rate | 0.57 | 0.40 | 0.39 | 0.40 | < 0.15 ✗ (level); flat-in-N and *below random* ✓ (structure) |
| bolt biggest-creature rate (multi-choice) | 0.39 | 0.34 | 0.61 | 0.69 | ≈ 0.5 — **rises with N** ✗ |
| bolt face share | 0.23 | 0.45 | 0.49 | 0.56 | > 0.30 ✓ |

Two registered levels were wrong but the split they were probing is real
and sharper than predicted: **the stats that improve with N are exactly the
immediate-value ones** (bolt target quality among simultaneously-visible
creatures: 0.34 → 0.69), and **the stats that require valuing the future
are flat or worse than random at every N** (instant-holding ~0.40 vs
random's 0.57; a quarter of all Counterspells still burned on 1-mana
Raging Goblins at N=256; countered-MV mean stuck ~2.3 in a format whose
real threats cost 3). Search-N buys discrimination *within* the present
decision, none *across* turns.

## Verdict

**H2 confirmed at the decision level — the flat-MC/random-rollout pilot
cannot play control, and more simulations do not help.** All three
pre-registered scenario claims held: no scenario above 0.39 at any N,
Δ(N=256 − N=16) ≤ 0.02 everywhere, and uniform random outright *beats*
every search strength on hold-the-wipe (0.23 vs 0.00), race-vs-block, and
hold-up-quench. The named mechanisms were both observed directly: strategy
fusion (Pyroclasm cast on turn 1 in 300/300 search runs; the ogres never
held back as blockers) and the rollout counter-burning pathology (the bolt
spent before the key threat even appears in 100% of S3 runs; the
Counterspell put on the bait 61–70% of S1 runs and *always* spent).

**Consequence for exp-08:** the UR-at-22% number says nothing about the UR
Lessons deck. The pilot it was measured under cannot execute any of the
five competencies that deck is built around. H1 (GW structurally favored)
is *neither confirmed nor refuted* — it is unmeasurable with this pilot,
and the micro-format result shows why the win-rate lens alone misleads in
both directions: a control-shaped deck can still post 63% under this pilot
when its reactive cards are individually strong enough to be misplayed as
tempo/value cards (Bolt as creature removal now, Recall as raw draw), while
a deck like UR Lessons whose value is locked behind *timing* (Divide by
Zero, It'll Quench Ya!, learn selection) gets no such rebate.

**The one-line summary:** search-N buys within-decision discrimination
(which creature to bolt, which attacks are safe) but zero cross-turn
discrimination (when to *not* act); every future policy should be run
through this suite, and any policy scoring near the search rows on S1–S5
should not be called a control player regardless of its win rate.

**Strongest single piece of evidence:** S2 hold-the-wipe — 300/300 search
runs at every strength cast Pyroclasm at the first legal moment, wiping
exactly the 2 visible creatures, while uniform random scores 23% correct by
sometimes doing nothing. Search is not merely failing to find the control
line; it is systematically *more* certain than chance to take the
immediate-value line.

**Next question (C10):** does policy-guided rollout (C5's expert-iteration
loop) move the scenario scores at all, or does the strategy-fusion ceiling
require root-level information-set handling (the Exit-1 belief-based
path)? The scenario suite is now the gate: a search variant that cannot
lift S2/S5 off zero is not a control player, whatever the ladder says.

## Caveats

- Scenario hero is always seat 0 (on the play); no seat balancing inside
  scenarios — the correct lines are seat-specific by construction.
- The scripted villain is not an equilibrium opponent; scenario scores are
  decision-quality probes, not win rates. Search agents model the villain as
  *random* inside rollouts (that mismatch is part of what is being measured
  under H2).
- Determinization resamples the villain's hand, so the "villain will cast X"
  knowledge is deck-composition-level, not certainty; scenario decks are
  threat-dense to keep the correct lines information-sound.
- State injection bypasses ETB triggers (scenario permanents arrive without
  entering-the-battlefield events) — positions are constructed, not reached.
- The exp-07 student checkpoints exist
  (`.../agent-aa323507a808e4181/.runs/exp07/*.pt`) but were trained on the
  pre-eval-stack observation encoding (player/card/permanent dims 27/37/7
  vs the current 38/24/28) and cannot load in this world; the skip is
  recorded per scenario in `exp-09-scenarios.json` under `student_r0`.
  Re-scoring a matching-dims student is one `--agents checkpoint:<path>`
  invocation away.
- Absolute levels of several pre-registered numbers were wrong (S1 random
  floor, S3 search levels, micro win-rate trajectory, behavior-stat
  levels); each miss is marked in the tables. The three registered
  *structural* claims (flatness in N, no scenario above 0.50, the S2
  do-nothing anomaly) and the decision rule all resolved cleanly.
- Micro behavioral rates aggregate over all 300 games of a cell (not
  per-game averages), and counter-target attribution uses the top of the
  stack at cast time (exact whenever one spell is on the stack, which is
  the overwhelmingly common case in these decks).
