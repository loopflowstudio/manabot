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

PENDING — filled in after the runs below.

### Scenario table

PENDING

### Micro-format table

PENDING

### Behavioral stats (control side, control-vs-aggro)

PENDING

## Verdict

PENDING

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
