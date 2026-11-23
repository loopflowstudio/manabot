# Forward KL and larger tau on TLA trophy policies

Fresh w64 policies trained on the TLA trophy decks lock their combat choices:
by update 100 they declare nearly every offered attacker and blocker, and the
gap between the two options keeps growing. The damped update's reverse KL
cannot pull a locked two-offer choice back, because its restoring pull
vanishes at large gaps. This exploratory study asks whether a small forward
KL to the reference (ε · KL(q ‖ π)) or a larger tau keeps those choices open,
and what that costs in playing strength. It makes no committed prediction.

**Status, 2026-11-23: final.** All 19 runs reached update 400 and every
held-out evaluation through 400 is complete. Seed 40902 of tau ×2 + forward
KL 0.002, tau ×2 and tau ×4 continued to update 1,000 and met in a final
round at update 1,000 on 2,000 deals drawn from every held-out pool.

## Answer

Every treatment removes the lock at update 400 and none costs strength
there. Each treated arm scores 0.70–0.75 against the heuristic at update
400, against the control's 0.64, and plays the control about evenly head to
head. Forward KL 0.01 also unlocks an already locked policy within 25
updates, but on fresh runs it over-regularizes: it learned more slowly and
trails the control head to head.

Longer training separates the two tau-based mechanisms. By update 1,000 tau
alone re-locks part of combat (tau ×2: 35% of probe attacks and 17% of
blocks saturated; tau ×4: 27% and 12%), while tau ×2 + forward KL 0.002
stays open (no saturated attacks, 3% of blocks). But the open policy is
slightly weaker: in the final round it scores 0.466 against tau ×2
(3,584 games, ±0.008) and 0.488 against tau ×4, and 0.750 against the
heuristic to tau ×2's 0.782. Tau ×2 and tau ×4 are level (0.502 over 4,168
games). The partial re-lock does not cost tau ×2 strength at this horizon.

**Recommended default for new trophy runs:** tau_scale 0.1 (tau ×2) with no
forward-KL floor. The baseline recipe now trains at tau_scale 0.1. The floor
remains available (`reference_forward_kl`) for when open combat choices
matter more than a few points of strength.

| Arm | Seeds | Score vs heuristic, 400 | vs control, 400 | Attackers declared, 400 | Blockers declared, 400 | Saturated attacks, probe 400 | Rare offers, 8+ offers, 200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control (ε = 0, τ 0.05) | 3 | 0.64 | — | 0.97 | 0.86 | 36–65% | 27.5% |
| Forward KL 0.002 | 3 | 0.71 | 0.51 | 0.84 | 0.69 | 0% | 8.8% |
| Forward KL 0.01 | 3 | 0.70 | 0.46 | 0.81 | 0.68 | 0% | 0.0% |
| Tau ×2 (τ 0.1) | 2 | 0.73 | 0.50 | 0.78 | 0.75 | 9–25% | 16.5% |
| Tau ×4 (τ 0.2) | 2 | 0.72 | 0.52 | 0.77 | 0.71 | 1–2% | 7.9% |
| Tau ×2 + forward KL 0.002 | 2 | 0.74 | 0.53 | 0.85 | 0.71 | 0% | not measured |

Scores are the held-out checkpoint's share of points over 32 deals with both
seats and both decks (128 games per seed). "vs control" pairs each treated
checkpoint with every control seed at the same update. Declaration rates
divide declarations by offers in those held-out games. Saturated attacks are
the share of probe decisions whose declare-vs-decline gap exceeds 9.2 log-odds
at the arm's last probe. Rare offers are the share of legal offers below
0.01% probability on the 81 held-out decisions with eight or more offers.
Seeds average equally; arms with one seed at a checkpoint say so.

![Run legend](../data/mtg408/report/legend.png)

## Do the treatments keep combat choices open?

Yes. During training the control's mean attack gap reaches 8–11 log-odds and
its block gap 12–24 in two of three seeds; 36–65% of its probe attacks and
up to 96% of its blocks are saturated. The treated arms hold both gaps near
1–4. One exception: tau ×2 alone peaks at an attack gap of 5.3–5.5 near
update 300 and has 9–25% of its probe attacks saturated at 400, so larger
tau alone delays the lock rather than removing it. Tau ×2 + forward KL 0.002
stays the most open (attack 1.7–2.7, block 0.1–0.6, no saturation).

![Attack and block log-odds](../data/mtg408/report/combat-probe.png)
![Saturated combat decisions](../data/mtg408/report/saturation.png)

In held-out play the control declares 97–99% of offered attackers from update
100 on. The treated arms declare 77–89%.

![Attackers declared](../data/mtg408/report/attackers.png)
![Blockers declared](../data/mtg408/report/blockers.png)

## Does the floor pull an already locked policy back?

Yes, within 25 updates. Both warm starts begin from MTG-409's update-400
seed 40902 policy, whose probe gaps were 11.3 (attack) and 21.2 (block).
Without the floor the policy stays locked; with forward KL 0.01 it opens and
stays open.

| Warm start, 100 updates | Attack gap, 25 / 100 | Block gap, 25 / 100 | Saturated attacks, 100 | Saturated blocks, 100 |
| --- | ---: | ---: | ---: | ---: |
| ε = 0 | 6.2 / 6.6 | 12.5 / 11.2 | 16% | 61% |
| ε = 0.01 | 2.2 / 2.4 | 0.9 / 0.3 | 0% | 0% |

After unlocking, the ε = 0.01 warm start attacks less into an untapped 9/9 as
training continues (shift −0.4 at update 25, −1.1 at 100). The warm starts
are one seed each and have no held-out evaluation.

## Do the choices still follow the board?

Mostly. The probe edits each position — adding a 0/1 or a 9/9 on either
side — and measures how far the declare gap moves. Every arm moves in the
sensible direction. The control's shifts are larger in log-odds but sit on
gaps of 8–24, where its probabilities barely change; the treated arms sit
near the decision boundary, so shifts of 2–3 change play. Forward KL 0.002
attacks less into an untapped opposing 9/9 than the control (−3.5 against
−1.7 at update 400). Forward KL 0.01 moves least.

![Attack gap shift under board edits](../data/mtg408/report/attack-shifts.png)
![Block gap shift under board edits](../data/mtg408/report/block-shifts.png)

## What does it cost in strength?

Nothing measurable. Against the heuristic at update 400 every treated arm
scores 0.70–0.75, against the control's 0.64. Head to head with the control
at update 400, forward KL 0.002, tau ×2, tau ×4 and the combined arm score
0.50–0.53. Forward KL 0.01 learned more slowly — 0.60 and 0.63 against the
heuristic at updates 200 and 300 before reaching 0.70 at 400 — and still
trails the control head to head (0.43 at 200, 0.46 at 400).

![Score against the heuristic](../data/mtg408/report/score.png)
![Against the control at the same update](../data/mtg408/report/vs-control.png)

Every arm beats its own checkpoint from half as many updates, 0.55–0.59 at
update 400, so all of them were still improving.

![Against itself at half its updates](../data/mtg408/report/half-training.png)

## What happens with longer training?

Seed 40902 of the three tau arms continues past update 400 (dotted line) to
update 1,000. Every arm keeps improving: each scores 0.56–0.62 against its
own checkpoint from half as many updates at 1,000, and 0.78–0.81 against the
heuristic on the original held-out deals. Combat behaviour diverges. Tau ×2
alone keeps 10–40% of its probe attacks saturated from update 250 on; tau ×4
drifts toward a lock after 600 as its tau decays to 0.017; both start
saturating blocks after 600. The combined arm's attack gap stays mostly at 1–3
with no saturated attacks and under 5% of blocks saturated throughout.

![Seed 40902 through update 1,000](../data/mtg408/report/extensions.png)

| Seed 40902 | vs heuristic, 400 / 600 / 800 / 1,000 | vs half its updates, 1,000 | Attack gap, 1,000 | Saturated attacks, 1,000 | Saturated blocks, 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tau ×2 + forward KL 0.002 | 0.75 / 0.74 / 0.76 / 0.78 | 0.60 | 1.2 | 0% | 3% |
| Tau ×2 | 0.72 / 0.75 / 0.78 / 0.80 | 0.56 | 5.1 | 35% | 17% |
| Tau ×4 | 0.71 / 0.75 / 0.76 / 0.81 | 0.62 | 5.4 | 27% | 12% |

The update-400 points are 128 games; later points are 570–700 games of the
original 32 held-out deals plus 128 more. Head to head at matched updates
from 500 to 1,000, pooled, the combined arm scores 0.43 against tau ×2
(1,540 games) and 0.51 against tau ×4 (1,844); tau ×4 scores 0.48 against
tau ×2 (2,124). These panels are drawn from the seed's own records: the
Etude page's curves average a job's seeds and would jump at update 400
where seed 40901 stops.

### Final round at update 1,000

The update-1,000 checkpoints played 2,000 deals drawn from every one of the
873 held-out pools, each deal with both seats. The round against the
heuristic stopped once each arm had 1,600–3,700 games, enough for ±0.01.

![Final round at update 1,000](../data/mtg408/report/final-round.png)

| Pairing at update 1,000 | Score of first | Games | 95% interval |
| --- | ---: | ---: | ---: |
| Tau ×2 + forward KL 0.002 vs tau ×2 | 0.466 | 3,584 | 0.450–0.482 |
| Tau ×2 + forward KL 0.002 vs tau ×4 | 0.488 | 3,896 | 0.472–0.504 |
| Tau ×4 vs tau ×2 | 0.502 | 4,168 | 0.487–0.517 |
| Tau ×2 + forward KL 0.002 vs heuristic | 0.750 | 3,260 | 0.735–0.765 |
| Tau ×2 vs heuristic | 0.782 | 1,616 | 0.762–0.802 |
| Tau ×4 vs heuristic | 0.769 | 3,748 | 0.755–0.782 |

Intervals treat games as independent; both seats of a deal share its decks,
so the true intervals are somewhat wider.

## Cards: Manabot against 17Lands players

Every card-evidence game in which both seats were Manabot checkpoints
(41,372 player-games, so Manabot's overall win rate is exactly 50%) gives a
games-in-hand win rate (GIH WR) per card. The 17Lands public TLA Premier
Draft game dump (562,924 games, CC BY 4.0) gives the human GIH WR for the
same cards. 299 cards have both; the median card has 1,132 Manabot GIH
games and 166 cards have at least 1,000.

Measured against each side's own average, Manabot's card values correlate
0.38 with humans' over those 166 cards. Cards Manabot values most above humans include
Wartime Protestors, Diligent Zookeeper and The Boulder, Ready to Rumble
(10–14 points above humans relative to average); those it values least
include Cunning Maneuver, Sun-Blessed Peak, Secret Tunnel and Airbender's
Reversal (8–16 points below). Decks are fixed trophy decks, so a card's
GIH WR also carries its deck's strength.

[Per-card table: Manabot vs 17Lands GIH WR](../data/mtg408/cards.html)

## Does the zero-weight control reproduce MTG-409?

Yes. MTG-409's `tla_trophies/trophy-w64-raw` (`f9c19a2656`, the same recipe
before the floor existed) is the historical control. Seed by seed it shows
the same combat pattern, including seed 40903's open blocks.

| Seed | Study | Score | Attack declared | Block declared | Attack gap | Block gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 40901 | MTG-409 | 71.9% | 91.8% | 100% | 6.87 | 17.64 |
| 40901 | Fresh control | 67.2% | 93.5% | 100% | 9.84 | 15.38 |
| 40902 | MTG-409 | 68.0% | 99.7% | 100% | 11.33 | 21.19 |
| 40902 | Fresh control | 60.2% | 99.9% | 100% | 10.88 | 24.01 |
| 40903 | MTG-409 | 68.0% | 99.0% | 64.1% | 6.68 | 0.76 |
| 40903 | Fresh control | 65.6% | 97.3% | 57.0% | 7.71 | −0.82 |

The fresh control's mean final score, 64.3%, is five points below MTG-409's
69.3%; matched seeds still diverge in what they learn. The ε = 0 path is
inert, as the zero-weight loss regression checks directly.
[MTG-409 report](tla_trophies.md#held-out-evaluation-through-update-400)

## Learning dynamics

Tau follows the trophy schedule, 0.05 / position^0.3 with the 400-update
run mapped onto a 10,000-step curve; the tau arms scale it by two and four.

![Policy damping](../data/mtg408/report/tau.png)
![KL divergence](../data/mtg408/report/kl.png)
![Policy entropy](../data/mtg408/report/entropy.png)
![Losses](../data/mtg408/report/losses.png)
![Gradient norm](../data/mtg408/report/grad-norm.png)

## Cost

USD35 approved. The launcher's per-run estimates total USD33.52: USD25.61
for the 19 runs to update 400, the rest for extending seed 40902 of the
three tau arms to update 1,000. Evaluation ran on owned machines.

## Limits

- Seeds: three per forward-KL arm, two per tau arm. Differences of a few
  points in score are within seed spread; the head-to-head games, which pair
  every treated seed with every control seed, are the tighter comparison.
- Length: 400 updates for the comparison, 1,000 for one seed of three
  arms. The policies are still improving at 1,000, and the final-round
  verdict is one training seed.
- Rare offers were measured at update 200, before the combined arm existed.
- Evaluation failures: about a dozen of ~3,000 matches through update 400
  failed and are excluded, and a few dozen of the final round's matches. Most
  are the card-capture bug (Task a383b2, then MTG-415: "unrecorded physical
  copy"), plus one interrupted match and one observation over the 32-slot
  stack. If the capture bug favours some decks, those decks are slightly
  under-counted in every arm alike.
- Native panics (Task 6213de) occurred in training games; the game error
  policy recovered them.
- Rare offers come from 81 positions played by one frozen control
  checkpoint.

<details>
<summary>Appendix: method, runs and history</summary>

### Arms and schedule

The recipe at `e5d6e045e` declares freshly initialized arms with seeds
40901–40903 (forward KL) or 40901–40902 (tau), up to 400 updates and raw
checkpoints every 100. Warm starts (`pullback-0`, `pullback-0.01`) begin
from MTG-409's update-400 seed 40902 checkpoint with fresh Adam and a fresh
damping schedule for 100 updates. Evaluation per arm: 32 held-out deals
against the heuristic; the same deals against the checkpoint at half the
updates; and treated arms against every control seed at updates 200 and 400.

### CPU smoke, 2026-11-20

At update 400 of MTG-409's raw trophy run (`4c38775ec499`), seed 40902 had
the largest mean attack and block gaps (11.33 and 21.19). Each of the five
original jobs completed one local CPU update with two streams; Etude verified
the warm-start file against SHA-256
`896ee236e0227ae4b94f68e2732b8955a2a7c7c1cb9f9415e62a11296c7d3ce4`. On one
small batch, one update at ε = 0.01 cut saturated attacks from 15/19 to 7/19.

### Launches and resumes

Etude admitted the first eleven runs at `6ae723eb8421` on 2026-11-21 05:37
UTC. The tau ×2 and ×4 arms launched at 16:50 UTC and the combined arm at
20:20 UTC. Several runs stopped at placement deadlines (exit 75) and resumed
from their saved state to update 400: `floor-0.002@40901` at 321, the three
`floor-0.01` seeds at 313–318 and `tau-x4@40902` at 298. `tau-x2@40901` was
cancelled at update 215 by an operator reading stale status and resumed from
214. The job log of `floor-0.002@40901` retained a native trigger-mode panic.
All 19 runs reached update 400 by 01:10 UTC on November 22. The extensions
to 1,000 resumed seed 40902 of three arms from their update-400 state at
about 02:30 UTC on one rented three-slot L4.

### Held-out controls at initialization

All three update-zero controls scored 7/128 against the heuristic and
declared 43% of offered attackers and 60% of offered blockers.

</details>
