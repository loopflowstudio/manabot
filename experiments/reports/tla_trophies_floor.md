# Forward KL and larger tau on TLA trophy policies

Fresh w64 policies trained on the TLA trophy decks lock their combat choices:
by update 100 they declare nearly every offered attacker and blocker, and the
gap between the two options keeps growing. The damped update's reverse KL
cannot pull a locked two-offer choice back, because its restoring pull
vanishes at large gaps. This exploratory study asks whether a small forward
KL to the reference (ε · KL(q ‖ π)) or a larger tau keeps those choices open,
and what that costs in playing strength. It makes no committed prediction.

**Status, 2026-11-21 22:15 UTC.** Control, forward KL 0.002, tau ×2 and both
warm starts are complete. Forward KL 0.01 and tau ×4 seed 40902 are resuming
to update 400, and the combined tau ×2 + forward KL 0.002 arm is at about
update 150. Cells marked — are not yet played.

## Answer so far

Every treatment removes the lock and none costs strength. Forward KL 0.01
also unlocks an already locked policy within 25 updates, which the damped
update alone does not. The two tau arms
and forward KL 0.002 score higher against the heuristic than the control at
update 400, and they play the control evenly head to head. Forward KL 0.01
removes rare offers entirely but trails the other arms in strength.

| Arm | Seeds | Score vs heuristic, 400 | vs control, 400 | Attackers declared, 400 | Blockers declared, 400 | Saturated attacks, probe | Rare offers, 8+ offers, 200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control (ε = 0, τ 0.05) | 3 | 0.64 | — | 0.97 | 0.86 | 0.36–0.65 | 27.5% |
| Forward KL 0.002 | 3 | 0.69 | 0.50 | 0.82 | 0.69 | 0.00–0.02 | 8.8% |
| Forward KL 0.01 | 3 | 0.63 (300) | 0.43 (200) | 0.81 (300) | 0.67 (300) | 0 | 0.0% |
| Tau ×2 (τ 0.1) | 2 | 0.73 | 0.50 | 0.78 | 0.75 | 0 | 16.5% |
| Tau ×4 (τ 0.2) | 2 | 0.73 (1 seed) | 0.53 | 0.84 | 0.67 | 0 | 7.9% |
| Tau ×2 + forward KL 0.002 | 2 | — | — | — | — | — | — |

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
up to 96% of its blocks are saturated. Every treated arm holds both gaps near
1–3 with essentially no saturated decisions.

![Attack and block log-odds](../data/mtg408/report/combat-probe.png)
![Saturated combat decisions](../data/mtg408/report/saturation.png)

In held-out play the control declares 97–99% of offered attackers from update
100 on. The treated arms declare 78–87%.

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

Nothing measurable for forward KL 0.002 or either tau arm. Against the
heuristic at update 400 they score 0.69–0.73, against the control's 0.64.
Head to head with the control at matched updates they score 0.47–0.53.
Forward KL 0.01 learned more slowly: 0.60 and 0.63 against the heuristic at
updates 200 and 300, and 0.43 against the control at 200.

![Score against the heuristic](../data/mtg408/report/score.png)
![Against the control at the same update](../data/mtg408/report/vs-control.png)

Every arm beats its own checkpoint from half as many updates, 0.56–0.60, so
all of them were still improving at 400.

![Against itself at half its updates](../data/mtg408/report/half-training.png)

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

USD35 approved for the experiment. Etude's per-run estimates total about
USD22 so far across 17 runs on rented L4s; evaluation ran on owned
machines.

## Limits

- Seeds: three per forward-KL arm, two per tau arm; tau ×4 has one seed at
  update 400 so far. Differences of a few points in score are within seed
  spread.
- Length: 400 updates. The policies are still improving; a 1,000-update
  follow-up on one seed of the leading arms is the next step.
- Evaluation failures: about a dozen of ~3,000 matches failed and are
  excluded — a card-capture bug on cards entering from outside the game
  (Task a383b2), one interrupted match and one observation over the 32-slot
  stack. If the capture bug only fires on Lesson decks, those decks are
  slightly under-counted in every arm alike.
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

### Held-out controls at initialization

All three update-zero controls scored 7/128 against the heuristic and
declared 43% of offered attackers and 60% of offered blockers.

</details>
