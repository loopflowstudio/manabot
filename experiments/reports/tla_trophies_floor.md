# Forward KL and larger tau on TLA trophy policies

Fresh w64 policies trained on the TLA trophy decks lock their combat choices:
by update 100 they declare nearly every offered attacker and blocker, and the
gap between the two options keeps growing. The damped update's reverse KL
cannot pull a locked two-offer choice back, because its restoring pull
vanishes at large gaps. This exploratory study asks whether a small forward
KL to the reference (ε · KL(q ‖ π)) or a larger tau keeps those choices open,
and what that costs in playing strength. It makes no committed prediction.

**Status, 2026-11-22 06:40 UTC.** All 19 runs reached update 400 and every
held-out evaluation through 400 is complete. Seed 40902 of tau ×2 + forward
KL 0.002, tau ×2 and tau ×4 is extending to update 1,000; it has reached
610–650, and the section on longer training reports it through 600.

## Answer

Every treatment removes the lock and none costs strength. Each treated arm
scores 0.70–0.75 against the heuristic at update 400, against the control's
0.64, and plays the control about evenly head to head. Tau ×2 + forward KL
0.002 is the most open and the strongest: no saturated decisions, 0.74
against the heuristic and 0.53 against the control. Tau ×2 alone is starting
to re-lock by update 400 (9–25% of probe attacks saturated), so the floor
matters most as tau keeps decaying. Forward KL 0.01 also unlocks an already
locked policy within 25 updates, but on fresh runs it over-regularizes: it
learned more slowly and trails the control head to head.

**Recommended default for new trophy runs:** tau_scale 0.1 with a 0.002
forward-KL floor, pending the 1,000-update extensions.

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

Seed 40902 of the three tau arms continues past update 400 (dotted line).
Through update 600 every arm still holds 0.70–0.77 against the heuristic,
and tau ×2 alone keeps its partial lock rather than deepening it: 20–34% of
its probe attacks are saturated from update 400 on, against none for the
combined arm. Tau ×4 begins to drift toward a lock near update 600 (attack
gap up to 4.7, up to 11% saturated) as its tau decays to 0.019; the combined
arm's attack gap stays at 1–3 with no saturated attacks and under 3% of
blocks saturated.

![Seed 40902 past update 400](../data/mtg408/report/extensions.png)

| Seed 40902 | vs heuristic, 400 / 500 / 600 | vs half its updates, 400 / 600 | Attack gap, 600 | Saturated attacks, 600 | Saturated blocks, 600 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tau ×2 + forward KL 0.002 | 0.75 / 0.70 / 0.73 | 0.63 / pending | 2.0 | 0% | 3% |
| Tau ×2 | 0.72 / 0.77 / 0.75 | 0.57 / 0.55 | 5.2 | 27% | 3% |
| Tau ×4 | 0.71 / 0.73 / 0.77 | 0.58 / 0.58 | 4.1 | 6% | 3% |

Each held-out point is 128 games of one seed, so differences of 0.05 are
within noise; tau ×2's update-600 point has 48 games so far. Against the
half-update checkpoint the arms still score above 0.5, so all three are
still improving at 600. These panels are drawn from the seed's own records:
the Etude page's curves average a job's seeds and would jump at update 400
where seed 40901 stops.

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

USD35 approved for the experiment. The 19 training runs to update 400 cost
USD25.61 by Etude's per-run estimates, on rented L4s; evaluation ran on
owned machines. The 1,000-update extensions share one L4 rental capped at
USD6.60 to 11:11 UTC, with a second short rental to finish the remaining
updates.

## Limits

- Seeds: three per forward-KL arm, two per tau arm. Differences of a few
  points in score are within seed spread; the head-to-head games, which pair
  every treated seed with every control seed, are the tighter comparison.
- Length: 400 updates for the comparison, 600 so far for one seed of three
  arms. The policies are still improving.
- Rare offers were measured at update 200, before the combined arm existed.
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
All 19 runs reached update 400 by 01:10 UTC on November 22. The extensions
to 1,000 resumed seed 40902 of three arms from their update-400 state at
about 02:30 UTC on one rented three-slot L4.

### Held-out controls at initialization

All three update-zero controls scored 7/128 against the heuristic and
declared 43% of offered attackers and 60% of offered blockers.

</details>
