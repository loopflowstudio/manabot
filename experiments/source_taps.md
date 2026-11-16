# Does choosing mana sources one tap at a time slow early learning?

Exploratory pilot (MTG-347), 2026-11-14. Code: `experiments/source_taps.py` at
`3367815384`.

## Answer

No slowdown is visible in the first 75 updates. Against the heuristic, the arm
that picks sources tap by tap scores 0.427 at update 75 and the automatic-payer
arm scores 0.425, with four seeds each. The per-seed differences at every
checkpoint have mixed signs and are smaller than the spread between seeds.

The tap arm does pay for the extra decisions: it makes 9 to 21% more learner
decisions per update and takes about 6% longer on the clock to reach update 75.

## Arms

Both arms train the Elemental recipe and spend pooled mana automatically
(`MANAGYM_EXPERIMENT_AUTOSPEND=1`). They differ in one switch:

| Arm | `MANAGYM_EXPERIMENT_SOURCE_TAPS` | Payment decision |
| --- | --- | --- |
| automatic | 0 | `PayCost` lets the automatic payer choose the sources. |
| taps | 1 | Each fixed source that helps pay is offered as a tap step beside `PayCost`, which finishes the payment automatically. |

A source is offered only while tapping it raises the number of cost units the
pool can cover and the cost stays payable; no step is offered when every
remaining useful source must be tapped anyway. The heuristic opponent takes
`PayCost` in both arms.

## Score against the heuristic

Mean score over 200 replay-checked games per checkpoint (50 Elemental pairings,
both seats, on the play and on the draw), by training seed.

| Update | Arm | 34701 | 34702 | 34703 | 34704 | Mean | SD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | automatic | 0.225 | 0.240 | 0.215 | 0.190 | 0.217 | 0.021 |
| 20 | taps | 0.170 | 0.190 | 0.215 | 0.200 | 0.194 | 0.019 |
| 40 | automatic | 0.310 | 0.305 | 0.280 | 0.315 | 0.302 | 0.016 |
| 40 | taps | 0.305 | 0.345 | 0.305 | 0.330 | 0.321 | 0.020 |
| 75 | automatic | 0.430 | 0.390 | 0.470 | 0.410 | 0.425 | 0.034 |
| 75 | taps | 0.435 | 0.385 | 0.440 | 0.450 | 0.427 | 0.029 |

Taps minus automatic, same seed: update 20 −0.055, −0.050, 0.000, +0.010 (mean
−0.024); update 40 −0.005, +0.040, +0.025, +0.015 (mean +0.019); update 75
+0.005, −0.005, −0.030, +0.040 (mean +0.003).

## Cost of the extra decisions

Means over four seeds at each checkpoint.

| Update | Learner decisions, automatic | Learner decisions, taps | Hours, automatic | Hours, taps |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 0.95M | 1.15M | 0.182 | 0.211 |
| 40 | 1.78M | 2.10M | 0.340 | 0.386 |
| 75 | 3.58M | 3.89M | 0.675 | 0.715 |

The gap narrows with training: 21% more decisions at update 20, 9% at update
75. Before training, local mirror games gave decisions per game of 305.6
automatic against 323.4 taps under the heuristic (+5.8%) and 213.2 against
276.7 under random play (+30%) on the Elemental decks.

At equal learner decisions the tap arm is therefore slightly behind at update
20 (0.194 at 1.15M against 0.217 at 0.95M) and level by update 75.

## Cost

Launcher figures: eight training runs on rented L4 machines about USD3.9
(USD0.47 to 0.51 each); two evaluation runs on rented 32-vCPU machines about
USD0.39.

| Run | Arm | Seed |
| --- | --- | ---: |
| `ffe2e017862b` | automatic | 34701 |
| `d4e0478c02a0` | automatic | 34702 |
| `ac790a85f0b2` | automatic | 34703 |
| `40156a45b18d` | automatic | 34704 |
| `8333f56c94a5` | taps | 34701 |
| `a26afd557369` | taps | 34702 |
| `000cef54013b` | taps | 34703 |
| `0dd8cd6d3a1a` | taps | 34704 |

Evaluation runs: `source-taps-automatic-120d86c254d2`,
`source-taps-taps-71b18f143f90`, 600 matches each, all completed and valid.

## Limits

- 75 updates is early: scores are still below the heuristic's 0.5, and a
  difference that appears later would not show here.
- Four seeds and 200 games per checkpoint resolve differences of roughly 0.05,
  not smaller ones.
- The heuristic never chooses sources, so the score measures whether tap
  choice costs learning, not whether a learner uses it well. Whether the tap
  arm leaves better mana untapped was not measured.
- Elemental decks rarely make source choice matter: offline, a payment that
  strands an affordable follow-up was about 5% of probed payments.
- The automatic payer in both arms is today's, which can tap more than the
  cost needs.
