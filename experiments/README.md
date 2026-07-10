# experiments/

One experiment = one report (`exp-NN-<name>.md`). Raw data in `data/`.
Run provenance lives in the verify store (`.runs/verify.sqlite`).
**Runners:** the exp-specific driver scripts live in [runners/](runners/) —
`manabot/` keeps only reusable instruments.

## Discipline

1. **Predict first, in git.** Question, numeric prediction, kill criteria,
   and cost cap are committed *before* the run. No pre-run commit → the
   result is exploratory, not evidence.
2. **Name the strongest confound.** Every report says how its result could
   be wrong and what would discriminate.
3. **Mechanism over aggregates.** "Can/cannot" claims need a behavioral
   probe (competency scenario, action-level stat, per-bucket metric) — win
   rates alone are strength claims only.
4. **Numbers trace.** Every number in any doc traces to a report; every
   report traces to store rows or data files. Refutations stay on the
   record, dated, never silently edited.
5. **Seeds are the unit.** Game-level CIs quantify one checkpoint's eval
   noise; claims about a *method* need independent training seeds and
   cross-seed uncertainty. Three seeds are three data points.
6. **Protect the instrument.** Engine determinism, throughput, state
   injection, and search primitives are what make experiments cheap —
   changes that break them are failing changes regardless of green tests.

## Index

| exp | question | verdict |
|---|---|---|
| [00](exp-00-decision-profile.md) / [00-cost](exp-00-cost-basis.md) / [00c](exp-00c-seat-balanced-baselines.md) | calibrate the instrument | 194 decisions/game; seat advantage 94% (inverts on interactive deck); single-init baselines meaningless; $0.44/1M steps |
| [01](exp-01-c1-training.md) | does the shaped recipe survive a real deck? | no — 0/3 seeds; one seat-parasitic; `cast_when_able` flipped sign |
| [02](exp-02-flat-mc.md) | how much intelligence is free? | search-256 = 99% vs random at $0; every trained policy below N=16 |
| [03](exp-03-distillation.md) | is distillation cheaper than RL? | yes — 90.5% vs matched-cost PPO's 52.7%; ladder ≈ N=8 at 1ms |
| [04](exp-04-potential-shaping.md) | is bias-free dense signal possible? | shaping was the disease — terminal-only wins; pass-collapse fails replication |
| [07](exp-07-expert-iteration.md) | does the expert-iteration crank compound? | no — R1 loses 74–26 to R0 (label economics); batched inference 12× landed |
| [06](exp-06-newworld-training.md) | did observation growth hurt training? | benign — seeds in/above the historical band |
| [08](exp-08-two-deck-matchup.md) / [08b](exp-08b-ancestral-dose.md) | what decides a matchup table? | pilot + card quality — 4× Recall swung it 55 points; UR-22% was a pilot artifact |
| [09](exp-09-control-competency.md) | can the pilot play control? | no — ≤0.39 correct at any N; random beats search on 3/5 scenarios; win rate masks incapacity |
| [10](exp-10-value-gate.md) | does search-with-V beat V-greedy? | gate 57.5% (marginal) but V loses to random rollouts at ¼ sims; Spearman 0.23 in interaction states |
| [11](exp-11-curriculum-exploitability.md) | does a stronger opponent teach better? | self-play quietly wins; the opponent installs the strategy; student robust (exploiter ≤26%) |

(Older platform docs: [first-light-run-1](first-light-run-1.md),
[sps-closeout](sps-closeout.md).)
