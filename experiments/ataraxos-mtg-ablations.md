# Ataraxos-inspired Magic ablations

2026-10-04. Jack Heart requested these experiments. ETU-90 owns treatment
correctness; ETU-91 owns this study. The proposals below are not funded runs.

```bash
uv run --extra notebook experiments/runners/run_training_regimes.py --study ataraxos-ablations --profile smoke --out .runs/ablations
uv run --extra notebook experiments/runners/run_training_regimes.py --report-only .runs/ablations
```

The smoke runs all five core arms for two checkpoints and two updates per
checkpoint, on one initialization seed. Each checkpoint completes a four-leg
block against the control and fixed random anchor. It is deliberately too small
to accept or reject a treatment. Regimes in `experiments/regimes/` are bounded
pilot recipes; their counts do not encode the draft hour-long screen below.
Named add-ons have separate runnable recipes and bounded optimizer fixtures;
they are not secretly added to the core five-arm run list.

## Method inventory

| Technique | Disposition |
| --- | --- |
| Separate policy/value traces | Implemented scalar-value adaptation |
| Advantage quantile + threshold | Implemented before selected-row normalization |
| Uniform and action-type-uniform reference | Implemented MTG grouping; not Stratego piece grouping |
| Collection-policy KL | Full legal behavior distributions retained for each rollout |
| LR and reference schedules | Configurable independent power schedules; pilot constants |
| EMA | Implemented evaluation artifact per collect/update clock; collection remains raw |
| Categorical outcome values | Separate representation experiment; scalar values retained |
| Compound actions | Separate trainable decoder build |
| Belief sampling and search updates | Separate builds, not runnable stage names |

The historical reference is the accessible preprint appendix D.4/Table 22,
not a verified final Nature supplement. This is not an Ataraxos reproduction
or a claim of its game-theoretic guarantees.

## Second executable use: Ataraxos-inspired ablations for Magic

Deliver `experiments/ataraxos-mtg-ablations.md`, explicit regime variants,
and the same runner/notebook path, not a paragraph suggesting future tests.
This study asks which learning treatments transfer; it does not claim to
reproduce Ataraxos or prove its game-theoretic guarantees.

Use the five core independent fresh-training arms, all on the same corrected collector,
architecture, match, gamma=1, beta=.1 and common KL-reference choices:

| Arm | Value/policy lambda | Retained rows | LR and tau |
| --- | --- | --- | --- |
| RL control | .95 / .95 | All | Constant initial values |
| Separate estimators | 1 / .95 | All | Constant |
| Advantage filtering | .95 / .95 | Largest absolute 50% | Constant |
| Coordinated decay | .95 / .95 | All | Power schedules above |
| Combined | 1 / .95 | Largest absolute 50% | Power schedules above |

Add named recipe contrasts to make the additional review testable:

- **Horizon:** control at gamma=.99 versus gamma=1; separately increase policy
  lambda to .99 at gamma=1. Do not bundle discount and trace changes.
- **Paper estimators:** outcome lambda=.8 / policy lambda=.5 versus the .95/.95
  control. A scalar-value adaptation is labelled as such; the paper's
  categorical win/loss/draw value objective is a separate representation test.
- **Reference:** structured-uniform versus flat-uniform with all other
  settings fixed. **Step constraint:** beta=0 versus .1 collection-policy KL.
- **Paper filter:** upper quartile AND magnitude >=.01 versus the 50% filter,
  with identical estimates and sample accounting.
- **Averaging:** raw versus EMA checkpoint of each run; no extra training seed
  is manufactured by evaluating two outputs of the same run.

These contrasts are delivered as runnable recipes, but are not all charged to
the five-arm screen. Each add-on has an explicit run list and budget in the
experiment document. The short screen cannot silently select B's week-long
recipe on the same final evaluation deals.

Three paired initialization seeds per arm; one-factor additions identify
effects against the control. The combined arm tests whether the package helps;
it does not identify all interactions. A subsequent leave-one-out design is
warranted if combined performance contradicts the isolated effects. An optional
learning-rate-only versus tau-only study resolves an observed decay benefit;
it is not silently added to the initial budget.

Proposed separate screening profile: one hour/run (15 training hours total),
checkpoints at 15/30/60 minutes, 16 common four-leg development blocks per
anchor at each checkpoint, and 32 untouched blocks per arm/control seed pair
at the endpoint. Freeze a measured evaluation allowance before launch. These
hours are NOT extra work hidden inside the two-regime week. If both studies
must share that week, reduce the week allocation explicitly before execution.
Short screens identify candidates, not whether a treatment will win after a
week. The three full-budget RL seeds are not automatically independent of
screening if their initialization or deals were reused for selection.

Retain diagnostics that explain the result: raw and retained advantage
distributions by action type, policy entropy/KL, value residuals by distance
to termination, bootstrapped tail fraction, episode length, fraction of forced
steps skipped, optimizer examples/second, and collection/optimization time.
Analyze whether filtering selects critic mistakes or terminal proximity rather
than useful choices. For a short-horizon versus long-horizon claim, define
bins from a frozen development population before scoring; episode length
under a trained policy is an outcome, not an independent treatment variable.

The notebook produces an ablation effect plot with every seed visible,
cost-based learning curves, retained-example diagnostics and linked gameplay
traces. The report answers: retain a treatment, reject it at this budget, or
collect more evidence. Proposed screening threshold is a >=.05 mean anchor
score gain at equal cost; uncertainty and all three seed effects accompany
the decision. Confirmatory follow-up uses fresh seeds/deals. Negative and
inconclusive results remain first-class outputs.

Representational changes (autoregressive compound actions), frozen-policy
belief learning and learned-belief search are separate mechanism experiments,
not toggles with claimed implementations. Add a method inventory to the report
mapping each paper technique to implemented treatment, existing behavior,
separate build or omitted mechanism and its reason. This keeps the entire
Ataraxos learning agenda visible without conflating it with the first screen.


## Proposed additional allocations

Each contrast below uses three fresh paired initialization seeds and the same
15/30/60-minute checkpoints as the core screen. Each named arm receives one
training hour per seed. Its corresponding control is rerun with those seeds;
previous core runs are not silently reused as independent evidence. Evaluation
allowances remain unfrozen until calibration; none of these allocations is
authorized by the implementation task.

| Contrast | Recipe versus control | Proposed training hours |
| --- | --- | ---: |
| Discount | horizon-discount versus rl-control | 6 |
| Policy trace | horizon-trace versus rl-control | 6 |
| Paper estimators | paper-estimators versus rl-control | 6 |
| Reference | structured-reference versus rl-control | 6 |
| Step constraint | no-collection-kl versus rl-control | 6 |
| Filter | paper-filter versus advantage-filtering | 6 |
| Averaging | averaging versus rl-control; raw/EMA paired within averaging | 6 |

The equal-cost effect plot uses the common observed cost horizon and displays
every measured seed. If the arms have no overlapping cost range, the effect
is explicitly unavailable. One seed and one deal block cannot support treatment
selection: the smoke's decision is always to collect more evidence under a
separately frozen and funded protocol.
