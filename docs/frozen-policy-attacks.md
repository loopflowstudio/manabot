# Frozen-policy attack studies

An `AttackPlan` defines exact target checkpoint bytes, producer seeds, independent
attacker seeds, a cumulative update ladder, final deal seeds and a complete cost
allocation. It maps every target/attacker-seed pair to an ordinary TrainingRegime
using the existing frozen-opponent collector. Every rung continues the same
optimizer and collector. A target checkpoint remains frozen; only the attacker
receives gradients. This is bounded adversarial evaluation, not exact exploitability.

Freeze the JSON plan before executing:

```bash
uv run python -m experiments.runners.run_attack_plan --plan attack.json --out .runs/attack-001
```

Use a fresh output directory. `AttackPlan` lives in
`manabot/training/attacks.py`; execution and reporting live in
`manabot/training/attack_execution.py`. Start with the selected-match self-play
recipe as `template`, retaining exactly one initialization stage. Supply:

- `targets`: unique IDs, exact `policy.path` and SHA-256, and `producer_seeds`.
- `attacker_seeds`: at least two distinct initialization seeds, separate from
  producers; scientific comparisons should declare their full seed cohort.
- `cumulative_updates`: strictly increasing rungs, for example `[100, 300, 900]`.
- `final_deal_seeds`: untouched deals shared across all rungs for paired scoring.
- `template`: one self-play TrainingRegime; the runner inserts the frozen target
  and turns cumulative rungs into continued incremental stages.
- `total_seconds`, `evaluation_seconds`, `game_seconds`, `max_commands` and a
  predeclared `prediction`. Total allocation must cover every training run plus
  evaluation. Each template wall cap applies to the full attacker, not each rung.

The evaluator currently admits only the selected Lessons/Allies matchup and
ordinary CPU policy checkpoints. Compound-submission and belief-input targets
are rejected before training because this collector cannot supply their inputs. Checkpoints must include that matchup's semantic
program input. Target admission checks world/setup, observation ABI and frozen
bytes. Literal seed reuse across producer/attacker and reserved executor seed
namespaces is rejected. This does not mathematically prove disjoint PRNG paths;
inspect retained game identities and data provenance when freezing a real study.

## Evidence and interpretation

Each cohort retains `plan.json`, `result.json`, `report.md`, canonical training
records in `training.sqlite`, all TrainingRun directories, and arena registrations,
rows, traces and replay receipts. Planned run paths enter the result before
training so failures are visible even before an ID is returned. A failed cell
stops the cohort; no unsuccessful seed is omitted from a reported complete result.

Every attacker faces its target before training and at every rung, with all four
seat/deck assignments. Only terminal, nontruncated, replay-verified games count.
Reports show actual training time and update counts; updates are not equal wall
cost across architectures. Final paired improvement uses the same deals and
resamples attacker seeds for an exploratory percentile bootstrap interval.
Targets are reported separately: this does not include uncertainty across
producer seeds or certify robustness against stronger attacks. Small cohorts
and few deals support workflow checks, not confirmatory statistical conclusions.

The POSIX main-thread runner owns a total deadline and an evaluation sub-budget.
It restores the prior signal handler and thread count, retains failure state,
and refuses an already active timer. These limits are execution guards; calibrate
a full scientific cohort before launch. This task allocates no additional time
beyond the existing campaign or paid hardware. Scientific target-policy attacks
need a separately frozen allocation; the active ETU-91 run is unchanged.

`scenarios.json` records legal scripted contrasts for historical S1–S5. All five
fixtures resolve their intended contrasts on the tested runtime, but their
custom decks differ from the selected matchup. They are explicitly marked
unsupported for selected-checkpoint scoring. Their results are mechanism checks,
not full-game arena strength, historical rating parity or optimal-strategy proof.

## Development evidence

The selected-world workflow smoke executed two attacker seeds, baseline plus two
rungs, one four-leg deal per cell: 24/24 games replayed in 45.68 seconds. Scores
were unchanged across rungs (.5 and .75 for the two seeds). This is expected at
the tiny update budget and makes no strength claim.

The real-PPO positive control uses a deliberately passive frozen toy policy and
a 40-card Mountain/Gray Ogre deck. With initial pass bias 4, 64 updates yielded
65,536 learner transitions and 1,086 training games in 25.28 seconds. On 64 fresh
games before and after training, score increased from .59375 to 1.0. This
establishes sensitivity to one known weakness only.

Retained fixture-development failures: Grizzly Bears was absent (no training
ran); a 16-card supported deck scored .5 before and after 16 updates (7.10s);
a 40-card deck with pass bias 3 started at .96875 and ended at 1.0 (7.36s), leaving
insufficient improvement margin; bias 5 scored .515625 to .5 (8.11s); bias 4 with
16 updates scored .59375 to .640625 (8.26s). The final test kept its seed and
success threshold, increasing the bounded learning budget to 64 updates. These
are fixture development observations, not independently preregistered findings.

Scientific attacks on the main experiment's policies, larger attacker budgets,
comparisons with fixed anchors, current-match control scenarios and any historical
opponent-population treatment remain open ETU-97 work. A failed bounded attacker
never certifies Nash equilibrium. ETU-85 demo acceptance is separate.
