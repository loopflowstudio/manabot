# 04: Verification Harness

Replace the rigid ladder with a reusable first-light research harness.
Keep the two cheap sanity checks, but make the main question about
learning against random explicit and repeatable.

## Finish line

- `python -m manabot.verify.step0_env_sanity` still works
- `python -m manabot.verify.step1_trivial_reward` still works
- `python -m manabot.verify.run_first_light` records a run in SQLite
- `python -m manabot.verify.report_first_light --run-id <id>` emits a
  markdown report with a recommendation up front

## What changed

### Kept

- `step0_env_sanity`
- `step1_trivial_reward`
- stochastic evaluation metrics in `manabot.verify.util`

### Replaced

- DuckDB metrics logging
- step-shaped verification as the primary interface
- `step4_beat_random` as the main decision surface

### Added

- SQLite store at `.runs/verify.sqlite`
- `run_first_light` harness
- `report_first_light` markdown report generator

## Why

The important first-light question is no longer "which step are we on?"
It is:

> Does the current shaped PPO recipe produce real learning against a
> random opponent on the causal-chain metrics that matter?

Those metrics are:

- play lands when available
- cast spells / play creatures when available
- reduce pass-collapse in pass-vs-land states
- improve win rate against random relative to an untrained baseline

## Command surface

```bash
python -m manabot.verify.run_first_light --mode dev --seed 42 --label shaped-v1
python -m manabot.verify.run_first_light train --mode decision --seed 7
python -m manabot.verify.run_first_light eval --opponent passive --eval-num-games 50
python -m manabot.verify.report_first_light --run-id 17
```

## Storage

All first-light runs go into one SQLite database:

```text
.runs/verify.sqlite
```

Schema includes:

- `runs`
- `run_configs`
- `evaluations`
- `evaluation_choice_sets`
- `evaluation_actions`
- `reports`

Detailed per-action rows are stored only for baseline and final evals by
default.

## Recommendation rule

The harness recommends moving on only when all of these are true in the
final random-opponent eval:

1. win rate improves meaningfully relative to baseline
2. land-vs-pass choice states tilt toward land instead of pass
3. spell-play metrics no longer suggest the causal chain is broken

Otherwise it recommends staying in first-light and names the weakest
signal.
