# First Light Experiment Harness — Validation

## Quick smoke test

```bash
# Sanity checks still pass
python -m manabot.verify.step0_env_sanity
python -m manabot.verify.step1_trivial_reward

# Short dev run (trains ~500k steps, records to SQLite)
python -m manabot.verify.run_first_light --mode dev --seed 42 --label smoke-test

# Generate report from the run
python -m manabot.verify.report_first_light --run-id 1

# Eval-only run (no training, just baseline eval)
python -m manabot.verify.run_first_light eval --eval-num-games 50
```

## Verify SQLite storage

```bash
sqlite3 .runs/verify.sqlite "SELECT id, label, status, mode FROM runs;"
sqlite3 .runs/verify.sqlite "SELECT run_id, kind, step, win_rate, landed_when_able FROM evaluations;"
```

## Tests

```bash
pytest tests/verify/test_first_light.py -v
pytest tests/verify/test_util.py -v
pytest tests/env/test_match.py -v
```
