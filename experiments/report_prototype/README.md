# Evaluation performance viewer

This directory retains the reviewed evaluation activity overview. Training
reports use Etude experiment files; the former training-data builder and
experiment definitions are removed.

```sh
uv run python -m experiments.report_prototype.build \
  --eval-receipt .runs/eval/panel --out .runs/eval/report
```

Repeat `--eval-receipt` to compare retained attempts. The builder executes no
games and copies linked evidence into the output directory. Open `report.html`.
Inference, Game engine, Evidence and replay, and Other and unmeasured are the
default activity groups; click for measured phases. Each resource keeps its own
clock, and incomplete coverage never becomes zero.

Etude 0.2.5's report panels read training metrics and evaluation score tables;
they do not yet accept this resource-specific evaluation performance contract.
The viewer remains until that integration preserves clocks, coverage, activity
drill-down, shared allocation accounting, and diagnostic evidence links.

Check: `./scripts/verify-eval-performance-report`.
