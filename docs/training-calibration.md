# Complete-loop training calibration

Run the fixed CPU workflow in a new output directory:

```bash
uv run -m experiments.runners.calibrate_training --out .runs/full-loop-calibration
```

This is a bounded workflow measurement, not a scientific throughput campaign.
It freezes a separate `workflow-smoke` protocol before execution: seed 197,
two self-play updates of 256 learner transitions each, two ordinary raw model
exports, and eight arena games against random over both deck/seat assignments.
Each game is replayed exactly. Training has a 60-second run allowance and
30-second stage allowances; the existing study process deadline is 180 seconds.
Notebook rendering is skipped. Neither the ETU-91 protocol nor its live checkout
is changed. All calibration attempts count toward the existing 168-hour cap;
this command does not allocate a new scientific budget or consult the live
campaign's accounting. Include its costs when reconciling that campaign budget.

`training.sqlite`, `study.json`, `resolved-plan.json`, run exports, checkpoints
and arena traces remain the existing authorities. `calibration.json` is a typed
derived summary, including failed attempts from the canonical store. Existing
output directories are rejected; no automatic retry, seed selection or failed
artifact replacement occurs. A caught failure retains its costs and raises a
nonzero exit. Abrupt process death can leave only the study/run records; do not
interpret a missing final summary as zero cost or successful completion.
[Training recovery](training-recovery.md) remains a separate opt-in contract.

Read each stage's counters separately:

- `environment_decisions`: active native vector microsteps, excluding forced
  internal engine ticks; not just the learner's decisions.
- `learner_transitions`: samples supplied to the learning rule.
- `optimizer_exposures`: sample uses across minibatches/epochs, not optimizer
  update count. Filtering and repetition can change this independently.
- `games`: completed collector games; distinct from `complete_replayed_games`
  in evaluation.
- `collection_seconds`, `learning_seconds`, `export_seconds`: phase costs;
  setup, persistence and other overhead also contribute to run/outer time.
- `cumulative_seconds`: the observed training cost when that checkpoint became
  available. Never backfill a cheaper cutoff with a future checkpoint. Existing
  study analysis reports no equal-cost comparison without observed overlap.

Self-play stages record actual model device and Torch intra-op thread count.
Arena inference uses its existing CPU/one-thread registration. `--device mps`
and CUDA are rejected before creating output; an inference microbenchmark does
not establish support or performance for this entire path. GPU transfer time,
cloning and hidden-world sampling costs remain unmeasured, not zero.

Arena cost includes worker startup, play, trace export and exact replay.
`replay_seconds` is a measured subset of `evaluation_including_replay_seconds`;
never add them. Failed cells lacking a replay receipt make replay time unknown
(`null`). Parent process CPU time excludes arena subprocess CPU time. Training
RSS is sampled across the process tree; host snapshots contain parent RSS and
available memory. Neither is an exact peak. Load averages give context without
attributing interference. Thermal throttling is not instrumented.

Phase clocks use `perf_counter`, which may exclude sleep; outer
`sleep_inclusive_seconds` uses the recovery continuous clock. This does not
replace the study runner's signal deadline or reinterpret frozen scientific
schedules. Concurrent runs, sleeping and contention invalidate hardware speed
comparisons. The real-game test establishes execution, accounting, failure
retention and replay, not strength or calibrated performance.

ETU-98 remains open for uncontended independent-seed measurements as policies
change, supported CPU/MPS comparisons, cloning/sampling micro-costs, and
conservative complete-cohort projections. Those measurements require a separate
frozen protocol within the retained cap; rented hardware needs explicit funding.
