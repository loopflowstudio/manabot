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

ETU-99 owns uncontended independent-seed measurements as policies
change, supported CPU/MPS comparisons, cloning/sampling micro-costs, and
conservative complete-cohort projections. Those measurements require a separate
frozen protocol within the retained cap; rented hardware needs explicit funding.

## Capacity ladder software calibration

```bash
uv run -m experiments.runners.calibrate_training --capacity --out .runs/capacity-calibration
```

This freezes the existing `model_capacity.regimes` ladder: width64/depth1,
width64/depth2 and width128/depth2, all with four heads. `AgentSpec` remains the
only model configuration; `with_capacity` changes those fields without changing
pooling, information inputs or learning. The calibration holds historical mean
pooling and scalar output fixed on the semantic Allies/Lessons setup. It runs
one seed, two updates of 256 learner transitions per capacity, two raw exports
per capacity and the shared arena at both checkpoints. The protocol is a
workflow smoke, not a capacity study or strength comparison.

The total allowance is 900 seconds: the existing study has a 780-second deadline
and probes use only the remaining allowance. Each training run has 120 seconds,
with 60 seconds per stage and 30 seconds per arena game. No retry occurs; a fresh output directory retains the
resolved plan, canonical attempts, artifacts, traces and failures. The active
ETU-91 campaign is not opened or changed. Its scientific allocation is not
inherited. This software check does not authorize a later scientific run.

`calibration.json` includes the existing collection, learning, export, sampled
process-tree RSS, evaluation/replay and total elapsed costs. Its `inference`
rows add a probe of each final raw checkpoint: model construction time, ordinary
checkpoint loading time (including construction/admission), first forward on a
freshly loaded model, three unmeasured warmups, and ten timed batch-four forwards.
The reported steady time is the total for ten forwards; divide by 40 for seconds
per observation. Native collection supplies an identified real observation batch;
its separate probe cost does not add training samples. Each row binds the exact
checkpoint bytes, tensor-batch digest and architecture receipt. Probe seeds come
from the recorded run's collection stream. Construction/loading and first forward
are cold model operations in an already initialized process, not cold OS caches
or import startup. These costs overlap conceptually; do not add construction time
to loading time. Outer elapsed time includes all probe overhead.

Inference RSS is sampled before/after calls and training RSS at executor checks;
these are observed lower bounds on peak resident memory, not allocator peaks.
Host load snapshots disclose possible contention but cannot attribute it. A
concurrent run establishes software/accounting behavior only; use an independently
approved uncontended protocol before drawing scaling or laptop-throughput conclusions.

### Retained software proof (2026-10-05)

At source `90c55627`, `.runs/etu102-capacity-smoke` completed the documented
command in **212.56 seconds** on w4: three runs, 512 learner transitions per
run, six admitted raw checkpoints and 40 exact-replayed arena games. All three
inference probes completed. The resolved plan, run/source/world/setup identities,
checkpoint digests, collection seeds, architecture/component receipts and complete
arena attempts remain in that directory. No EMA, demo admission, strength or
human-challenger claim follows.

| Width / depth / heads | Trainable parameters | Collection seconds | Optimizer seconds | Observed training RSS (bytes) |
| --- | ---: | ---: | ---: | ---: |
| 64 / 1 / 4 | 138,434 | 3.20 | 0.62 | 585,646,080 |
| 64 / 2 / 4 | 188,418 | 4.02 | 0.96 | 623,181,824 |
| 128 / 2 / 4 | 712,706 | 5.66 | 1.68 | 974,766,080 |

The parameter counts are exact for this resolved configuration; time and RSS are
single-run observations. Host one-minute load was 50.46 before and 32.52 after;
contention was uncontrolled. These values cannot establish scaling efficiency.
Inference batches are identified separately per checkpoint, not guaranteed equal
across arms. Construction/load/first-call/warmed costs remain separate in
`calibration.json`. Arena time was 193.59 seconds including 16.82 seconds of
replay; it is not an additional training cost.

The gate also retained a failed default-calibration attempt under
`.runs/etu102-check-failures/calibration-timeout`: one game exceeded the original
10-second per-game allowance, while its partial command trace replayed exactly.
This is an archived copy of test artifacts; original absolute temporary paths in
its records are preserved. The software protocol now allows 30 seconds per game
without raising the total deadline. The affected check passed on rerun. Initial
collection failed because this fresh checkout lacked its native extension; the
local rebuild and a missing test-import repair preceded final verification.
The focused/gate checks passed with one optional notebook-dependency skip. The
single capacity attempt completed; no scientific capacity allocation was launched.
