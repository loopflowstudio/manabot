# Experiment report prototype

A working design prototype of the default experiment report (MTG-180). It is a
reference for how reports should look and behave while the production version
is built as `etude report EXPERIMENT`. It is not that version: data is assembled
by a one-off builder, and the page is plain HTML with no Etude integration.

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/seat-comparison-v1
uv run python -m experiments.report_prototype.build --out .runs/report-prototype
```

Open `.runs/report-prototype/report.html`, or serve that folder. The experiment
picker switches between:

| Experiment | Data |
|---|---|
| Seat comparison v1 | Real: the from-scratch one-seat vs two-seat study, from the artifact store. Default. |
| MTG-165 two-seat ladder | Real, but only where its retained snapshot exists locally; otherwise skipped. Not a matched comparison. |
| Elemental, ten decks | Synthetic, shaped like the MTG-181 design (10×10 grid vs heuristic, 50 games per cell). |

## What the prototype settles

- **Defined in Python with the experiment.** `definitions.py` declares a
  `Report` per experiment: eval panels (`Curve`, `Matrix`, `Ranking`,
  `HeadToHead`), an optional default clock and baseline. Perf and ML details
  default to panels over the trainer's own metric keys (`Series`, `Flame`,
  `Counters`), so most experiments only declare evals. No manifests or config files.
- **Jobs, not runs.** Every chart shows every job; there is no primary run.
  A job is labeled by its launch name plus seed count ("Uniform decks · 2 seeds")
  and shows the seed average: curves band the seed range, matrices pool games.
  Job-vs-job numbers use a point every job evaluated.
- **Labels come from the launch.** The name given when a job is created
  (Etude `training(..., name=...)` → `RunDeclaration.job_name`, or a manabot
  `Case(label=...)`), never from the report.
- **Baseline job, dotted.** The experiment's declared baseline is drawn dotted
  everywhere; `Report(baseline=...)` overrides it.
- **Two top-level parts.** Evals, then Perf and ML details. Each section owns
  its controls (x axis on updates, learner decisions or training hours; opponent;
  smoothing); only job selection and filters are global.
- **Name who plays what.** Matrices put the model's deck across and the
  opponent's deck down; labels read "model Allies vs heuristic Lessons".
  The matrix defaults to the latest checkpoint.
- **Where time goes, as a flame graph.** An icicle per job of recorded timers
  (`collection_timings`, `learning_timings` and stage totals), with derived
  remainders and untimed work hatched, comparable per million learner decisions.
- **Honest gaps.** Missing evaluations are gaps, never interpolated; metrics a
  run did not retain are listed as such; synthetic and illustrative data are labeled.
- **One meaning per color.** Job colors are only for jobs; stage colors only for
  time stages; green/red only for winning/losing and passing/failing checks.

## Files

- `definitions.py`: the report vocabulary and the three report definitions.
- `build.py`: reads the sources above and writes the page folder.
- `report.html`, `model.js`, `charts.js`: the page, its query layer and charts.

Arena receipts also feed this same Perf surface:

```sh
uv run python -m experiments.report_prototype.build \
  --eval-receipt .runs/eval/panel --out .runs/eval/report
```

Repeat `--eval-receipt` for job/seed comparison. This reads real receipts without
fetching training studies, running games or manufacturing timing subdivisions.
Coverage-aware seed aggregation omits unavailable children from the partition
and labels their measured population. Evaluation adds shared-allocation
accounting, request/batch timelines and diagnostic trace links. See
[shared evaluation](../../docs/shared-evaluation.md) for receipt and calibration
contracts. This remains the executable report prototype, not a claim of
production Etude report integration.
Cell and inference-allocation timers exclude the local CLI's earlier checkpoint
preparation. Their throughput and unavailable cost fields do not establish
whole-job cost or end-to-end speedups; see the linked measurement boundaries.
