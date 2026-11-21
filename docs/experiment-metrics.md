# Reading the experiment report

An experiment's page is drawn by `etude report` from its Etude runs
([writing a report](experiments.md#writing-a-report)). This guide defines the
metrics those pages and the readers in `manabot.reports` show, and what each
does not measure. Everything here reads Etude runs; the notebook dashboard
over saved regime records, its study and snapshot readers, and their
training-time profile are at the revision
[saved regimes](training.md#saved-regimes) names.

## Select a training cohort

Generate a self-contained training-time report from Etude runs, local or stored:

```sh
uv run --extra reports manabot training-report \
  --source ~/.etude/track \
  --source s3://your-bucket/your-etude-prefix \
  --out training-time.html
```

```text
training-time.html: 12 logical runs; 12 source/coverage warnings
```

The page provides search, sorting, run checkboxes, a largest-five selection,
logged active training time, descriptive per-run statistics, each run's
throughput, and a training-only timeline. It does not break a run's time into
phases or report rental cost: read phases from the `PERF` panels of
`etude report` ([costs](#costs)) and cost from the launcher's record. The logs
name no hardware, so throughput is not pooled across runs. Selecting a
recovered run includes its available attempts. Work counters come from the
retained trajectory, while duration includes failed/replayed attempts.
Incomplete or branching ancestry leaves ambiguous work counts unavailable.

The initial selection uses observed active training seconds and excludes explicitly
labeled benchmarks, smokes, fixtures, tests and evaluations. Unclassified records
remain visible and eligible; inspect them before calling the result a production
training cohort. Use `--top 10`, `--include-benchmarks`, or repeated `--run` with
the exact logical IDs shown in the page to change the initial selection. Page
filters do not silently remove already selected runs.

The local recipe writes `run/kind=smoke` and
`run/recipe=manabot.recipes.local` beside its update metrics, so new local smokes
are excluded by default even under a different experiment or run name. Explicit
Etude declaration kind/job fields take precedence over these logged labels.
Other recipes log no label, so their runs, like older ones, stay unclassified;
names alone do not establish purpose.

Use **Save selection** and **Restore selection** for exact membership tied to the
report's evidence snapshot. Restoring against different source bytes is rejected;
retain the original HTML and its `*-training-time.json` alongside the selection.
The URL also preserves selected identities for that snapshot.

`training_seconds` must be a declared cumulative **time** clock measuring active
training; `--training-clock NAME` selects another explicitly equivalent clock.
`learner_transitions`, `updates` and `steps` are distinct declared step clocks.
Resumed attempts need the active-time resume coordinate to establish their added
duration. Metric names are not interpreted as timing phases, and no rental
ceilings are converted into spend. A directory of saved regime records
is not a source: it holds no Etude run, and the report says so in its
warnings.

An attempt that ends before logging any clock coordinates leaves prior work
and observed active time visible. The report retains the attempt and warns that
its unmeasured time is excluded; it does not treat that time as zero. A run with
no observed attempts still has unavailable duration and work counters.

The reader uses the optional Etude SDK and existing credentials. Local/S3 reads
happen only during report generation; the resulting page needs no server or
network. Source failures remain visible, archives are not automatically expanded,
and no checkpoints are downloaded. A snapshot covers its configured sources,
not an assumed complete training history.

The accepted cross-project scope is recorded in
Training time reports from selected Etude runs (the plan was retired; the
cohort report in [training time](../experiments/reports/training-time.md) is the record).

## Learning

Progress is shown separately for each variant and independent training seed.
Updates are collect/update iterations, including empty-filter skips; planned
updates come from the job's target. An active-time recipe labels its update safety ceiling separately from
the active-hour target. Learner transitions, native decisions and
optimizer exposures count different work. SPS explicitly means cumulative
learner transitions divided by recorded training seconds at the latest diagnostic.
It includes measured training overhead, not just model inference. Host contention
is uncorrected. There is no uncertainty interval for these process measurements.

RL loss is the latest retained optimized minibatch objective, with policy/value
and regularization semantics defined by the recipe. It is not predictive log
loss, is not necessarily comparable across recipes, and can be absent on an
empty-filter update. Do not infer strength from the loss. Latest rows without a loss remain unavailable rather
than carrying a stale optimized value forward without a label.

To read a run's scalars yourself, each update's row holds them by name:

```python
from etude.train.read import read
from manabot.reports.update_metrics import updates_from_run

updates = updates_from_run(read(runs_dir, experiment, run_id))
[(update, row.get("collection_kl")) for update, row in updates.items()][:2]
```

```text
[(1, 0.0), (2, 0.0031)]
```

## Evaluation

Win fraction counts wins over all valid games; draws are not wins. The source
monitor also retains score (win + half draw) and draw fractions. Latest evaluation
is checkpoint status, not a ranking of variants. Each row identifies the original
checkpoint's updates/transitions, training-time lag, opponent and completed/planned
game counts. Incomplete or failed evaluation attempts cannot supply rates.

The saved 95% percentile interval resamples complete four-leg deals, retaining
both deck and seat assignments. It conditions on one checkpoint and does not
measure training-seed uncertainty. One deal can yield a zero-width interval,
including in the tiny demo; this does not establish precision. Monitoring deals
are inspected development evidence, not the scientific held-out endpoint cohort.

A `MonitorResult` (`monitor.json`) retains raw game rows, score/draw
intervals, protocol, artifact digest and opponent identity;
`python -m manabot.reports.training --evaluations` publishes saved results as
Etude curves ([training](training.md#checkpoint-evaluation)). Report paired independent-seed effects under the scientific
protocol before making a method claim; never pool games as independent fits.

## Comparisons

A comparison across runs is drawn only at coordinates every compared run
reached: Etude's report plots each job's checkpoints on the report's clock,
and `eval_jobs(state)` plays pairs at shared milestones. Protocol, opponent,
world and inference identity must agree. No interpolation, cherry-picked best
checkpoint or unequal-latest ranking is performed, and points stay per seed.

The time axis is original checkpoint cumulative recorded training seconds. The
work axis is cumulative native environment decisions, explicitly a work proxy,
not FLOPs or hardware-normalized compute. Equal updates need not mean equal
transitions, exposures, seconds or dollars. Curves show these differences; they
do not assert an equal-cost treatment effect. Sparse single-point fixtures do
not establish learning trends.

Equal-cost analysis needs overlapping observed support and a predeclared cutoff/selection
rule; do not extrapolate absent checkpoints. Keep recipe-specific configuration,
independent seeds and held-out scientific cohorts attached to any conclusion.

## Costs

Every self-play update logs host phase timers under `time/`, which Etude's
`PERF` panels (`manabot.training.panels`) draw. A key is the phase's path,
lowercased with underscores (`time/collection/observation_preparation`).
Collection holds observation preparation, policy inference and action
selection, rollout recording, opponent policy, environment step/encoding and
batch assembly. Learning holds batch transfer and behavior checks, targets,
selection and health diagnostics, and per minibatch forward and loss, backward
and gradient clipping, optimizer step and minibatch diagnostics, plus the EMA
update. A parent's seconds minus its children's are unattributed time, which
stays visible rather than being spread over the children. Seconds and call
counts are attempt-local.

PPO's `Minibatches/Preparation wait and transfer` measures the caller's wait
for the next host minibatch and its device upload. Host preparation overlaps the
preceding minibatch's training; its worker time is not added to the phase tree.
Earlier PPO timings included preparation inside `Forward and loss`, so compare
total learning time across that boundary. Uploads may complete during the forward.

Compute phases end at points where the code already waits for the device (a
`.cpu()` copy, a `float()` read, or a finite check), so a phase includes the
device work it queued and timing adds no CUDA synchronization. They are host
intervals, not GPU kernel timings or utilization, and PyTorch tracing stays off.
Learned state, diagnostics and collected batches are identical with timers on or
off (`tests/training/test_phase_timing.py`). A timer costs about 1 µs, and reading
cumulative stats about 0.3 ms per update: under 0.02% of a ten-second update.

Etude records each job's wall time and cost; a resumed run's attempts each
carry their own. Process seconds are occupied time, not CPU utilization or
hardware-normalized compute, and host dollars exist only where the launcher
recorded a price. Failures, interrupted attempts and pending work remain
visible; a missing cost record is not zero cost. Timing has no statistical
uncertainty estimate; repeat calibration for performance claims.

## Sampling

Diagnostics are plotted per job and metric, one line per training seed. A
smoothed line shows local variation; it is not a confidence interval or a
reconstruction of unrecorded minibatches.

For Ataraxos move updates, each learner timestep is an optimized minibatch across
selected environments. The loop overwrites `loss`, `policy_loss`, `value_loss`,
`entropy`, `collection_kl`, `reference_kl` and `reference_forward_kl` each time.
The saved values come from the **last optimized timestep minibatch**, evaluated
before its optimizer step. Positions and legal offers vary by minibatch, so the
values describe that minibatch alone.
Empty actor/critic filters skip optimization and retain no loss or entropy.

`policy_loss` is the clipped surrogate **plus** weighted collection/reference
reverse KL and, when `reference_forward_kl_coefficient` is nonzero, the weighted
forward-KL floor. The report uses the recorded policy loss because the surrogate
was not saved separately. The KL panels show unweighted nats. Scalar value loss
is MSE against lambda targets. Categorical value loss is cross-entropy against
outcome-distribution lambda targets. Neither is an independent terminal-outcome
test.

`rows` and `retained` cover the collection batch, unlike the last-minibatch losses.
The report totals these only where both are present, names the record count, and
separately counts explicit skips and records without a loss. Missing retention is
unavailable, never zero. Optimizer exposures can differ from selected rows because
recipes can reuse rows or select actor and critic samples differently.

Entropy is `−sum(p log p)` over legal offers, in nats. A uniform binary distribution
has entropy ln(2), about 0.6931, but that value alone cannot establish two legal
actions: larger skewed distributions or averages of different supports can have
the same entropy. Normalizing by log(number of legal actions) requires the actual
sampled support sizes, which these historical diagnostics did not retain. Entropy
measures dispersion, not competence; a confident wrong policy can have low entropy.
The saved history investigation that measured these counts was retired with the
September world; `git log` keeps it.

## Opponent asymmetry

Read a model's results by the deck **it** played, separately from its
opponent's. Each row uses `seat_decks[player_a_seat]`, never leg parity or seat zero
alone. Model-playing-Lessons means the opponent plays Allies. Greedy's creature
and attack choices on Allies may be stronger than its targeting choices on
Lessons: this is a hypothesis about the fixed opponent, not a causal conclusion
from scores. Aggregate scores can conceal this asymmetry. Random and greedy stay
in distinct panels with their immutable registration identity.

Deck intervals resample the protocol's complete deals, retaining both seat legs
for that deck and using the saved bootstrap seed/replicate count. Wins exclude
draws; raw wins/games and draw counts remain in the checkpoint table. Incomplete
cohorts supply no rate. Missing seat metadata supplies no deck estimate. Bands
measure deal noise conditional on one checkpoint; they do not measure independent
training-seed variability. No significance claim follows from inspecting a peak.

## Numerical health

Numerical diagnostics are read apart from playing strength. Gradient norms,
rounded legal zeros, skipped steps and measured parameter movement describe
optimizer behavior, not competence. Missing cadence samples stay unavailable.
See the [numerical health contract](https://github.com/loopflowstudio/manabot/blob/main/docs/numerical-health.md)
for units, sampling and failure investigation. A saturated all-win bootstrap may
also produce a zero-width interval; that is an empirical resampling limitation,
not certainty of winning future games.

## Self-play

A run's self-play outcomes, pooled into windows of updates:

```python
from etude.train.read import read
from manabot.reports.self_play import self_play_points
from manabot.reports.update_metrics import updates_from_run

updates = updates_from_run(read(runs_dir, experiment, run_id))
points = self_play_points(list(updates.values()), window_updates=100)
points[0]
```

```text
SelfPlayPoint(update=100, deck='7d663f0e…', opponent_deck='fc948195…', seat=0, wins=212, losses=187, draws=1)
```

`window_updates` defaults to 100 and must be a positive integer; use 1 to
inspect individual updates. Each point holds joint deck, opposing deck,
native seat (0 is on the play), win/loss/draw counts and the window's last
update.

Self-play deck and play/draw win rates describe the training distribution, not
playing strength. The panels plot completed outcomes in non-overlapping
training-update windows. Each game contributes one
observation on the play (seat 0, the engine's first player) and one on the draw
(seat 1). A drawn game counts in the denominator but not as a win. Rates weight
completed games, not updates or stream allocations.

Keep cross-deck and mirror matchups apart. A deck view pools both starting
seats; a play/draw view pools both decks within that matchup.
In a mirror match, pooled deck win rate is mechanically 50% without draws;
play/draw imbalance remains informative. Different training seeds never
share a curve. No uncertainty interval claims independent training seeds.

Keep the complete Etude metric run when copying evidence, including
`artifacts/`. Each update references an immutable Etude row table of
deck/opponent/seat counts; Etude resolves those references after selecting
resumed and duplicate updates. See [update metrics](training.md#update-metrics)
for storage and reader details. Copying only metric blocks leaves the counts
unavailable. A missing or damaged referenced table raises an error naming its
file: restore the original bytes from the run's retained source (or rerun
`uv run etude sync` after an interrupted delivery) and regenerate the report.
Do not edit the table or replace missing evidence with zero counts.

Runs without recorded outcomes yield no points. This instrumentation does not
change losses, actions or schedules.
