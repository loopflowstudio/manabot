# MTG-329: adopt Etude execution and agents

Design: design-train-api.md. Etude pin d9a6bc8f06b20948601d540be80e285d46010465;
Pydantic required. Jack Heart, 2026-11-16: "the whole point is the adoption";
"These should be done/mainatined in this task". The 2026-11-17 steer approves one L4 comparison under `--max-usd 3`;
no landing with this pin. Publication and gate remain the caller's steps.

## Delete — do not maintain

Removed training/execution, recovery/admission/learning_state, job/runner/study,
queue/monitor worker, TrainingRecipe/RunContext, Tracker/LoopState, local_launch
and exclusive tests. Removed sim/serving, serving_protocol, eval/endpoint,
execution, serving_benchmark, live allocations, allocation census/thread/latency
machinery and exclusive tests. No per-group child, remote fetch or own inference
service in managed evaluation. Keep scalar standalone inspection, frozen compute
readers, replay, Commands, admission, paired deals, cards and frozen identities.
Old execution revision f4e1822faf46bbc9cdc43909293f8cbc8de4a300; state.pt revision
f577362135562e569033fe33e48c7ef0411a829d. No old-folder loader.
Second cut (orphans of the retired executor): TrainingState/TrainingRuntime
recovery members and their Torch functions (policy/optimizer/RNG state, Adam
admission, clone/validate, device metrics), collector restore_rng, VerifyStore
(stage reader now in reports/snapshot.py), reports/selection.py,
training/clock.py and its test, report_baseline, local_checkpoint_file,
HardwareInventory, replay_prefix, host_facts and its CPU/cgroup/RunPod readers
and exclusive tests. Kept: StepCalibration (retained reader) and HostFacts /
MemoryLimit schemas for historical receipts.

## Built

Etude owns run execution/save/restore and logged ML schedules. All EvalJobs use
the same Agents(load/build). Loader owns Torch/device and checkpoint.path;
forward merges admitted Prepared rows into ordered masked logits. Torch-free
builders own host preparation and seeded NumPy sampling. Managed callbacks use
these in-process, one group per match and fresh agents-v1 tournament names.
Released player/match/compute readers and frozen evidence remain unchanged.
One owner each: ETUDE_COMPUTE_CLASS/ETUDE_SAMPLER_SPEC (sim/registration.py),
benchmark.play_deal (monitor and duel callbacks), supervision.verify_game and
is_game_error (in-process and child replay).

Cooperative seconds/Command caps and durable pending journals stay. Native
AgentError/PanicException is invalid, score null, with remaining games played
(`supervision.is_game_error`, 05bdc978da; QA's broad-`Exception` finding
predates it). Other Python errors propagate from managed in-process play/replay;
the standalone child replay (`replay_retained`, `eval/match.py`) records them
as `replay_passed: false` plus a replay error, never a pass. Elemental rejects a group
with no valid replayed games after retaining its evidence, so an all-invalid
smoke cannot pass. Removed the now-redundant RetryRequested exception override.
Etude Task 2d1933 owns hard per-match deadlines/dead processes; until it lands,
true hangs/exits retry the grouped job. No nested-process workaround.

## Validation and limits

Evidence and commands: implementation.md. Focused native, card, world, monitor,
recipe, journal and boundary checks pass at 852c0d866b; isolated dependency
integration and installed 3.12/3.13 smoke pass. At 57398243f6 the choice-corpus
KeyError (permanent/stack alias of a focused card) is fixed in
`manabot/verify/choice_corpus.py`, and the report test that shares
`report_recipe` with the nightly baseline passes (choice corpus, learning
regression, host facts: 9 passed). Not shown: CUDA batching, real Rust panic or native hang
injection, launched/provider runs, browser proof, full suite, automatic
policy→sampler scheduling. LLM experiments stay blocked on EXP-70/frozen
checkpoints; their declarations share Agents but are not a runnable adoption
proof. The earlier validation used merge base c035d5566a; origin/main now names
3afd92d8f3. No new comparison or sync was performed.

QA retry fix: each Elemental callback invocation writes its own
`arena/<match-key>/attempt-*` (`elemental_play.evidence_directory`), so failed
and interrupted attempts keep their journals while a retry completes.
`uv run --extra dev --extra artifacts pytest tests/eval/test_elemental_play.py -q`:
6 passed (2.81s) after compress, local and launched-style paths.

## Concept follow-up

The five directions from design-train-api.md (97338108bc review) are built:
TrainingRun names the live handle throughout docs; retained readers describe
saved regimes; report update readers live in reports/update_metrics.py;
SelectionGameSpec lives in rollout/selection_data.py; Collector.operation is
public. Live loop, Torch collection/runtime and duel/monitor import without
training.regime or training.records (fresh-interpreter boundary check).
The evaluation guide uses the local recipe's actual final EMA policy path.
Serialized keys and frozen records are unchanged. Focused import/outcome/timing
and resume/recipe/retained-reader checks: 25 passed; local recipe completed
two updates. Commands and evidence: implementation.md.

Fifth concept review (928ee3f4bc): the QA fixes settle as built — an invalid
game row means a native failure only, and an all-invalid Elemental group is
Etude's failed match. Nothing further to change; no new open choice.

## Remaining

QA fixes are built: evaluation error handling (24 passed, 37.31s) and snapshot
retry (`reports/snapshot.py` opens the source before creating the output; 1
passed). Commands and limits: implementation.md.

- Steer 2026-11-17 (complete text: `lf session steers`; quotes in
  positioning.md). No shipped doc changes for it: docs/evaluation.md
  ("Failure") already says native `AgentError`/`PanicException` give invalid
  score-null rows and "Other Python exceptions fail the match".
  - Fixed the stale red-CI test: RuntimeError propagates without a result row;
    native AgentError retains a crash row naming the responsible player.
    Compress split it into two tests; `pytest tests/eval/test_evaluation_match.py
    -q -k policy_error`: 2 passed (0.60s); Ruff check/format pass.
  - Speed against main on one rented L4, `--max-usd 3`, both sides in one
    session: Elemental training updates/s with collection and learning
    seconds per update and the cost of `run.save()`; evaluation games/s and
    decisions/s on the same checkpoints and paired deals, first saying whether
    CUDA batching worked. Numbers, method and limits go to implementation.md
    and the PR description; terminate the machine and confirm. More than a few
    percent slower: say where the time goes, do not optimize unasked.
    Blocked 2026-11-17: RunPod quotes $0.69/hour but no available L4; Vast
    and Lium also report none. No machine rented, no spend, no CUDA or speed
    numbers. Shared launcher DB unchanged; isolated launcher has no rentals.
    Rechecked 14:50 UTC through the isolated launcher: still none available.
    Retry when L4 capacity returns, keeping the approved USD 3 cap.
    Steers 2026-11-17 (.lf/prompts/20261117-065538-msrXVy-steer.md,
    -cZtiPA-steer.md, 20261117-065639-rst6T1-steer.md; quotes in
    positioning.md): any reasonable single GPU (L4, 4090 or similar price)
    may stand in, both sides on one machine in one session, the report names
    the GPU; Lium rental is untested, so confirm start and termination. Total
    cap `--max-usd 10` including this speed run.
  - Learning check, main against the branch (exploratory: question, method,
    confounds and cost cap written before scored runs; numbers and limits to
    implementation.md and the PR). Step 1 on CPU, no rental: fresh run, same
    recipe, config and seed, uninterrupted; compare every raw and EMA policy
    tensor after a few updates and report equal / summation-order only /
    different. Step 2 only on a real difference: Elemental, several seeds a
    side, curves and paired-deal strength including head-to-head at matched
    updates; if it cannot fit the cap, stop and say the cost. Step 3 on the
    rented GPU: one branch run stopped and resumed against one uninterrupted
    run to the same update count; compare strength and curves, not tensors.
    Evaluation speed: cite Etude's own measurements; measure only that
    Manabot's players run on CUDA and games/s against main.
- Steer 2026-11-17 on PR #500 (complete text:
  .lf/prompts/20261117-065016-7oIWNJ-steer.md). Jack Heart: "i also would
  have expected this pr to involve more deletion of the executor and training
  regime code, no?"; offered keep / thin reader / delete, he answered "3".
  Not started; comes after the L4 measurement, which stays pending, not abandoned.
  - Delete the retained regime/stage records and code that exists only to read
    them: `manabot/training/regime.py` (928 lines: TrainingRegime,
    TrainSelfPlay, other Stages, Execution, RecoveryPolicy, CheckpointCadence,
    import/origin records), reader-only parts of `manabot/experiment/` (1,367
    lines), `manabot/reports/`, `manabot/training/records.py` (247),
    `report.py` (72), `validation.py` (60), and tests only for them (~45 test
    files import these modules; tests/fixtures/saved_regimes.py and
    regime_builders.py among them).
  - Check every name for live users first; move what live code needs to where
    it is used and say what was kept and why. Known live users at 89fbdafed8:
    `ladder_precision` (infra/config.py, backends/torch/learner.py);
    `manabot.experiment.numerics` (recipes/elemental.py);
    `manabot.experiment.setup.machine_setup` (experiments/elemental.py,
    seat_comparison.py, batched_baseline.py, tla_frontier.py);
    `training.records.StageRecord` (reports/update_metrics.py) and
    `PhaseTiming` plus `experiment.records.ExperimentRun`
    (reports/time_profile.py, time_cohort.py).
  - Stop-and-say condition: `experiments/tla_frontier.py` still imports
    TrainingRegime/TrainSelfPlay and training.report as its settings holder
    for new launches (its executed runs are historical, report
    tla-frontier-c5fba4-results.md). Settle whether it is current before
    deleting that part. The nightly path (recipes/baseline.py,
    verify/learning_regression.py) imports none of these modules;
    `regime_digest` there is a serialized field name, kept.
  - Docs for this deletion are written (intended behavior, code not yet
    changed): docs/training.md (Saved regimes, reduced precision, update
    metrics), docs/experiment-metrics.md (rewritten around Etude runs),
    docs/experiments.md (retained comparisons, report example),
    docs/evaluation.md (attack plans, ownership), docs/observation-and-model.md
    (SelfPlay examples replace regime ones), docs/workflows.md,
    docs/development.md, manabot/README.md, manabot/model/training.md,
    manabot/model/agent.md, experiments/README.md. They name
    f4e1822faf (on main; executor, schema and readers) as the one revision
    that reads and runs the records. What they commit the code to, each my
    choice (questions.md): `manabot.experiment` keeps only `setup` and
    `numerics`; `manabot.training.report` stays and `job_labels` takes each
    job's `ModelConfig`; tla_frontier stays, its settings written in its own
    file with the regime JSON unread; `manabot.reports` keeps
    `update_metrics.updates_from_run`, `self_play.self_play_points` over one
    run's rows, `training --evaluations`, the Etude-only `training-report`,
    `belief`, `inference` and `evaluation`; StepCalibration, AttackPlan,
    checkpoint_evaluation, snapshot, notebook, study, learning, analysis,
    experiment dashboard and time profile go; no `precision="auto"`.
  - Name the last revision that reads the records in the guide beside the
    state.pt revision; the PR says plainly which reports can no longer be
    rebuilt from main. Frozen evidence, experiment ids, reports under
    experiments/, serialized keys and checkpoint fields are untouched.
    Report the line counts removed.
- Gate and publication are the caller's steps. The full nightly baseline
  (`.github/workflows/baseline.yml`, workflow_dispatch: `etude train` then
  `evaluate`) has not run since the corpus fix; dispatch it on the PR branch
  as the gate's proof of that path. PR #500 CI was running at 89fbdafed8
  (the policy-error split) when last checked; its result is not yet known.
