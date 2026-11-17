# MTG-329: adopt Etude execution and agents

Design: design-train-api.md. Etude SDK 0.4.1 from PyPI; Pydantic required. Jack Heart, 2026-11-16: "the whole point is the adoption";
"These should be done/mainatined in this task". The 2026-11-17 steers approve
the speed and learning measurements on one reasonable single GPU, total
`--max-usd 10`. The unreleased git pin has been replaced; publication and gate
remain the caller's steps.
Upstream (rechecked 2026-11-17 17:14 UTC): PyPI's latest etude-sdk is 0.4.1 and Etude
main has nothing after its release commit cd4ab8c5e9, which differs from d9
only in release metadata, so no Lium timestamp fix exists. Manabot main is
still 3afd92d8f3, the merge base.

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
and its stage reader, reports/selection.py,
training/clock.py and its test, report_baseline, local_checkpoint_file,
HardwareInventory, replay_prefix, host_facts and its CPU/cgroup/RunPod readers
and exclusive tests.
Third cut (PR #500 steer, built 2026-11-17 in compress): the saved-regime
schema and every reader of it. Gone: training/regime.py, records.py,
validation.py; experiment/records.py (StepCalibration, ExperimentRun) and
checkpoint_evaluation.py; reports/analysis, experiment, learning, notebook,
records, snapshot, study, time_profile and regime_study.ipynb; infra/host.py
(HostFacts/MemoryLimit: their one reader was RegimeRunRecord); ladder_precision
and `precision="auto"`; read_updates, stage_rows, self_play_figures,
training_dashboard, native_cohort; tests/fixtures/regime_builders.py,
saved_regimes.py and experiment-demo/, and the tests only for these.
Report cleanup complete: deleted the phase/flamegraph views, synthetic profile
hierarchy and unused CohortRun timing/cost fields. Removed the notebook extra,
nbformat and matplotlib requirements, with workflow/install commands updated.
Compress 2026-11-17: `CohortRun.hardware`/`model` (always "unknown") and the
cross-run rate pooling they keyed are gone, so throughput is one row per run;
the per-run "phase attribution unavailable" warning is gone; `lineage_roots`
lives with its one caller in time_sources. The release smoke no longer imports
`nbclient`, which left with the notebook extra (it would have failed the
release workflow). `pytest tests/training/test_time_cohort.py -q`: 8 passed.

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
injection, scored provider runs, browser proof, full suite, automatic
policy→sampler scheduling. LLM experiments stay blocked on EXP-70/frozen
checkpoints; their declarations share Agents but are not a runnable adoption
proof. The branch's merge base is origin/main 3afd92d8f3 (rechecked at the
second 2026-11-17 realign; no newer main commits); the CPU learning comparison used that main.

QA retry fix: each Elemental callback invocation writes its own
`arena/<match-key>/attempt-*` (`elemental_play.evidence_directory`), so failed
and interrupted attempts keep their journals while a retry completes.
`uv run --extra dev --extra artifacts pytest tests/eval/test_elemental_play.py -q`:
6 passed (2.81s) after compress, local and launched-style paths.

## Concept follow-up

The five directions from design-train-api.md (97338108bc review) are built:
TrainingRun names the live handle throughout docs; SelectionGameSpec lives in
rollout/selection_data.py; Collector.operation is public. The saved-regime
readers that review kept apart are deleted (Delete list, third cut), and with
them the fresh-interpreter boundary test that refused their import.
The evaluation guide uses the local recipe's actual final EMA policy path.
Serialized keys and frozen records are unchanged. Commands and evidence:
implementation.md.

Fifth concept review (928ee3f4bc): the QA fixes settle as built — an invalid
game row means a native failure only, and an all-invalid Elemental group is
Etude's failed match. Nothing further to change; no new open choice.

Sixth concept review (1af59dc55e), the first since the regime-reader deletion:
the experience is one path (`etude train FILE`, exit 75, `--resume`,
`etude continue`, `etude report`) and main holds no second way to run or read
a run. Nothing to change. The two write-docs defaults stand as reversible and
do not block: tla_frontier stays as the reader of its executed runs' report,
and `manabot training-report` stays as the cross-experiment run list
`etude report` does not draw. Naming leftovers are ideas in questions.md, not
implement directions.

Seventh concept review (c22355ca9c), covering the QA report fixes
(5d57401099, 3850764dd8, eef697a757). The third clock and the silent-attempt
rule settle as built. The report labels do not, and are the next implement
direction:

- What a run is for is the run's identity, which the design gives to the
  command ("Identity, inputs, continuation and paths belong to the command").
  Etude's `RunRecord` already has `kind` and `job`; 0.4.1's `track` and
  `etude train FILE` cannot set them, so the labels travel as text metrics.
  That stopgap is acceptable only where it is asked for.
- As built, `self_play(..., kind="training", recipe="")` makes every caller
  assert a purpose it never stated. Counterexample:
  `experiments/batched_baseline.py`, a benchmark, calls the loop with no
  label, so its rows say `run/kind=training` for good and the report includes
  it by default; Elemental's retained rows gain `run/kind=training` and an
  empty `run/recipe` on every update. docs/experiment-metrics.md says "names
  alone do not establish purpose"; a default does no better.
- Direction: the loop logs only labels its recipe gave. A recipe that says
  nothing writes no `run/kind` or `run/recipe` column and stays
  `unclassified`, as before 3850764dd8. Only `manabot/recipes/local.py`
  passes labels, so the stopgap lives in throwaway smoke runs and never in
  retained evidence. Test: a run through `baseline`/`elemental` settings has
  no label columns; the local recipe is still excluded by default.
- docs/training.md shows `self_play` as "the loop you would write" with the
  old signature and no label line. Make the listing match the code, and say
  in one sentence that labels are optional and what the report does without
  them.
- Proposed, not filed (Etude): `etude train FILE` records the file as the
  run's `job`, and the command or `track` accepts a `kind`. When a release
  has that, delete the label argument, the two text metrics and
  `time_sources.label`; the reader's `declared.kind` / `run.job` precedence
  already prefers the record.

Built after the seventh review: both loop labels default to `None` and only
supplied labels enter the update row. Local keeps its explicit smoke labels.
The training guide's example matches the signature and conditional logging.
`uv run --extra dev --extra artifacts pytest tests/training/test_time_cohort.py -q -k 'baseline_recipe_without_labels or local_recipe_is_a_named_smoke'`:
2 passed (7.11s); real baseline rows omit both columns and report as
unclassified, while local smoke remains excluded by default. GPU speed and
resume measurements remain unrun.

No open product choice: the smoke rows are throwaway and the Etude change
is additive.

Eighth concept review (089a6f80e1), covering 54b67f6cd9: the seventh
direction is built as written. A recipe states a purpose or the run has none;
`time_sources` reads record, then label, then `unclassified`. Nothing to
change in code and no open choice. One sentence in docs/experiment-metrics.md
said only "older" unlabeled runs are unclassified; corrected in this review
(new baseline/Elemental runs are too). The Etude `job`/`kind` proposal stays
unfiled and does not hold the PR.

## Report follow-up

QA's two retained-report defects (scratch/qa.html) are fixed by 3850764dd8,
  committed after that QA pass: new local smokes carry explicit
  kind/recipe text labels and are excluded by default. Empty attempts retain
  earlier observed work/time and remain counted, with a coverage warning for
  unmeasured time. Unlabeled historical runs stay unclassified. No config,
  checkpoint or frozen evidence was rewritten. Focused report, recipe and
  real resume checks: 14 passed (9.08s); evidence in implementation.md.
  Compress: the label reader takes the newest block carrying the text column
  and the AGENTS note sits under Etude training records; behavior unchanged.
  `pytest tests/training/test_time_cohort.py tests/training/test_self_play_loop.py -q`: 22 passed.

## Remaining

- Learning check, main against the branch (steer 2026-11-17,
  .lf/prompts/20261117-065538-msrXVy-steer.md; quotes in positioning.md).
  Exploratory: question, method, confounds and cost cap written before scored
  runs; numbers and limits to implementation.md and the PR.
  - Step 1 complete on CPU: three-update fresh local baseline, seed 197,
    main 3afd92d8f3 versus branch 9421c945d9. All 117 raw and 117 EMA tensors
    match bitwise; the 117 initial raw tensors also match. Six games, 1,421
    decisions and 268 optimizer updates on both sides. Evidence: implementation.md.
  - Step 2 is unnecessary for this uninterrupted case under the approved
    method: Step 1 found no difference.
  - Step 3 on the GPU: one branch run stopped and resumed against one
    uninterrupted run to the same update count; strength and curves, not
    tensors.
- Speed against main on the same GPU in the same session: Elemental training
  updates/s, collection and learning seconds per update, cost of `run.save()`;
  evaluation: show Manabot's players batch on CUDA, then games/s and
  decisions/s on the same checkpoints and paired deals (cite Etude's own
  evaluation measurements otherwise). More than a few percent slower: say
  where the time goes, do not optimize unasked. Any reasonable single GPU (L4,
  4090 or similar price; .lf/prompts/20261117-065639-rst6T1-steer.md,
  replacing the L4-only steer); the report names it. Use RunPod or Vast for the next retry; Lium is excluded. Terminate every machine and confirm.
  Harness: experiments/etude_execution.py; method, machine ids, events and
  cost estimates: experiments/reports/etude-execution.md. Nothing measured yet:
  a Vast 4090 went offline while compiling, Lium's pod was unreadable by Etude
  (timezone-naive `created_at`), and at 16:09 UTC on released 0.4.1 all eight
  RunPod/Vast templates had no capacity. Every machine is confirmed gone;
  spent ~USD 0.13.
  - Next: retry when RunPod or Vast offers a 4090/L4-class machine, from the
    fresh isolated `ETUDE_HOME=/tmp/mtg329-runpod-vast-etude` (the unresolved
    Lium intent in `/tmp/mtg329-performance-etude` stays as evidence and must
    not hold the rental); cap the rental at USD 9. Do not claim GPU acceptance
    until a scored run completes.
  - Local preflight passed: two-update CPU stop/resume, 16 replayed games.
    Direct SSH proves no queued launcher delivery. The harness is unchanged
    since ee370d9757 apart from compress edits, so a repeat may archive HEAD;
    /tmp/mtg329-gpu is temporary and may be rebuilt.
- Gate and publication are the caller's steps. The full nightly baseline
  (`.github/workflows/baseline.yml`, workflow_dispatch: `etude train` then
  `evaluate`) has not run since the corpus fix; dispatch it on the PR branch
  as the gate's proof of that path. PR #500 (open, not draft) head is
  eef697a757, this branch's HEAD, at the 17:14 UTC realign.
- Released dependency cut complete (pyproject, lock, release workflow,
  metadata validation, docs, AGENTS.md); evidence in implementation.md. Etude
  Task 2d1933 hard deadlines and the Lium fix remain external limitations;
  no Task for the Lium fix was found (questions.md).

## Steers 54c7d853 and 7d745eb2 (2026-11-17) — three things owed

Sources: .lf/prompts/20261117-105529-ssL53E-steer.md, -TkHneT-steer.md. "Do
not land on the commit pin" (already replaced by PyPI 0.4.1). Docs for these
are written: experiments/reports/etude-execution.md (retry rule at the top;
results go there) and docs/training.md "Saved regimes" (unchanged, already
states what cannot be rebuilt).

1. GPU measurements as in the Remaining entry above: RunPod or Vast, any
   reasonable single GPU, "try what is on offer", retry a few times across
   the run when a host dies or nothing is available, then say exactly what
   was tried. Lium excluded (Etude Task 73e1a9). About USD 0.13 of USD 10
   spent. Terminate and confirm every machine.
2. CI red at ce7e15ca27: Integration (ubuntu-latest, evaluation), run
   44136662771, 1 failed / 197 passed:
   `tests/eval/test_experiment_files.py::test_etude_finds_each_current_experiment_and_nothing_else`
   (line 77, set mismatch). Unverified reading: `experiments/etude_execution.py`
   is found as an experiment and is not in the test's expected set.
3. PR #500 body is wrong in two sentences. "Historical records remain
   readable, and frozen experiments retain their source revisions." — the
   regime readers are gone; say that reports built from regime records
   (any naming a regime, a `RegimeRun` or an `experiments/data/<name>/`
   SQLite store) cannot be rebuilt from main and are read from
   f4e1822faf46bbc9cdc43909293f8cbc8de4a300, with state.pt folders from
   f577362135562e569033fe33e48c7ef0411a829d. "the planned comparison could
   not start because no L4 was available" — replace with what the GPU retry
   actually showed. Not edited in write-docs: the second sentence depends on
   item 1.

## Built since the 2026-11-17 steers

- Policy-error test: RuntimeError propagates without a result row; native
  AgentError retains a crash row naming the responsible player
  (`pytest tests/eval/test_evaluation_match.py -q -k policy_error`: 2 passed).
  docs/evaluation.md ("Failure") already matched.
- Regime-reader deletion (PR #500 steer, .lf/prompts/20261117-065016-7oIWNJ-steer.md).
  Jack Heart: "i also would have expected this pr to involve more deletion of
  the executor and training regime code, no?"; offered keep / thin reader /
  delete, he answered "3". Built as the Delete list's third cut. Survivors:
  - `experiments/tla_frontier.py` states its settings in code and reads only
    the 134-deck `deck_source` of
    `experiments/regimes/tla-frontier-c5fba4-w64.json` as `UniformDecks`.
    Both sizes' `settings()` dumps equal the record-derived ones; `Protocol`
    is unchanged, so the environment digest is.
  - `manabot.experiment`: `setup`, `numerics`. `manabot.training.report`.
  - `manabot.reports`: `update_metrics.updates_from_run`,
    `self_play.self_play_points(rows, window_updates)`, `training`
    (`--evaluations` only), `time_sources`/`time_cohort` (Etude runs only),
    `belief`, `inference`, `evaluation`.
  - SelfPlay's refusals ported as `test_settings_refuse_what_the_loop_cannot_run`.
  - The PR must say which reports can no longer be rebuilt from main
    (docs/training.md "Saved regimes" lists them and names f4e1822faf).
