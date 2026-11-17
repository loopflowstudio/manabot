# MTG-329: adopt Etude execution and agents

Design: design-train-api.md. Etude pin d9a6bc8f06b20948601d540be80e285d46010465;
Pydantic required. Jack Heart, 2026-11-16: "the whole point is the adoption";
"These should be done/mainatined in this task". No paid/launched run or landing
with this pin. Publication and gate remain the caller's steps.

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
AgentError/PanicException is invalid, score null, with remaining games played.
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
proof. origin/main equals the merge base (c035d5566a); no upstream sync needed.

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

## Remaining

- Gate and publication are the caller's steps. The full nightly baseline
  (`.github/workflows/baseline.yml`, workflow_dispatch: `etude train` then
  `evaluate`) has not run since the corpus fix; dispatch it on the PR branch
  as the gate's proof of that path.
