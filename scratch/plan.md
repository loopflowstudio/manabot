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
HardwareInventory, replay_prefix. Kept: StepCalibration (documented retained
reader), infra/host.py (see questions.md).

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

Focused native, card, world, monitor and report checks, isolated dependency
integration, and installed 3.12/3.13 smoke pass; see implementation.md. Final
boundary/declaration verification (38 passed), lint and formatting pass. No CUDA/provider
proof or automatic policy→sampler scheduling claim. LLM experiments remain explicitly blocked on EXP-70/frozen checkpoints;
their declarations share Agents but their sessions are not a runnable adoption
proof. Earlier transcripts: 4de3a76ba0 and 382c653ec8:scratch/.

After main sync (3cb2aacbb5): rebuilt native cp313 extension; 233 distinct focused
collection/journal/disclosure/recipe and adjacent checks pass after fixing journal
read ordering and stale standalone worker fixtures. Rerun at be4c9397b8: 233 passed. Lint/format/diff checks pass;
commands and results in implementation.md. Gate/publication remain caller steps.

Concept-review cleanup (2026-11-16): job/run wording; RegimeRunRecord and
RegimeRun retained readers; shared schema/coordinate owners; retained builders
in test fixtures and belief demo in belief/; unused VerifyStore recovery/write
methods removed; AGENTS loop example corrected. Behavior and serialized keys stay.
Focused verification passes after correcting an unvalidated policy-import fixture;
commands/results in implementation.md. All six review fixes are complete.
