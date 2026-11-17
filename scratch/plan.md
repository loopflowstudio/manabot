# MTG-329: adopt Etude execution and agents

Design: design-train-api.md. Etude SDK 0.4.1 from PyPI; Pydantic required.
Jack Heart, 2026-11-16: "the whole point is the adoption"; "These should be
done/mainatined in this task". Full earlier plan and validation history:
3670249eb3:scratch/plan.md; exact remarks: positioning.md.

## Delete — do not maintain

Done: training/execution.py, recovery/admission/learning_state, job/runner/study,
queue/monitor worker, TrainingRecipe/RunContext, Tracker/LoopState, local_launch;
sim/serving, serving_protocol, eval/endpoint, execution, serving_benchmark,
allocation census and inference service; TrainingState/TrainingRuntime recovery,
collector restore_rng, VerifyStore, report selection/baseline, training/clock,
local_checkpoint_file, HardwareInventory, replay_prefix and host_facts.

PR #500's deletion steer also removed training/regime.py, records.py,
validation.py; experiment/records.py and checkpoint_evaluation.py; regime report
analysis/experiment/learning/notebook/records/snapshot/study/time_profile and
regime_study.ipynb; infra/host.py, ladder_precision and precision="auto";
read_updates, stage_rows, self_play_figures, training_dashboard, native_cohort;
saved-regime fixtures and exclusive tests. Removed phase/flamegraph/synthetic
profile reports, notebook extra, nbformat and matplotlib. No old-folder loader.

Preserve replay, Commands, admission, paired deals, cards, current-world
policies, released player/match/compute readers and frozen identities. Read
saved-regime reports from f4e1822faf46bbc9cdc43909293f8cbc8de4a300; state.pt
folders from f577362135562e569033fe33e48c7ef0411a829d. Frozen bytes never change.

## Built

Etude owns run execution/save/restore and managed batching through shared
Agents(load/build). The loop uses run.resume/steps/save/checkpoint; recipes are
train(config) plus training_jobs/eval_jobs. Library backend choice and Torch-free
host preparation remain separate. Cooperative Command/time caps, durable
journals and seeded paired deals survive. Native AgentError/PanicException is
invalid with null score, remaining games play; Python errors fail the match.
All-invalid Elemental groups fail after retaining evidence. Retry evidence uses
arena/<match-key>/attempt-* so interruptions never overwrite journals.

Reports read Etude runs only: explicit record kind/job takes precedence over
optional run/kind and run/recipe text metrics, otherwise unclassified. Only
local.py supplies smoke labels; baseline/Elemental do not. Silent attempts keep
earlier observed totals, count toward attempts, and warn about unmeasured time.
Training guide matches TrainingRun and the optional label arguments. Evaluation
guide uses the actual local EMA policy path. Saved-regime guide names what main
can no longer read. tla_frontier keeps its code settings and reads only the
134-deck deck_source from its frozen JSON. training-report remains the Etude
cross-run list. These reversible choices stand; no open Manabot product choice.

All shipped self-play recipe configs subclass `RunSettings` (frozen, extra keys
forbidden), so the four instrumentation keys are flat config keys and
`config.run_settings` passes the plain value to SelfPlay (8606b72454,
a23f14a56d). The collector's `no_operation` yields a NoOperation, so the
self-play step has no None guard.
Focused check: `uv run pytest -q tests/training/test_recipe_etude.py::test_elemental_records_games_for_one_update tests/training/test_training_game_journals.py::test_recipe_forwards_runtime_flag` — 3 passed, including a one-update CPU Elemental run that wrote game records.

## Validation

Evidence: implementation.md and experiments/reports/etude-execution.md. Focused
loop, recipe, replay, native/card/world/monitor/journal/boundary and report
checks passed; isolated dependency integration and installed 3.12/3.13 smoke
passed. The CPU uninterrupted comparison against main 3afd92d8f3 matched all
117 raw and 117 EMA tensors bitwise after three updates (seed 197, six games,
1,421 decisions, 268 optimizer updates), including initial policy tensors.
This settles only that short uninterrupted case; no strength claim follows.

CI 44136662771's one failure (experiment discovery omitted etude_execution)
is fixed; focused test passes (implementation.md). Other checks are the
caller's gate, including nightly baseline workflow_dispatch after the
choice-corpus alias fix. No full suite run in implement.

## GPU measurement complete — steer 8d007488

RunPod A6000, four alternating 256-stream Elemental arms: 954 decisions/s
branch (3670249eb3), 945 main (3afd92d8f3), opposite rankings across
pairs. The small write+fsync fell from 48.23 to 0.86 ms after sync.
Neither side stalled on its first durable write. Throttling on the earlier
Vast host remains unconfirmed. Direct SSH measures local Etude execution.

The first CUDA baseline exposed an invalid jagged-attention backward bound.
5826f7502e fixes it with a CUDA gradient regression (passed on A6000);
all three local padded/live parity cases passed. Both six-update trajectories
on 5826f7502e finished. Both endpoints won 11 of 128 replay-verified legs;
managed CUDA batches reached 14/13. Report and artifact manifest retain the
numbers, traces, differing source revisions and endpoint hashes. All machines
terminated and confirmed absent; cumulative launcher estimate ~USD 0.62.
No stall on a throttled disk was confirmed, so no Etude fsync Task was filed.
The timed arms predate 5826f7502e and the sync onto main dcee96ccd1 (merged
at c7fd53704a; choices in questions.md); neither was re-timed.

## Gate — 2026-11-17

Ruff check/format: 191 changed Python files pass. CI 44173418697 on
5583360199 failed only main's merged `test_player_crash_is_an_interrupted_record`
(a Python RuntimeError from a player; this branch propagates those). 0d4b8ba77b
raises a native AgentError there; CI 44175589618: all 25 checks pass.

## Remaining

Jack Heart, 2026-11-17: "ok, approved for landing" (steer 1fb8f71a).
Gate and publication carry the A6000 result and its limits into PR #500,
check CI on the final commit and report any remaining failure and its cause.
Remove scratch/ entirely in the pre-land commit, then land after green CI.

The dependency release
cut is complete. Known failure not caused by this branch:
`test_resource_limit_replay_rejects_old_schema...` diverges at action 32 of
the frozen TLA fixture under the current engine (questions.md). External limits: Etude Task 2d1933 owns hard per-match
deadlines (hang/death currently retries grouped job); 73e1a9 owns Lium lifecycle.
LLM jobs await EXP-70/frozen checkpoints. Automatic same-seed policy→sampler
scheduling and whether Agents(build=) builds every seat remain Etude questions,
not blockers. No real Rust panic/hang injection, browser or full-suite claim.
