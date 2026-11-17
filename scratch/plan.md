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

## Validation

Evidence: implementation.md and experiments/reports/etude-execution.md. Focused
loop, recipe, replay, native/card/world/monitor/journal/boundary and report
checks passed; isolated dependency integration and installed 3.12/3.13 smoke
passed. The CPU uninterrupted comparison against main 3afd92d8f3 matched all
117 raw and 117 EMA tensors bitwise after three updates (seed 197, six games,
1,421 decisions, 268 optimizer updates), including initial policy tensors.
This settles only that short uninterrupted case; no strength claim follows.

Current implement: CI 44136662771 confirmed exactly one failure (197 passed):
experiment discovery omitted etude_execution. Expected membership now includes
it; focused test passes (1 passed, 12 deselected, 8.54s). Other checks remain
the caller's gate, including nightly baseline workflow_dispatch after the
choice-corpus alias fix. No full suite run in implement.

## Remaining — latest steers 54c7d853 / 7d745eb2

1. GPU speed, CUDA batching and resume measurement using the existing harness
   experiments/etude_execution.py; method and attempts live in its report.
   Main 3afd92d8f3 versus branch on one reasonable single GPU; two alternating
   three-update Elemental runs per revision, seed 197, normal 256 streams;
   six-update branch baseline uninterrupted and stopped after update 3 then
   resumed; 16 paired eight-leg deals per endpoint. Calibration only, no
   strength equivalence claim. Direct SSH does not prove queued job delivery.
   Total approved --max-usd 10, about USD 0.13 spent before this run. Lium
   excluded (Etude Task 73e1a9); RunPod or Vast only; retry a few times across
   this run on capacity failure/host death, terminate and confirm every machine.
   Earlier Vast host died during compilation, Lium creation was unreadable;
   both confirmed deleted. At 16:09 UTC eight templates had no capacity.
   Latest retry: branch 3670249eb3, Vast 4090 instance 61113127. CUDA check
   passed, but zero updates: repeated durable-write stalls (traceback at
   Etude run.py:412 os.fsync). Terminated and provider-confirmed absent at
   19:17:48 UTC; launcher estimate ~USD 0.14. Nine follow-up RunPod/Vast
   capacity checks found no alternate host. No rental/server/watcher remains.
   GPU acceptance is still unmet; a repeat needs working durable storage.
2. CI expectation fixed above; leave CI rerun/publication to caller.
3. PR #500 body corrected: saved-regime reports cannot be rebuilt from main;
   name f4e1822faf and f577362135, replace the stale no-L4 sentence with the
   actual GPU retry result. Description updated; gate/publication stay with caller.

Gate and publication remain caller's steps; do not land. Dependency release
cut is complete. External limits: Etude Task 2d1933 owns hard per-match
deadlines (hang/death currently retries grouped job); 73e1a9 owns Lium lifecycle.
LLM jobs await EXP-70/frozen checkpoints. Automatic same-seed policy→sampler
scheduling and whether Agents(build=) builds every seat remain Etude questions,
not blockers. No real Rust panic/hang injection, browser or full-suite claim.
