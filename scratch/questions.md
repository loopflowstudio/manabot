# 4ceb1d — experiment performance workflow

## Direction and design

The Task brief quotes Jack Heart's focus as "how fast can we get learning to happen"
and "inference during training time: basically training speed, inference speed,
per size of model". `docs/workflows.md`, “Training and evaluation performance”,
is the shipping guide; `experiment-measure` and `experiment-performance-review`
own measurement and review. Software delivery grants no scientific execution or
spending authority and cannot complete a parent experiment.

Reversible choice: keep this single consumer's optimization loop in
`.lf/graphs/experiment-performance.yaml`, composing builtin refresh and deliver.
Coverage still consumes the engine's shared `engine-optimization`; no second
optimization subgraph is needed. Task 855bff owns the standing size-ladder
benchmark. Its 2026-11-11 22:04 UTC status showed an active `code` Graph at
`write-docs`, with no PR. Use the current benchmark until its documented command
ships; add no guessed CLI or second ladder.

## Delete — do not maintain

No production deletion targets: this is an additive workflow. Preserve the engine
graphs, Coverage composition, scientific boundaries and frozen evidence/configs.
Inherited b015d5 execution notes are retained at
`3c169916e2d642b5ead16aa303b979232635a63f:scratch/questions.md`; they grant this
Task no execution authority. Earlier workflow review notes were checkpointed at
`bacd71935f527bfe24278bc60ea2a50a13243e13:scratch/questions.md` before compression.

## Evidence and limits — 2026-11-11

The documentation step restored
`experiments/data/training-throughput-b015d5-v1/report.md` from the local artifact
cache (one file, zero network bytes); SHA-256 matched the manifest. C1/C2 support
same-host controls, overlapping packed samples and matched solo ordinals, with
per-seed latency separate from aggregate throughput. Three seeds were not tested,
so the report's earlier “two-seed ceiling” is qualified by
`experiments/training-update-time.md`. Precision health is not learning equivalence.

Reconciliation found Infrastructure memory's CUDA admission status stale. Repaired
C1 passed regression/worker/checkpoint admission on 2026-11-10; C2 supplied 6–7
matched large-model updates per packed seed. The reconciled memory preserves the
original failed/short attempts, one-host limits and unknown-acquisition retry
lesson. No immediate Infrastructure child memories were found; the Wave objective
still fits. Related Experiments memory coverage is recorded in Infrastructure
memory, not claimed as a full review here.

Concept review walked a slow policy-inference update through evidence and delivery
and found no open product choice. Existing runs, update rows, evaluation plans and
receipts suffice. Guard against treating faster updates as faster learning,
packing as faster per-seed progress, and last-attempt timers as cumulative timers.
Capacity failure or a behavior mismatch invalidates the comparison; retain failed
evidence and revise the plan. Missing CUDA stays an exact deferred check.

The post-realignment concept review reaches the same outcome: no design change
and no open product choice. Its concrete counterexample is a small inference batch
caused by complete-game closure, not a serving limit. The implemented measurement
guidance already handles this; changing closure would cross the existing sampling
boundary and require separate learning evidence. Evaluation recovery retains the
request identity and completed evidence; a cached success adds no timing sample.
Next: show that evidence trail at demo, without starting a benchmark in this review.

## Implementation and remaining work

Compression keeps runnable commands in `docs/workflows.md` and evaluation plan
requirements in `docs/evaluation.md`, with links from the measurement skill.
Benchmark settings and output retention now sit together in the guide; repeated
review history is reduced here. The compiled-loop check names its return target
directly. No further code simplification was warranted.

QA's repository guidance distinguishes actual-workload measurements from forbidden
artificial contention (`TESTING.md`); workflow-only QA needs neither. The CPU
example still matches `scripts/bench_training_update.py:main`'s explicit default.
The installed compiler exposed 13 nodes, an implementation return edge and
interactive demo/verify returns to measurement. After realignment,
`uv run python scripts/check_loopflow_workflows.py` passed all eight graphs and
loop/delivery/scientific boundaries; focused Ruff check/format also passed.

Implementation and concept review found no outstanding runtime repair or product
question. Gate and delivery remain with the selected Task graph. Presentation:
`scratch/concept-review.html` and `scratch/qa.html`.
`lf open --host desktop --file docs/workflows.md` returned “No Desktop Files view
attached; nothing opened.” Human presentation remains for demo. No benchmark,
training/evaluation job, rental or server was started.

Realignment found fetched `origin/main` at `3c169916e`, already included here.
Task 855bff's new collection finding was checked against
`manabot/training/rollout/collection.py`: complete-game streams idle after reaching
their closure target. Measurement now reads `StageRecord.idle_stream_steps`
without treating a legacy default zero as evidence of no idle slots. The benchmark
docstring's cross-machine timing claim was corrected to match same-machine controls.
These are guidance changes, not collector changes or new speed measurements.
The workflow compiler passed all eight graphs after these corrections;
`git diff --check` passed and `lf context` remained within both budgets.

## Main sync — 2026-11-11

Reversible resolution: keep this branch's experiment-performance context active.
Main's MTG-271 direction, exact comments and dependency/release findings remain at
`9f69a96a1ca5f0c03a9df5af7217138390d415d9:scratch/questions.md`. They do not
select work here. QA combines the workflow guidance with main's Academy launch,
clean-machine and library-consumer checks, using its detailed Academy procedure.
