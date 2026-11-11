# 4ceb1d — experiment performance workflow

## Direction and design

Task 4ceb1d adds `experiment-performance` for training/evaluation speed. The
Task brief quotes Jack Heart's focus as "how fast can we get learning to happen"
and "inference during training time: basically training speed, inference speed,
per size of model". The shipping guide is `docs/workflows.md`, “Training and
evaluation performance”; its documentation handoff was preserved during implementation.

The graph uses measurement → implement → compress → builtin refresh → concept
review → performance review → implementation loop, followed by the same
publish/demo/deliver/verify boundaries as `engine-performance`. Two repository
skills own measurement and review. Existing workflow validation checks the new
compiled loop, reference resolution and delivery boundaries without execution.

Reversible choice: keep this single consumer's optimization steps in
`experiment-performance.yaml`, composing builtin refresh and deliver directly.
No second optimization subgraph is needed; the engine's shared subgraph remains
unchanged because Coverage consumes it. Task 855bff owns the future standing
size-ladder benchmark. Its 2026-11-11 status read showed no PR or Graph process;
the current benchmark remains the reference until its documented command ships.
No guessed CLI or duplicate benchmark is added.

## Delete — do not maintain

No production deletion targets: this is a new workflow with a distinct domain.
Preserve `engine-performance`, `engine-optimization`, Coverage composition,
scientific experiment boundaries, frozen recipes/worlds/inputs and retained data.
Inherited b015d5 execution notes were removed from active scratch; their complete text
is retained at `3c169916e2d642b5ead16aa303b979232635a63f:scratch/questions.md`.
They grant no execution authority to this Task. Its report remains at
`experiments/data/training-throughput-b015d5-v1/report.md`, with maintained
interpretation in `experiments/training-update-time.md`.

## Evidence and limits

The documentation step restored the b015d5 report from the local artifact cache
(one file, zero network bytes); its SHA-256 matched the manifest. The report's
C1/C2 analyses support same-host controls, matched post-warmup ordinals and
separate per-seed latency/aggregate throughput. Its earlier “two-seed ceiling”
wording is qualified by the maintained summary: three seeds were not tested.
Precision health did not establish learning equivalence. These observations
inform the skill boundaries; no training, evaluation, rental or campaign is
needed to implement or validate the workflow.

Reconciliation found Infrastructure memory still describing full CUDA admission
as unproved. The retained report's repaired C1 section records successful
regression/worker/checkpoint admission on 2026-11-10; C2 adds large-model packing
with 6–7 matched updates per seed. Memory now distinguishes those later results
from the original failed/short attempts, preserves precision and one-host limits,
and carries the selector's unknown-state retry lesson. No child Infrastructure
memories exist in this checkout. The Wave objective still fits.

The measurement and review skills now explicitly separate per-seed latency from
aggregate throughput/cost and require overlapping packed samples with matched
solo ordinals. This follows the existing C1/C2 evidence; it adds no campaign.
Upstream's fresh-device selection does not change this workflow's CPU example:
`scripts/bench_training_update.py:main` still explicitly defaults `--device` to
`cpu`. No executable mismatch was found.

## Implementation and remaining work

Documentation handoff (2026-11-11): `docs/workflows.md` remains the shipping
guide. The write-docs pass preserved the existing examples and added separate
baseline/candidate output paths, interrupted-receipt interpretation and recovery
from invalid comparisons. No open product question or new implementation scope.
`lf open --host desktop --file docs/workflows.md` returned “No Desktop Files view
attached; nothing opened.” The next step should read the guide in this checkout.

Compression keeps runnable benchmark examples and evaluation plan requirements
in `docs/workflows.md` and `docs/evaluation.md`; the measurement skill links to
those owners instead of repeating commands and the plan schema. The compiled
loop check names its return target directly rather than relying on list offsets.
No further deletion targets were found.

The implementation recheck corrected QA guidance that conflated actual-workload
benchmarks with forbidden artificial contention. `TESTING.md` permits the former;
workflow-only QA needs neither. The QA transcript now uses uv for Python commands.
No executable behavior or graph changed, so the passing compiler and Ruff results
below remain applicable.

Implementation and concept review are complete; gate and delivery remain with the
selected Task graph. No outstanding product decision, runtime repair or additional
scope was found. No benchmark, training/evaluation job, rental or server was started.
The documentation preview previously returned “No Desktop Files view attached;
nothing opened.” Human presentation remains for demo; it does not block headless
workflow validation.

After realignment: `uv run python scripts/check_loopflow_workflows.py` — all eight graphs
compile and resolve, including the new 13-node workflow; loop/delivery/scientific
boundaries pass. `git diff --check` passes. Compression's focused Ruff check/format
results remain applicable: realignment changes only Markdown. `lf context` reports
memory and scratch below both configured limits.

## Concept review — 2026-11-11

Outcome for loop-or-next: nothing to change and no open product choice. The
headless walkthrough checked a slow policy-inference update from graph selection
through retained comparison, behavior review and software delivery. Existing
`TrainingRun`, update rows, evaluation plans and receipts carry the evidence;
the graph adds orchestration without a second experiment or persistence model.

The principal conceptual risk is treating faster updates as faster learning.
The implemented guide and skills already keep precision/sampling adoption
dependent on separate learning evidence. Counterexamples also remain explicit:
packing can slow each seed, an interrupted receipt can mix attempt-local timers
with cumulative counts, and capacity failure cannot justify a smaller workload.
Recovery retains failed evidence, revises invalid comparisons before further
implementation, and leaves unavailable CUDA as an exact deferred check. These
paths require no additional product decision within this Task's stated scope.

The installed compiler exposed the expected 13 nodes, implementation return
edge and interactive demo/verify returns to measurement during this review.
Next presentation: `scratch/concept-review.html`, showing routing, evidence
ownership and failure/recovery. No benchmark or execution was started.
