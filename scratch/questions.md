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
size-ladder benchmark. Use its documented command once shipped; current tooling
remains the reference until then. No guessed CLI or duplicate benchmark is added.

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

The earlier documentation preview could not open: `lf open --host desktop --file
docs/workflows.md` reported “No Desktop Files view attached; nothing opened.”

## Remaining work

Compression keeps runnable benchmark examples and evaluation plan requirements
in `docs/workflows.md` and `docs/evaluation.md`; the measurement skill links to
those owners instead of repeating commands and the plan schema. The compiled
loop check names its return target directly rather than relying on list offsets.
No further deletion targets were found. Context is below both size limits.

Implementation and focused checks are complete; review, gate and delivery remain
with the selected Task graph. No outstanding product question or additional scope
was found. No benchmark, training/evaluation job, rental or server was started.

After compression: `uv run python scripts/check_loopflow_workflows.py` — all eight graphs
compile and resolve, including the new 13-node workflow; loop/delivery/scientific
boundaries pass. Focused Ruff check/format and `git diff --check` pass.
