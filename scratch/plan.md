# MTG-324 — precision default

## Outcome and evidence

Fresh precision authoring defaults to `auto`; BF16 is enabled only if the
learning gate admits it. The Task explicitly provides the rejection outcome:
`auto` resolves to float32 everywhere. Explicit precision, saved regime
numerics, checkpoint compatibility and frozen world/input bindings stay intact.

The learning gate rejected BF16. Retained `2be568a65518/training/run.json`
records `behavior_log_probability_mismatch`, iteration 141, 140 completed
updates, `rejected_updates: 1`, `nonfinite_failure: 0`. Float32 seeds reached
188/170 updates; the other BF16 seed reached 290. Shared scheduled checkpoint
M = 75. No strength evaluation was launched: numerical rejection already
refutes the pre-registered gate. No strength or trajectory-equivalence claim.
Report: `experiments/precision_ladder.md`; raw records and figure are in the
artifact manifest. Launcher training cost USD 24.35; evaluation USD 0.

The w256l8 L4 run (`f30d26f9bc7a`) measured a 1.66× bucketed learner speedup,
recorded in `experiments/cuda_precision.md`. Width 256 is the measured speed
crossover, inactive after the learning rejection. No arm reached 300 updates;
width 256 has no learning comparison, and the mismatch's cause is unknown.

## Implementation

`manabot/training/regime.py` owns `Execution.precision`,
`BF16_MIN_HIDDEN_DIM = 256` and `ladder_precision`, which returns float32 for
all widths/devices after rejection. `TrainingRegime.resolve_forward_layout`
resolves precision and both layout fields together. `authored_dump` uses
`model_fields_set` to restore automatic choices for builders while preserving
explicit assignments/copies. `_read_saved_stage` reads omitted execution
fields as float32/off/off. There is no second resolver or private origin flag.

`manabot/training/execution.py` passes `resolved_precision` to the Torch
learner and collector; precision handling remains in
`manabot/backends/torch/learner.py`. `experiments/precision_ladder.py::arm`
explicitly retains `live_tokens=False`; the frozen A/B does not admit the
BF16/live-token combination. `docs/training.md` describes the rejection.

Main's shared authoring mechanism (PR #502) and Torch backend move are merged.
On 2026-11-13, fetched `origin/main` at `6ce618b8a` had zero commits absent
from HEAD `0338b8a90`; no additional upstream reconciliation was needed.

## Direction retained

Jack Heart, 2026-11-12: “When you resume after the A/B, merge main and express
precision="auto" through that same mechanism (resolve_forward_layout /
authored_dump / _read_saved_stage), one resolution path for all three fields,
not a second ladder.”

Budget steer: “Run the full-length version, about 300 updates per arm, 2 seeds
per arm, float32 vs BF16 at w384l8 with bucketed shapes, plus its evaluation;
Task cap is now USD 35 total.” Earlier plan/history: git `e181224e7`.

## Remaining and limits

- Implementation and evidence publication are complete: the manifest contains
  seven A/B files and three w256 crossover files. Gate, review and PR
  publication remain. Existing checks below are evidence, not a completed gate.
- No new experiment or paid evaluation is planned. Omitting strength evaluation
  after the decisive numerical rejection is recorded in `scratch/questions.md`
  and the report; it does not establish strength or trajectory equivalence.
- Separate follow-up candidate: investigate the retained probability mismatch;
  BF16 admission would require new learning evidence. No numerical-guard or
  frozen-evidence change belongs to this implementation. No product decision
  is needed to retain the specified float32 fallback.

## Concept review (2026-11-13)

Nothing to change, no open product choice. One concept reaches authors:
`precision="auto"` (float32 today), resolved and saved with the two layouts.
Normal path, saved-regime resume and builder rebuild keep explicit and recorded
values. `ladder_precision` and `BF16_MIN_HIDDEN_DIM` stay despite being
inactive: the report's paragraph committed before the scored run names both,
and the Task's rejection outcome is "`auto` resolves to `float32` everywhere".
`resolve_forward_layout` now also resolves precision; its docstring says so and
the name is the one Jack's direction cites, so no rename here.

## Validation

Prior implementation run: `uv run --extra dev --extra artifacts pytest tests/training/test_precision.py tests/training/test_bucketed_execution.py -q` had 35 passing cases and one failing fresh-default fixture. After correction, its focused rerun plus `tests/eval/test_experiment_files.py` passed (5 tests). Artifact tracking check: 1 passed. Ruff and `git diff --check` passed.

Compress: 12 focused precision cases passed. The consolidated override test
initially expected disabled layouts to be present in dumps; corrected the test
to respect their omission. `uv run --extra dev --extra artifacts pytest tests/training/test_bucketed_execution.py::test_builder_keeps_explicit_edits_after_auto_resolution -q`: 10 passed. Ruff check/format and `git diff --check` passed. No runtime changes or paid runs.

Realign changed only scratch context. Code, tests and report were inspected;
existing passing checks were not rerun. `git diff --check` passed and
`lf context` confirmed scratch remains within both size targets.
