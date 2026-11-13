# MTG-324 — precision default

## Decision and implementation

The learning gate rejected BF16. Retained `2be568a65518/training/run.json`
records `behavior_log_probability_mismatch`, iteration 141, 140 completed
updates, `rejected_updates: 1`, `nonfinite_failure: 0`. Float32 seeds reached
188/170 updates; the other BF16 seed reached 290. Shared scheduled checkpoint
M = 75. No strength evaluation was launched: numerical rejection already
refutes the pre-registered gate. No strength or trajectory-equivalence claim.
Report: `experiments/precision_ladder.md`; raw records and figure are in the
artifact manifest. Launcher training cost USD 24.35; evaluation USD 0.

`auto` now resolves to float32 everywhere. Width 256 remains the measured
speed crossover, inactive after the learning rejection. Explicit precision
and saved regime numerics remain unchanged.

Main is merged, including PR #502 and the Torch backend move. Precision uses
`resolve_forward_layout`, `authored_dump`, `model_fields_set` and
`_read_saved_stage` alongside both layout fields. Explicit assignments and
copies after resolution stay explicit; saved omissions mean float32/off/off.
The A/B arm explicitly retains `live_tokens=False`.

## Delete — do not maintain

- Removed `resolve_precision_ladder`, `_from_ladder`, custom `Execution.__eq__`
  and `_read_saved_precision`: one upstream resolution path owns all defaults.
- Kept main's deletion of `manabot/training/precision.py`; Torch precision
  handling lives in `manabot/backends/torch/learner.py`.
- Preserve explicit choices, saved hashes, runtime guards and authored rebuilds.
- Removed the unverified docs assertion that BF16 passed the learning A/B.
- Removed `_precisions` and the duplicate precision-only override test. The
  parameterized test in `tests/training/test_bucketed_execution.py` covers
  assignment/copy overrides for all three automatic fields, including the
  untouched fields' authoring defaults. Precision cases independently cover
  each explicit value, saved round trips and width changes.
- The experiment module introduction now describes its entry points; the
  report owns the prediction, measurements and launch history. No remaining
  deletion targets in the production resolution path.

## Direction retained

Jack Heart, 2026-11-12: “When you resume after the A/B, merge main and express
precision="auto" through that same mechanism (resolve_forward_layout /
authored_dump / _read_saved_stage), one resolution path for all three fields,
not a second ladder.”

Budget steer: “Run the full-length version, about 300 updates per arm, 2 seeds
per arm, float32 vs BF16 at w384l8 with bucketed shapes, plus its evaluation;
Task cap is now USD 35 total.” Earlier plan/history: git `e181224e7`.

## Remaining

- Implementation and seven published evidence files are complete.
  Gate/review/publication remain caller steps.
- Separate follow-up candidate: investigate the retained probability mismatch;
  do not loosen the numerical invariant or change frozen experiment evidence.

## Validation

`uv run --extra dev --extra artifacts pytest tests/training/test_precision.py tests/training/test_bucketed_execution.py -q`: 35 passed; corrected the one failing fresh-default fixture, then its focused rerun plus `tests/eval/test_experiment_files.py` passed (5 tests). Artifact tracking check: 1 passed. Ruff and `git diff --check` passed. No repeated passing suite; broader acceptance belongs to gate.

Compress: 12 focused precision cases passed. The consolidated override test
initially expected disabled layouts to be present in dumps; corrected the test
to respect their omission. `uv run --extra dev --extra artifacts pytest tests/training/test_bucketed_execution.py::test_builder_keeps_explicit_edits_after_auto_resolution -q`: 10 passed. Ruff check/format and `git diff --check` passed. No runtime changes or paid runs.
