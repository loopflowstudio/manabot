# MTG-281 collection growth

Read the retained w64 `59718fe34945` and w160 `5b7d8fa7b38c` updates one block
at a time, keeping timing, work, RSS, game/error and provenance fields. Plot
against updates, permanent block count/header bytes and game length; distinguish
reconstructed index history from directly retained observations. Compare the
models at equal update counts and explain the counterevidence.

Choose a matched before/after horizon from that evidence, retaining CUDA,
seed 21501, the w64 recipe and L4 machine type. Sources are `1659b0e29^` and
`1659b0e29`; all rental costs must fit USD15. Do not touch the original runs,
arena disk or EXP-46. If the evidence-based horizon cannot fit, report that
limit rather than spend on a shorter run that cannot answer the question.

## Delete — do not maintain

No production implementation is replaced. Inherited MTG-271 direction is removed
(history reference in `questions.md`). The capture-only command no longer has
a subcommand dispatcher; report generation no longer writes into its inputs.
Preserve frozen runs, source recipes, metric values and the MTG-273 index
reproduction. No predecessor code remains to remove. The diagnostic projection
stays scoped to this measurement, not a second general Etude reader.

Keep batched ranges in `scripts/measure_collection_growth.py::_arrays`: Etude's
`StoredBlock.array` would make a remote request per metric; its normal header
reader materializes the omitted matchup columns. `report_collection_growth.py`
accumulates header bytes once and assigns each update to its nearest endpoint
once. New report directories and exclusive snapshot creation protect retained
evidence from reproduction commands.

## Retained result and remaining work

The report and seven manifest-published artifacts retain 1744 w64 and 932
w160 measured updates. Both show RSS growth and collection spikes immediately
after permanent-block endpoints. Exact historical manifest sizes/fold times,
final CPU quotas/models and final game-error counts were not retained in the
available live evidence; do not infer zero errors or causal isolation.

The prospective comparison is 500 updates per arm, CUDA/L4/seed 21501,
unchanged damping horizon 2474, source `b7c102f2c4266dab4be91b8b54901861745b9a3c`
versus `1659b0e290f55daa632ec8f869adff4a3059d79f`. Reserve $7.20/12 whole-rental
hours per arm plus $0.60 ancillary within $15; no pilot or automatic retry.
Historical 500-update elapsed is 10.05 hours, not a fresh quote or future bound.

At 2026-11-11 22:42:24 UTC, configured L4 availability failed with
`ProviderUnavailable: GET catalog/gpus: no reply`. No rental/run was created;
new rental spend $0. Remaining: fresh price/capacity and complete source/runtime
admission, matched execution, comparison, errors/recovery and whole-rental
accounting/teardown. The Task's scientific outcome is not complete. No remaining
cause follow-up is justified before observing the repaired arm.

Check: `PYTHONPATH=. uv run --no-sync pytest tests/infra/test_collection_growth_capture.py tests/infra/test_experiment_data_untracked.py -q`
— 7 passed; Ruff passed. Offline regeneration in
`.runs/collection-growth/compress-report/` produced byte-identical `summary.json`,
`phases.png` and `growth.png`; existing evidence-directory output was refused.
All seven retained files still match the manifest. No rental or training ran.

Gate: `PYTHONPATH=. uv run --no-sync pytest --collect-only -q tests/infra/ tests/academy/test_clean_machine_verifier.py deploy/play/tests/`
collected 161 tests, then failed on `test_runtime_config.py`: this checkout has
no `manabot.managym._managym` extension. Complete that collection with the built
native runtime; this diagnostic itself needs no native imports.
