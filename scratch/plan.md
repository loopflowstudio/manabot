# MTG-436: manabot's published library API, its contract check, and an external-consumer test

## Scope

The Task brief listed five items. A later comment on the Task narrowed it:

> Scope split: a sibling Task now owns making managym.ObservationBatch picklable and adding RolloutBatch.concatenate / ObservationSpace.max_actions for the 0.2.6 patch release. Focus MTG-436 strictly on: 1. Declaring manabot's published library API (`__all__` on the public library modules while keeping package `__init__.py` files empty, documented in manabot/README.md and AGENTS.md). 2. Adding the checked-in public API contract snapshot and pytest check (covering managym and manabot's declared public surface: symbols, callable signatures, dataclass/Pydantic fields). 3. Adding the in-repo external RL library consumer test (tests/infra/test_external_consumer.py exercising custom RL loop + open_collector/collect + AdmittedRollout + save_checkpoint/load_checkpoint + managym.catalog_tokens(); leave multiprocessing ObservationBatch pickling to the sibling 0.2.6 patch Task). 4. Documenting the patch-clean main release rule in AGENTS.md and docs/development.md (during a 0.Y.x cycle, main stays patch-compatible; next-minor 0.(Y+1).0 removal PRs wait to merge together right before cutting 0.(Y+1).0). Do not modify Rust ObservationBatch pickling or add RolloutBatch.concatenate in MTG-436 so there are no conflicts with the sibling 0.2.6 Task.

The sibling is MTG-437. This change touches no Rust and adds no library behavior. It stays patch-compatible with 0.2.5 and removes nothing deprecated. The Graph stops at `ship-review`.

## User-visible outcome

- **A library consumer** (jace is the live one) can tell directly from `manabot/README.md` or from `__all__` inside the installed `manabot` package which symbols are promised across `0.Y.x`, with each symbol published from its one defining module. Every published `manabot` module defines `__all__`, and no internal module carries a misleading `__all__`.
- **A contributor** who removes or reshapes a published symbol, parameter, field, attribute, or enum value gets a failing test that names the exact break, states the patch-clean `0.Y.x` rule first (restore or keep a `@deprecated` alias until `0.(Y+1).0`), and refuses to record breaks silently. Internal RL hot-path optimizations to opaque handles (`Prepared`, `Collector`, `Prefix`, `MemoryArchive`) and internal Pydantic validator refactors do not trigger false breaks.
- **A reviewer** sees every published API change as a readable diff of `tests/public_api.json`, sees every `0.3.0`-deprecated shim tracked explicitly so minor rollover only drops already-deprecated entries cleanly, sees any break-glass override recorded with its reason, and gets CI coverage whenever `manabot/README.md` edits the Library API table.

## Demo

```bash
uv run python scripts/public_api.py check        # "public API matches tests/public_api.json"
# rename the `num_envs` parameter of open_collector, then:
uv run python scripts/public_api.py check
#   BREAKING manabot.training.rollout.collection.open_collector: parameter num_envs removed
#   BREAKING manabot.training.rollout.collection.open_collector: required parameter envs added
#
#   main is on the 0.2.x patch line (pyproject.toml: 0.2.5).
#   Restore the published API or keep a DeprecationWarning alias targeting 0.3.0.
#   Only for an unavoidable break: uv run python scripts/public_api.py record --allow-break --reason '<why>'
uv run --extra dev pytest tests/test_public_api_contract.py tests/infra/test_external_consumer.py -q
```

## Acceptance check at gate

`uv run --extra dev pytest tests/test_public_api_contract.py tests/infra/test_external_consumer.py tests/test_package_boundaries.py tests/infra/test_ci_selection.py -q` passes, and `scripts/public_api.py check` fails (and plain `scripts/public_api.py record` refuses) after each of three temporary edits, each reverted: deleting a name from an `__all__`, renaming a parameter of a published function, and dropping a field from a published config.

## Design in brief

The published surface, contract format and classification rules live in `manabot/README.md#library-api`, `scripts/public_api.py` (docstring, `PUBLIC_MODULES`, `OPAQUE_TYPES`) and `docs/development.md#patch-clean-main`. Choices made for review are in `scratch/questions.md` and `scratch/concept-review.html`. Key invariants:

- Annotation text differs across CPython 3.12/3.13 and NumPy, so the contract records parameter/field names, kinds and required-ness only.
- 53 of managym's 128 public callables have no runtime signature (PyO3), so managym's contract reads `managym/__init__.pyi` with `ast`.
- `getattr` does not trigger method/decorator deprecations, so deprecation is read from `@deprecated` (stub AST and `__deprecated__`), stub docstrings/comments (`Deprecated` docstring or inline/immediately-preceding `# Deprecated`), and module `__getattr__` warnings.
- Pydantic `@field_validator`, `@model_validator`, `@field_serializer`, and `@model_serializer` callbacks (`cls.__pydantic_decorators__`) are internal hooks, not public methods; real methods and all fields on published `BaseModel` classes remain recorded.
- Each published `manabot` symbol is exported from its defining module only (no duplicate re-exports across `PUBLIC_MODULES`).
- `manabot/README.md` is a test-parsed contract input (`test_readme_lists_the_published_exports`), so `.github/ci/select-tests.sh` selects `integration` when it changes.

## Status

1. Export lists, contract recorder and compatibility tests: `d70731a562`, compressed in `06f2634d42`, QA'd in `bb824541f4`.
2. External CPU learner and README example fix: `c205dfb579` (0.72 s test body).
3. CI selection and integration wiring: `c205dfb579`.
4. README-only changes select integration (`af931f56d4`). The contract excludes Pydantic callbacks, publishes `PackedActions` from its defining module, and recognizes inline and preceding stub deprecation comments (`8d665adbdd`).

The regenerated 0.2 baseline dropped ten callback entries and the duplicate export. It marked `PlayerConfig.disclosure` and `Observation.legacy_card_lists` deprecated and retained an empty break history. The contract check and Ruff passed. The four pytest files listed above finished with 118 passed and one skip (MLX unavailable).

Implementation is ready for review. Gate and ship-review remain.

## Coordination with MTG-437

MTG-437 adds `concatenate` and `max_actions` to published classes. Whichever Task lands second runs plain `uv run python scripts/public_api.py record` (additions only, no `--allow-break`).

## Exclusions

`ObservationBatch` pickling and `concatenate`/`max_actions` (MTG-437); removing 0.3.0-deprecated APIs (MTG-435); managym Python submodules (follow-up Task); per-attribute deprecation markers for `Card`/`Permanent` flat characteristic aliases in `managym/__init__.pyi`; MLX backend, recipes, loops and CLI (internal).
