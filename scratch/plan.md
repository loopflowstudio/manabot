# MTG-436: manabot's published library API, its contract check, and an external-consumer test

## Scope

The Task brief listed five items. A later comment on the Task narrowed it:

> Scope split: a sibling Task now owns making managym.ObservationBatch picklable and adding RolloutBatch.concatenate / ObservationSpace.max_actions for the 0.2.6 patch release. Focus MTG-436 strictly on: 1. Declaring manabot's published library API (`__all__` on the public library modules while keeping package `__init__.py` files empty, documented in manabot/README.md and AGENTS.md). 2. Adding the checked-in public API contract snapshot and pytest check (covering managym and manabot's declared public surface: symbols, callable signatures, dataclass/Pydantic fields). 3. Adding the in-repo external RL library consumer test (tests/infra/test_external_consumer.py exercising custom RL loop + open_collector/collect + AdmittedRollout + save_checkpoint/load_checkpoint + managym.catalog_tokens(); leave multiprocessing ObservationBatch pickling to the sibling 0.2.6 patch Task). 4. Documenting the patch-clean main release rule in AGENTS.md and docs/development.md (during a 0.Y.x cycle, main stays patch-compatible; next-minor 0.(Y+1).0 removal PRs wait to merge together right before cutting 0.(Y+1).0). Do not modify Rust ObservationBatch pickling or add RolloutBatch.concatenate in MTG-436 so there are no conflicts with the sibling 0.2.6 Task.

The sibling is MTG-437. This change touches no Rust and adds no library behavior. It stays patch-compatible with 0.2.5 and removes nothing deprecated. The Graph stops at `ship-review`.

## User-visible outcome

- **A library consumer** (jace is the live one) can tell directly from `manabot/README.md` or from `__all__` inside the installed `manabot` package which symbols are promised across `0.Y.x`. Every published `manabot` module defines `__all__`, and no internal module carries a misleading `__all__`.
- **A contributor** who removes or reshapes a published symbol, parameter, field, attribute, or enum value gets a failing test that names the exact break, states the patch-clean `0.Y.x` rule first (restore or keep a `@deprecated` alias until `0.(Y+1).0`), and refuses to record breaks silently. Internal RL hot-path optimizations to opaque handles (`Prepared`, `Collector`, `Prefix`, `MemoryArchive`) do not trigger false breaks.
- **A reviewer** sees every published API change as a readable diff of `tests/public_api.json`, sees every `0.3.0`-deprecated shim tracked explicitly so minor rollover only drops already-deprecated entries cleanly, and sees any break-glass override recorded with its reason.

## Demo

```bash
uv run python scripts/public_api.py check        # "public API matches tests/public_api.json"
# rename the `num_envs` parameter of open_collector, then:
uv run python scripts/public_api.py check
#   BREAKING manabot.training.rollout.collection.open_collector: parameter num_envs removed
#   BREAKING manabot.training.rollout.collection.open_collector: required parameter envs added
#
#   main is on the 0.2.x patch line (pyproject.toml: 0.2.5).
#   Do not break published APIs on 0.2.x: restore the old API or keep a
#   DeprecationWarning alias targeting 0.3.0.
#   Only if a break on 0.2.x is unavoidable:
#     uv run python scripts/public_api.py record --allow-break --reason '<why>'
uv run --extra dev pytest tests/test_public_api_contract.py tests/infra/test_external_consumer.py -q
```

## Acceptance check at gate

`uv run --extra dev pytest tests/test_public_api_contract.py tests/infra/test_external_consumer.py tests/test_package_boundaries.py tests/infra/test_ci_selection.py -q` passes, and `scripts/public_api.py check` fails (and plain `scripts/public_api.py record` refuses) after each of three temporary edits, each reverted: deleting a name from an `__all__`, renaming a parameter of a published function, and dropping a field from a published config.

## Source of truth

- **manabot**: the fifteen published modules listed once in `PUBLIC_MODULES` (`scripts/public_api.py`) and each module's `__all__`. To make `__all__` truthful inside the installed wheel as well as in the checkout, legacy unused `__all__` lists on internal modules (`manabot/belief/*.py`, `manabot/semantic/learning.py`) are removed and `tests/test_public_api_contract.py` asserts that (a) no unpublished `manabot` module defines `__all__` (except explicitly listed `0.3.0`-deprecated shims: `manabot/sim/search/flat.py`), and (b) `manabot/README.md`'s Library API module list matches `PUBLIC_MODULES`.
- **managym**: `managym/__init__.pyi`, which already is its published top-level boundary (`managym/__init__.py` docstring). No `__all__` is added there.
- **The contract**: `tests/public_api.json`, derived from those two and nothing else.

## Findings that shape the design

**What the external consumer uses.** jace at `9d212cb` (2026-11-24) imports from the fifteen `manabot` modules listed below, plus `managym` and four `managym` submodules (`recording`, `enums`, `policy`, `disclosure`). It relies on methods of the classes it imports, not only module-level names: `ActorCritic.forward_distribution`, `.prepare`, `.inference_scope`, `.cache_programs`, `.clear_cache`; `AdmittedRollout.admit_rows`, `.minibatches`, `.next_observations`, `.legal`; `RolloutBatch.recorded` and its `rows`, `index`, `needs`, `memory`, `gather_needs`; `ObservationSpace.encode`, `.shapes`, `.config`, `.encoder.max_actions` (the last is restored by MTG-437). So the contract records the public members of non-opaque published classes, including inherited ones (`admit_rows` lives on `AdmittedRows` in `manabot/training/rollout/admission.py`). It imports `PackedActions` from `manabot.training.rollout.collection`, where it is a re-export, so `__all__` there lists it.

**No-underscore does not mean external API in `manabot`.** `AGENTS.md` and `tests/test_package_boundaries.py` (`test_shared_manabot_imports_use_public_names`) forbid leading underscores on internal helpers shared across modules. Freezing all non-underscore dataclass fields of internal transfer/state containers would lock down `Prepared` (62 host-to-device routing fields), `Collector` (21 internal stream/snapshot/profiler fields), `Prefix` (4 internal trie fields referencing unpublished `PrefixIdentity`/`Tail`/`RecordVersion`), and `MemoryArchive` (6 internal chunk-storage fields) — breaking internal RL throughput optimizations on `0.2.x` for zero benefit to external consumers. Those four classes are therefore recorded as **opaque handles** (`kind: "opaque"`): their class identity is promised for annotations and `isinstance`, while their internal fields stay free to evolve.

**Runtime annotation text is not stable.** Runtime introspection of the fifteen modules takes ~1 s and is byte-identical across hash seeds, but annotation strings leak the interpreter and NumPy version (`str | pathlib._local.Path` on 3.13 vs `pathlib.Path` on 3.12; `numpy.ndarray[tuple[typing.Any, ...], ...]` for `NDArray`). CI runs 3.13 and consumers also run 3.12, so the contract records structural shape and no annotation text.

**PyO3 objects have no signatures and use stub attributes/enums.** 53 of `managym`'s 128 public callables raise from `inspect.signature`, and PyO3 classes (`Player`, `Card`, `Permanent`, `Action`, `Observation`, `ContinuationFrame`, ...) expose attributes via class-body `AnnAssign` rather than dataclass/Pydantic fields. `managym`'s contract therefore reads `managym/__init__.pyi` with `ast`, recording functions/`@overload`s, class `AnnAssign` attributes, `@property` getters, methods/staticmethods/classmethods (skipping leading `_` such as `VectorEnv._load_shard_buffers` except `__init__`), `IntEnum` member names and integer values, `TypedDict` keys (with `NotRequired`), constants/aliases, and explicit stub re-exports (`format as format`; unaliased annotation imports are private to the stub). The check also verifies every non-type-only stub name resolves on `managym` at runtime (excluding TypedDicts and `Literal` type aliases).

**Deprecations live on decorators and methods, not just module `__getattr__`.** Reading `getattr(managym, "learning_catalog")` or `getattr(ObservationSpace, "encode")` does not emit a `DeprecationWarning` — calling or instance-access does. The codebase already uses `from typing_extensions import deprecated` (in `manabot/backends/torch/rl.py`, `manabot/backends/mlx/rl.py`, and `managym/__init__.pyi`), which works on both 3.12 and 3.13 and sets `__deprecated__`. The contract therefore detects deprecation from: (1) `@deprecated(...)` decorators in `managym/__init__.pyi` AST, (2) `__deprecated__` on `manabot` callables/classes/methods/`property.fget` at runtime, and (3) `DeprecationWarning` emitted on `getattr(module, name)` for module `__getattr__` aliases (`ObservationEncoder`, `WORLD_VERSION`). In `manabot/env/observation.py`, `ObservationSpace.encode`, `ObservationSpace.encoder`, `ObservationSpace.allocate`, and `encode_observation` (all tagged for `0.3.0` removal; `encode_observation` is named in `ObservationSpace.encode`'s own deprecation message) carry `@deprecated(..., category=None)` so they are recorded as `deprecated: true` without altering runtime warning behavior or stacklevels on `0.2.x`.

**Where the tests run.** `tests/infra/` runs in `python_unit` (which builds the engine via `uv sync` and installs CPU Torch via `play-runtime`). A change under `manabot/*` selects only `integration` (`.github/ci/select-tests.sh`), and `manabot-integration.yml` lists test files explicitly. Both `tests/test_public_api_contract.py` (in the `model` group beside `tests/test_package_boundaries.py`) and `tests/infra/test_external_consumer.py` (in the `training` group, kept fast at ~2–5 s via `exact-quota` so `python_unit` stays quick too) join `manabot-integration.yml`.

## Design

### `__all__` on the fifteen published modules

Package `__init__.py` files stay empty. Each list holds only what consumers import, what the README promises, and the types those calls take or return. Speculative exports with no external user (`RuntimeConfig`, `SimulationConfig`, `collect_pair`, `load_metadata`, `MemoryStore`, `catalog_programs`) are left out — adding a name later is patch-compatible, withdrawing one is not.

| Module | Published `__all__` | Contract notes |
| --- | --- | --- |
| `manabot.infra.config` | `ObservationConfig`, `MatchConfig`, `ModelConfig`, `RewardConfig`, `UnsupportedPolicyInput`, `PolicyInputVersion`, `POLICY_INPUT_VERSION` | Pydantic config fields + exception/constants |
| `manabot.env.observation` | `ObservationSpace`, `column`, `columns`, `column_names`, `width`, `encode_observation`, `ObservationEncoder` | `encode_observation`, `ObservationEncoder`, and `ObservationSpace.{encode,encoder,allocate}` marked `deprecated: true` (0.3.0) |
| `manabot.env.match` | `Match`, `Reward` | Behavioral/dataclass surface |
| `manabot.env.memory_store` | `MemoryArchive`, `MemoryCapacities`, `Prefix`, `archive_memory`, `restore_memory` | `MemoryArchive`, `Prefix` recorded as `opaque` |
| `manabot.training.decks` | `UniformDecks` | Pydantic config fields + methods |
| `manabot.backends.torch.policy` | `ActorCritic`, `PolicyInference` | Public methods/properties (MRO restricted to `manabot`/`managym`) |
| `manabot.backends.torch.checkpoint` | `load_checkpoint`, `save_checkpoint` | Functions + `@overload` shapes |
| `manabot.backends.torch.bot` | `evaluate_observation` | Function |
| `manabot.backends.torch.arrays` | `on_device` | Function + `@overload` shapes |
| `manabot.backends.torch.collection` | `CollectionSession` | Behavioral class |
| `manabot.backends.torch.rollout` | `AdmittedRollout` | Includes inherited `AdmittedRows` methods |
| `manabot.policy.prepare` | `Prepared`, `reference_distribution` | `Prepared` recorded as `opaque` |
| `manabot.training.rollout.collection` | `open_collector`, `collect`, `close_collector`, `Collector`, `RolloutBatch`, `RecordedObservations`, `PackedActions` | `Collector` recorded as `opaque`; `PackedActions` re-exported |
| `manabot.training.rollout.actions` | `PackedActions`, `ActionRows` | Dataclass fields + methods |
| `manabot.training.state.game_errors` | `GameErrorPolicy`, `GameAttempt`, `GameErrorState`, `GameErrorLimit`, `GameOutcome`, `settle` | Pydantic fields + function/exception/alias |

The `*_INDEX` and `*_DIM` constants in `manabot.env.observation` stay internal; `column(array, name)` is the published lookup. Legacy unused `__all__` lists in internal modules (`manabot/belief/*.py` and `manabot/semantic/learning.py`; zero star-imports exist in the repo) are removed so `__all__` has one unambiguous meaning in `manabot/`. `manabot/belief/__init__.py` keeps its `__getattr__` lazy map for belief test helpers (without `__all__`), and `manabot/sim/search/flat.py` stays untouched as an explicit `0.3.0`-deprecated shim exempt from the unpublished-`__all__` check.

### The contract: `scripts/public_api.py` and `tests/public_api.json`

One script with two commands (`check`, `record`), following `scripts/pinned_pilot.py` and `scripts/engine_budget.py`:

- `check` extracts the live surface, compares it with `tests/public_api.json`, and exits non-zero on any drift with a classified diff (`ADDED`, `DEPRECATED_REMOVED`, `BREAKING`).
- `record` rewrites `tests/public_api.json` with strict patch-clean / minor-rollover rules derived from `pyproject.toml`'s `X.Y` minor line versus the file's `line`:
  - **Same minor line (`0.2` == `0.2`):**
    - `ADDED` only: plain `record` succeeds.
    - Any `BREAKING` or `DEPRECATED_REMOVED`: `check` prints the patch-clean `main` rule first (restore or keep a `@deprecated` shim targeting `0.3.0`). Plain `record` refuses. Only an explicit break-glass override `record --allow-break --reason "<why>"` writes the file and appends the reason and affected symbols to `breaks`.
  - **New minor line (`0.3` > `0.2`):**
    - `ADDED` and `DEPRECATED_REMOVED` (removal of symbols/members that were already `deprecated: true` on `0.2`): plain `record` succeeds, updates `line` to `"0.3"`, and clears prior `breaks`.
    - `BREAKING` (removal or breaking reshape of a **non-deprecated** symbol/member/field/param/enum value): still requires `--allow-break --reason "<why>"` and records it in the new line's `breaks`, so minor rollovers never silently drop live APIs.

The JSON holds `line` (`"0.2"`), `breaks`, and one entry per published symbol under each module key (`"managym"` and the fifteen `"manabot.*"` modules):

- **function**: ordered parameters (name, kind, required); `typing.overload` variants likewise; optional `deprecated: true`.
- **class**:
  - `fields` (Pydantic `model_fields`, dataclass fields, or `TypedDict` keys with `required: bool`),
  - `attributes` (stub `AnnAssign` attributes on PyO3 classes),
  - `properties` and `methods` (skipping leading `_` except `__init__`, MRO restricted to `manabot`/`managym` so Torch/Pydantic/Gymnasium internals stay out; each carries `deprecated: true` when decorated).
- **opaque**: `{"kind": "opaque"}` for the four opaque handles (`Prepared`, `Collector`, `Prefix`, `MemoryArchive`) declared in `OPAQUE_TYPES` in `scripts/public_api.py`.
- **enum**: `{"kind": "enum", "members": {name: int_value}}` for `IntEnum` classes in `managym/__init__.pyi` (changing an integer value or removing a member is `BREAKING`).
- **value / type_alias / reexport**: symbol name, kind, and optional `deprecated: true`.

Classification rules:
- **ADDED** (non-breaking): new module/symbol/member/attribute/enum member; new optional parameter or optional field; required parameter/field made optional; marking an existing symbol/member `deprecated: true`.
- **DEPRECATED_REMOVED**: removing a symbol or member that was `deprecated: true` in the baseline (patch-incompatible on `0.2.x`; expected on `0.2 -> 0.3` rollover).
- **BREAKING**: removing a non-deprecated symbol, member, field, attribute, parameter, or enum member; changing an enum member's integer value; changing a symbol's kind (e.g. class -> opaque or function -> value); changing a parameter's kind or positional order; adding a required parameter or required field; making an optional parameter/field required.

`tests/test_public_api_contract.py` runs `scripts/public_api.py check` in a subprocess and asserts:
1. Live surface matches `tests/public_api.json`.
2. Every name in every published module's `__all__` resolves and does not start with `_`.
3. No unpublished `manabot` module defines `__all__` (other than the explicit `0.3.0`-deprecated shim `manabot.sim.search.flat`).
4. The published module list in `manabot/README.md`'s "Library API" section matches `PUBLIC_MODULES`.
5. Every non-type-only top-level name in `managym/__init__.pyi` resolves on `managym` at runtime.

**Choices made for review**

- *Opaque handles (`Prepared`, `Collector`, `Prefix`, `MemoryArchive`).* Chosen over freezing all dataclass fields of every published class because `manabot` forbids leading underscores on internal cross-module fields (`Prepared` has 62 internal routing fields; `Collector` has 21 internal stream fields). Passed over: freezing all fields (blocks internal RL optimizations on `0.2.x`) and per-class member allowlists for all classes (creates a second source of truth).
- *Remove unused legacy `__all__` from internal `manabot/belief/*` and `manabot/semantic/learning.py`.* Chosen so `__all__` means one thing across `manabot/` (even inside an installed wheel without `scripts/`). Passed over: leaving 12 internal `__all__` lists in place, which makes `__all__` lie to consumers. Zero star-imports exist in the repository; `manabot/belief/__init__.py` keeps its `__getattr__` lazy map without `__all__`.
- *Strict `record` with `--allow-break --reason`.* Chosen over advertising `record --reason` as the normal fix on `0.2.x` and over silently accepting non-deprecated removals on minor rollover. Enforces the patch-clean `main` rule without requiring network access to fetch the previous PyPI wheel.
- *Deprecation detection via `@typing_extensions.deprecated` + module `__getattr__` warning.* Chosen over `getattr`-only warning capture, which misses methods, properties, and stub-decorated callables.
- *Shape, not types.* Parameter/field names, kinds, and required-ness without annotation text or defaults. Passed over: runtime annotation strings (differ across 3.12/3.13 and NumPy) and source annotations (every widening would trigger churn). Return-type regressions are caught by the consumer test.
- *Narrow initial `__all__` (drop speculative exports).* Dropped `RuntimeConfig`, `SimulationConfig`, `collect_pair`, `load_metadata`, `MemoryStore`, `catalog_programs` because no external consumer uses them; adding any later is patch-compatible.
- *`managym` submodules stay out of MTG-436.* jace imports `managym.recording`, `managym.enums`, `managym.policy`, and `managym.disclosure`; auditing `managym/*.py` to publish them belongs in a separate follow-up Task rather than growing MTG-436's scope.

### The external-consumer test: `tests/infra/test_external_consumer.py`

One test file that imports only published names and plays the part of an external RL library with its own training loop:

1. Build `ObservationSpace(ObservationConfig())` and a small `ActorCritic` (`hidden_dim=16`, `num_attention_heads=2`).
2. `open_collector(...)` with a `CollectionSession` on CPU, two streams, self-play (`opponent_mode="self"`, `learning_seats="one"`, `batch_closure="exact-quota"`), a `GameErrorPolicy` (with `recover_capacity=True`), and a `record_attempt` callback built on `settle`; `collect` 16 decisions; `close_collector`.
3. `AdmittedRollout(agent, batch)`: bootstrap values through `next_observations`, call `admit_rows`, then run one pass of `minibatches` computing a plain policy-gradient-plus-value loss from `agent.forward_distribution` and `AdmittedRollout.legal` with `torch.optim.SGD`. Assert parameters moved and loss is finite.
4. Save the generic model with the exact decks (`player_configs=match.to_rust()`); a deck-unrestricted checkpoint requires the semantic card model. `save_checkpoint` to `tmp_path`, `load_checkpoint` back, and assert both policies produce matching action probabilities on one observation through `inference_scope` and `evaluate_observation`.
5. `managym.catalog_tokens()` returns the keys the policy's card table was built from.

A second test in the file parses its own AST imports and asserts:
- Every imported `manabot.<module>.<name>` is present in `tests/public_api.json` under `manabot.<module>` **and** in `module.__all__` (so internal modules with or without `__all__` cannot slip through).
- Every imported `managym` symbol (and `managym.<attr>` accessed in the test) is present in `tests/public_api.json` under `"managym"`.

Pickling a batch across processes and merging worker batches belong to MTG-437 and are absent here. The file joins the `training` group of `manabot-integration.yml` (and stays fast, <10 s on CPU, for `python_unit`), and `tests/test_public_api_contract.py` joins the `model` group beside `tests/test_package_boundaries.py`.

### Documentation and CI wiring

- `manabot/README.md`: a "Library API" section listing the fifteen published modules (verified against `PUBLIC_MODULES` by `tests/test_public_api_contract.py`), what publication promises (`0.Y.x` patch stability, `@deprecated` shims targeting `0.(Y+1).0`), that everything else is internal, and linking `tests/public_api.json` and `scripts/public_api.py`.
- `AGENTS.md`: the published-versus-internal boundary beside the existing renaming rule, the contract check/record commands, and the patch-clean `main` rule.
- `docs/development.md` (`Releases` and `Validation by change area`): during a `0.Y.x` line, `main` stays patch-compatible with no un-shimmed removal of published APIs; next-minor `0.(Y+1).0` deprecation-removal PRs wait and merge together immediately before cutting `0.(Y+1).0`. Add `tests/test_public_api_contract.py` and `tests/infra/test_external_consumer.py` to validation guidance.
- `.github/ci/select-tests.sh` and `tests/infra/test_ci_selection.py`: `tests/test_public_api_contract.py`, `tests/public_api.json`, and `scripts/public_api.py` select `integration`; `tests/infra/test_external_consumer.py` selects `python_unit` and `integration`.

## Absent and error states

- A listed module without `__all__`, an `__all__` name that does not resolve, or an `__all__` name starting with `_`: `check` fails naming it.
- An unpublished `manabot` module defining `__all__` (outside the explicit `0.3.0`-shim exemption): `test_public_api_contract.py` fails naming it.
- `tests/public_api.json` missing or unparseable: `check` fails printing the record command; never treats a missing file as an empty contract.
- A non-type-only stub name absent from `managym` at runtime: `check` fails naming it.
- Torch not installed: `scripts/public_api.py` fails on import with the normal error (requires the same environment as integration tests).

## Exclusions

- `ObservationBatch` pickling, `RolloutBatch.concatenate`, `RecordedObservations.concatenate`, `ObservationSpace.max_actions` (MTG-437).
- Removing anything deprecated for 0.3.0 (MTG-435).
- `managym` Python submodules (`managym.enums`, `managym.disclosure`, `managym.policy`, `managym.recording`, etc.) — separate follow-up Task.
- MLX and other backends' modules, recipes, loops, and CLI: internal for now.

## What goes

Only unused internal `__all__` lists in `manabot/belief/*.py` and `manabot/semantic/learning.py` (no behavior or import paths change; `manabot/belief/__init__.py` keeps its `__getattr__` lazy map). Nothing deprecated is removed. `tests/test_package_boundaries.py` keeps all current checks.

## Coordination with MTG-437

MTG-437 adds public members (`concatenate`, `max_actions`) to published classes. Whichever of the two lands second runs `uv run python scripts/public_api.py record`; the diff is additions only and needs no `--allow-break`.

## Implementation status

1. Completed the fifteen export lists, contract recorder and compatibility tests in `d70731a562`. The same snapshot matched CPython 3.13 and an isolated CPython 3.12 installation. Both `check` and plain `record` refused a removed export, a renamed required parameter and a removed config field in the live checkout, preserving the snapshot.
2. Completed the external CPU learner and corrected the README example in `c205dfb579`. Eight decisions per stream produce sixteen training rows. A generic model needs the match's exact decks to save its checkpoint. The external test passed with both dependency groups (0.72 s test body with play-runtime); the README snippet ran through restored inference.
3. Completed CI selection and integration wiring in `c205dfb579`. Four path-selection cases passed. Pytest collected the full CI invocations containing the new modules (1,505 model-group tests and 384 training-group tests). Ruff checked and verified formatting of all 30 changed Python files.

The next step is gate, including the package-boundary suite and CI. MTG-437 owns batch pickling and concatenation.

## Done when

- `manabot/README.md` and module `__all__` lists agree on the exact published surface, and no unpublished `manabot` module carries a misleading `__all__`.
- Removing a published symbol, renaming a parameter, dropping a config field, or renumbering a `managym` enum fails CI with the patch-clean rule, and plain `record` refuses breaks on `0.2.x` while allowing `0.3.0`-deprecated removals on a `0.3` minor rollover.
- Opaque handles (`Prepared`, `Collector`, `Prefix`, `MemoryArchive`) do not freeze internal routing/stream fields.
- `tests/infra/test_external_consumer.py` imports only contract-verified published names, collects, runs one custom SGD step, round-trips a checkpoint, and checks `managym.catalog_tokens()`.
- `AGENTS.md` and `docs/development.md` state the patch-clean `main` release rule.

## Docs written

- `manabot/README.md` (`## Library API`)
- `docs/development.md` (`**Published library API.**`, `Selection`, and `### Patch-clean main`)
- `AGENTS.md` (published-vs-internal boundary, contract commands, and patch-clean `main` rule in `## Naming`)
