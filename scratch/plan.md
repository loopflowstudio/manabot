# MTG-434: remove deprecated managym APIs, reorder first-strike step, drop the `academy` shim

One PR for 0.3.0. Every managym name that warns "removed in 0.3.0" and is not
dense observation transport or policy input v9/v10 goes; `StepKind` takes its
rules order; `Observation.toJSON()` writes step and phase names.

## User-visible outcome

- A managym caller on a removed name gets `AttributeError` / `TypeError` /
  `ModuleNotFoundError` instead of a `DeprecationWarning`. The replacements
  already exist and are what every in-repo caller uses after this change.
- `managym.StepEnum.COMBAT_FIRST_STRIKE_DAMAGE == 7`; `COMBAT_DAMAGE` through
  `ENDING_CLEANUP` are 8–12. Step integers now sort in turn order.
- `Observation.toJSON()` writes `"turn": {"phase": "COMBAT", "step":
  "COMBAT_FIRST_STRIKE_DAMAGE", ...}`. `Observation.from_json` and
  `validate_observation_json` still read captures that hold the old integers.
- The managym wheel and staged sdist have no top-level `academy/` package or
  files.

## Built (pass 1, 0605ac68f7; docs daffa5c394)

- `StepKind` in rules order (`FirstStrikeCombatDamage = 7` … `Cleanup = 12`,
  `PartialOrd, Ord`); `PhaseKind/StepKind::name/from_name` own the public enum
  spellings; `from_legacy_code` decodes 0.2.x integers (`12` = first strike)
  only in `observation_json.rs` and the baseline fixture test.
- `parse_observation` normalizes integer `turn.phase/step` before its canonical
  comparison; the 11 frozen captures keep their bytes and digests and differ
  from the writer only in those two fields.
- Decision memory shares the encoder's frozen 12-step index (`0..=11`, both
  damage steps 7, Cleanup 11). A live double-strike game and both committed
  pilots confirmed it; `viewer_state_hash` values change (live-to-live only).
- Docs for the whole Task (list below) were committed in daffa5c394, ahead of
  passes 2–3: they already describe removals the code has not made yet, so
  re-read each named doc against the code at the end of pass 3.

## Findings that shape the remaining removals

1. **The singular `cast_permission` is two things.** The Python getter/setter
   on `Card` (`bindings.rs:1998-2006`, marked 0.4.0, explicitly approved for
   0.3.0) is the deprecated API. The JSON key `cast_permission` for a single
   grant is the frozen schema-8 card wire (`observation_json.rs:194`), mirrored
   by the experience projection (`experience_projection.py:145`), the protocol
   schemas and the UI type. The key stays; only the Python accessor goes.
2. **`ObservationCharacteristics` is intentionally read-only.** One test
   (`tests/verify/test_native_observation_contract.py:345-359`) mutates
   `card.keywords = ...` and `permanent.keywords = ...` through the deprecated
   flat setters. Removing those setters requires rewriting that test via JSON
   round-trip (`json.loads(obs.toJSON())` → mutate → `Observation.from_json`),
   not adding setters to `ObservationCharacteristics`.
3. **`RecordedGame` has four 0.3.0 properties, not three.** `deal_seed`
   (`managym/academy/launch.py:51`) calls `_warn_listing_field` alongside `leg`,
   `game_trace_sha256`, and `decks` (and is documented in `docs/academy.md:351`
   and tested in `tests/academy/test_game_records.py:239`). All four go
   together; `_record` stays because `launch.py:447` passes it to `import_game`.
4. **Release staging also references `academy/`.** `scripts/release/prepare.py`
   still includes `"academy"` in the engine staging tree (copying
   `academy/README.md` into the staged engine sdist) and `managym/pyproject.toml`
   keeps `academy/ui/**` and `academy/README.md` excludes for that shim. Both
   must be cleaned alongside deleting `academy/__init__.py` and tightening
   `scripts/release/validate.py`.
5. **No pytest warning filter exists**, so deprecated use is silent today.
    After removal each use is a hard error, which is the detector: running the
    consumers' tests finds every straggler. Grep found flat-accessor use only
    in tests (9 files) — library code already reads `.characteristics`.
6. `git grep -E '\b…'` matches nothing on this machine; use `-w` or explicit
    character classes when sweeping for callers.


## Design

### Removals

Native (Rust, PyO3, stub) — each with its `.pyi` lines:

| Goes | Where | Callers to convert |
|---|---|---|
| Flat characteristic getters/setters on `Card` and `Permanent` (`name`, `text_box`, `power`, `toughness`, `is_legendary`, `is_ally`, `is_lesson`, `ward_cost`, `kicker_cost`, `card_types`, `keywords`, `mana_cost`), `Permanent.card`, `StackObject.card`, `warn_observation_alias` | `python/bindings.rs` | 9 test files; `tests/env/test_observation_characteristics.py` keeps its non-alias cases; `tests/verify/test_native_observation_contract.py:345-359` mutates via JSON round-trip (`Observation.from_json`) |
| `Observation::card_records` | `agent/observation.rs:2584` | none |
| `ExileCastGrant` | `flow/cast_permission.rs:147` | none (one `.lf/wave` research note mentions it) |
| `Card.cast_permission` getter/setter | `bindings.rs:1998` | `tests/semantic/test_player_control.py:279` |
| `opponent_policy` on `VectorEnv`, `RecordedVectorEnv`, `resolve_player_policies`, and the pass-through on `manabot.env.vector_env.VectorEnv` | `vector_env_bindings.rs:241,1415-1432`, `recording.py:1221`, `manabot/env/vector_env.py:30` | `tests/env/test_native_player_policies.py`, `test_captured_vector_bookkeeping.py` |
| `enable_behavior_tracking` on `managym.Env`, Rust `Env::new`, and `manabot.env.Env` | `bindings.rs:3985`, `agent/env.rs:241`, `manabot/env/env.py:60` | six callers pass `False` |
| `PlayerConfig.disclosure` argument, getter, setter, `legacy_disclosure`, the hint/conflict branch of `game_setup` | `bindings.rs:994-1076,5640-5680`, `deck_bindings.rs:166` | `tests/env/test_decklist_disclosure.py`, `managym/tests/fixtures/payment_search/manabot_slow_step.py` |

`Card`'s flat accessors are the same alias family as `Permanent`'s (one warn
helper, one `.pyi` note) and go together. *Choice made for review*; the brief
names only permanents and spells. `Permanent.name` stays: it is the
effect-set name override (not a deprecated alias; renaming it to `name_override`
would break schema-8 wire JSON and is a separate outcome).

Python & packaging:

| Goes | Where | Callers to convert |
|---|---|---|
| `learning_catalog`, `learning_token_value`, `learning_record_names`, `LearningCatalog`, `_deprecated_alias`, module `__getattr__`/`WORLD_VERSION` | `managym/__init__.py`, `__init__.pyi` | `experiments/mana_taps.py:406` and `tests/eval/test_etude_agents.py:238` write the literal `"w17"`; alias cases in `tests/env/test_named_observation_layout.py` |
| `evidence()`, `FailureEvidence`, `read_failure`, module `__getattr__`/`CapturedVector` | `managym/recording.py` | `tests/env/test_game_record_writer.py`, `test_engine_recording.py`, `test_captured_vector_bookkeeping.py`, `experiments/vector_statistics_cost.py` |
| `RecordedGame.deal_seed`, `.leg`, `.game_trace_sha256`, `.decks`, `_warn_listing_field` | `managym/academy/launch.py` | `tests/academy/test_game_records.py`, `test_configured_launch.py` |
| `villain_type == "checkpoint"` | `managym/academy/server.py:680` | Academy test files send `"policy"` |
| `academy/__init__.py`; `"academy"` in `managym/pyproject.toml` `python-packages` and its vestigial `academy/` excludes; `"academy"` staging in `scripts/release/prepare.py`; the `academy/__init__.py` allowance in `scripts/release/validate.py:116` | — | shim case in `tests/test_package_boundaries.py:687` |

With the last flat alias gone, `tests/env/test_observation_characteristics.py`
loses its alias cases and keeps the projection ones.

### What must survive

- **Retained evidence reads as before.** `GameRecord.opponent_policy` and
  `resolved_player_policies` (`recording.py:287,337`) still decode old
  receipts; the JSON key `cast_permission`; `Observation.from_json` and
  `validate_observation_json` on integer-step captures; `tests/fixtures/*.json`
  and `managym/tests/fixtures/observation_json/*.json` bytes are not rewritten.
- **Stored checkpoints compute what they computed.**
  `tests/model/test_pinned_pilot.py` passes without a new record; if it moves,
  the decision-memory cell or a tensor column moved and the change is wrong.
- Everything owned by sibling Tasks stays and keeps its 0.3.0 tag:
  dense transport and v9/v10 (`set_buffers`, `encode_observation*`,
  `axis_extents`, `object_references`, `legacy_card_lists`, the Rust
  `legacy_card` helpers, `InputLayout::Legacy`) — MTG-433; `manabot.*` shims
  including `manabot.env.failures.CapturedVector` — MTG-435.
- `manabot.model.world.current_binding_form`, `SUITE_PREFIX = "w17"` and
  optional `world` fields in old files are untouched.
- `RecordedGame._record` stays (`managym/academy/launch.py:447` passes it to
  `import_game`).

### Absent and error states

- Integer `turn.step` outside 0–12, or an unknown name (including Rust PascalCase
  like `"FirstStrikeCombatDamage"`): `from_json` raises the existing
  `unsupported enum value` error. A float (`1.0`), boolean, or null: serde type
  error.
- `PlayerConfig(..., disclosure=...)`, `VectorEnv(opponent_policy=...)`,
  `Env(enable_behavior_tracking=...)`: Python `TypeError` for an unexpected
  argument. No custom message; the release notes carry the migration.
- `villain_type: "checkpoint"`: the existing "villain_type must be one of"
  `ValueError`.
- `import academy`: `ModuleNotFoundError`.

## Docs

Updated in place for `write-docs`:

- `docs/engine.md` — added "Turn steps and Observation JSON" with rules-order `StepEnum` values, enum-name JSON serialization and 0.2.x integer decoding example; removed the `opponent_policy` deprecation, the "Deprecated names" table and the `WORLD_VERSION` alias note.
- `docs/engine-python.md` — removed the `FailureEvidence` / `read_failure` alias note.
- `docs/academy.md` — removed the `RecordedGame` legacy property note (`deal_seed`, `leg`, `decks`, `game_trace_sha256`).
- `docs/observation-and-model.md` — replaced the `.card` / flat-characteristic deprecation note with direct `.characteristics` guidance.
- `managym/README.md` — added rules-order `StepKind`/`StepEnum` discriminants and Observation JSON step/phase name serialization with link to `docs/engine.md#turn-steps-and-observation-json`.
- `academy/README.md` — replaced the top-level `academy` deprecation notice with `managym.academy`.
- `WORLDS.md` — removed the `managym.WORLD_VERSION` alias sentence.
- `AGENTS.md` — removed the `WORLD_VERSION` alias sentence, the `.card`/flat-getter "removed in 0.3.0" clause, and the out-of-order integer-enum rollover example while keeping the wire/checkpoint compatibility rule.
- `manabot/README.md` — removed the `PlayerConfig.disclosure` alias sentence.
- `protocol/README.md` — updated `from academy.advice import advice_schema` to `from managym.academy.advice import advice_schema`.

`RELEASE_NOTES.md` and `release/` are release output and are not edited.

## Exclusions

- No version bump; 0.3.0 is cut by the release Graph.
- No change to `PHASE_STEPS`, combat rules, the policy tensor, decision-memory
  `0..=11` step encoding, or protocol schema shapes.
- No renaming of `Permanent.name` (name override) — separate outcome.
- No pytest `-W error` policy for deprecations (worth its own Task; noted in
  `scratch/questions.md`).

## Passes

1. **Done (0605ac68f7)** — turn order, JSON names, decision memory. Demo
   printed `PRECOMBAT_MAIN PRECOMBAT_MAIN_STEP` and `7 8`; optional u1952
   checkpoint was absent, both committed pilots replayed.
2. **Next — native removals.** The first table, stub lines with it, one extension
   rebuild, then every converted caller's tests (including
   `test_native_observation_contract.py` rewritten via JSON round-trip).
3. **Python removals, shim, packaging, docs.** The second table (including
   `RecordedGame.deal_seed`, `prepare.py`, `pyproject.toml`, `validate.py`),
   docs. Ends with the acceptance check.

## Acceptance check at gate

```bash
CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml --workspace --profile ci --test-threads 2
uv run --extra dev --extra artifacts --extra reports pytest \
  tests/env tests/academy tests/semantic tests/verify tests/belief \
  tests/model/test_pinned_pilot.py tests/model/test_decision_memory.py \
  tests/test_package_boundaries.py \
  tests/semantic/test_elemental_world.py tests/eval/test_evaluation_models.py \
  tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py \
  tests/model/test_choice_separability.py -q
(cd academy/ui && npm run check && npm test)
uv run python scripts/release/prepare.py   # then scripts/release/validate.py on its output
git grep -n -i -E 'deprecat.*0\.3\.0|removed? in 0\.3\.0|removal in 0\.3\.0' -- managym academy docs AGENTS.md WORLDS.md
```

All green; the last command lists only dense-transport and v9/v10 lines
(MTG-433). Engine Budget passes on the PR with unchanged digests.

## Demo

```bash
uv run python - <<'PY'
import json, managym
from manabot.verify.text_observation import priority_root
env = priority_root((), ("Gray Ogre",))
obs = env.observation_for_player(env.current_agent_index())
wire = json.loads(obs.toJSON())
print(wire["turn"]["phase"], wire["turn"]["step"])          # PRECOMBAT_MAIN PRECOMBAT_MAIN_STEP
wire["turn"]["phase"], wire["turn"]["step"] = 1, 3           # a 0.2.x capture
print(managym.Observation.from_json(json.dumps(wire)).toJSON() == obs.toJSON())  # True
print(int(managym.StepEnum.COMBAT_FIRST_STRIKE_DAMAGE), int(managym.StepEnum.COMBAT_DAMAGE))  # 7 8
for name in ("learning_catalog", "WORLD_VERSION"):
    print(name, hasattr(managym, name))                      # False False
PY
uv run python -c 'import academy'                            # ModuleNotFoundError
```

## Done when

- The demo prints exactly the commented values.
- A 0.2.x Observation capture with integer steps still loads (including
  `"step": 12` for first strike); a stored decision-memory checkpoint reads the
  exact same `0..=11` step cell as before (`FirstStrike`/`CombatDamage` = 7,
  `Cleanup` = 11).
- No managym, Academy or docs line promises a 0.3.0 removal except the
  dense-transport and v9/v10 ones owned by MTG-433.
- The built managym wheel and staged sdist pass `validate.py` with no `academy/`
  entry.
