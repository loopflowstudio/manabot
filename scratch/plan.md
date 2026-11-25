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

## Built

- Pass 1 (0605ac68f7) put turn steps in rules order, wrote public enum names to Observation JSON, and kept the 0.2.x capture reader and the frozen 12-step policy and decision-memory index. All 11 captures and both committed pilots passed.
- Pass 2 removed the native observation, policy, behavior-tracking and disclosure aliases across Rust, PyO3, stubs and Python wrappers. The CPython 3.13 extension built and 34 focused tests passed, including retained policy replay, keyword round trips and Temur mana encoding.
- Pass 3 removed the Python catalog, recording and Academy aliases. The release smoke validated two sdists and two wheels. The documentation from daffa5c394 was checked against the final APIs and the engine guide example ran successfully.

## Design

### Pass 3 removals

| Goes | Where | Callers to convert |
|---|---|---|
| `learning_catalog`, `learning_token_value`, `learning_record_names`, `LearningCatalog`, `_deprecated_alias`, module `__getattr__`/`WORLD_VERSION` | `managym/__init__.py`, `__init__.pyi` | `experiments/mana_taps.py:406` and `tests/eval/test_etude_agents.py:238` write the literal `"w17"`; alias cases in `tests/env/test_named_observation_layout.py` |
| `evidence()`, `FailureEvidence`, `read_failure`, module `__getattr__`/`CapturedVector` | `managym/recording.py` | `tests/env/test_game_record_writer.py`, `test_engine_recording.py`, `test_captured_vector_bookkeeping.py`, `experiments/vector_statistics_cost.py` |
| `RecordedGame.deal_seed`, `.leg`, `.game_trace_sha256`, `.decks`, `_warn_listing_field` | `managym/academy/launch.py` | `tests/academy/test_game_records.py`, `test_configured_launch.py` |
| `villain_type == "checkpoint"` | `managym/academy/server.py:680` | Academy test files send `"policy"` |
| `academy/__init__.py`; `"academy"` in `managym/pyproject.toml` `python-packages` and its vestigial `academy/` excludes; `"academy"` staging in `scripts/release/prepare.py`; the `academy/__init__.py` allowance in `scripts/release/validate.py:116` | — | shim case in `tests/test_package_boundaries.py:687` |

### What must survive

- The reported game in `managym/tests/fixtures/payment_search/manabot_slow_step.py` keeps its historical per-seat disclosure argument. Its SHA-256 matches the original report documented in `docs/development.md`.
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
- In a clean wheel installation, `import academy` raises `ModuleNotFoundError`. The source checkout contains `academy/ui`, so its directory is discoverable as a Python namespace package.

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
2. **Done — native removals.** The rebuilt extension exposes characteristics and object identity directly. The player-control test still checks single-grant wire data; that field belongs to the projection.
3. **Done (b4e54bfd19, 64152a8760)** — Python catalog, recording and launch aliases, the checkpoint opponent spelling and the top-level Academy shim. Both source archives and both local wheels passed release validation. An isolated CPython 3.13 wheel install rejected the old import.
4. **Done (4ddd16655d, 73f8ffb25b)** — Verified a separate parity receipt for the 174-command tape. Across all three surfaces, the two receipts differ only in source identity and the two viewer hashes at each checkpoint. The LLM text field inventory now lists the remaining native accessors. All four focused regression cases passed.

## Pass 3 verification

- The Python acceptance run covered 1,619 cases across two invocations. Before corrections, 1,568 passed, 43 failed and eight skipped. Five cases passed after their fixes: the checked parity receipt, the three LLM text field inventories, and a recorded-game test that had assumed an empty global session registry. The corrected cases were rerun individually, including the recorded-game case immediately after a session-creating test.
- The other 38 failed at the same assertions or errors using a CPython 3.13 extension built from the branch base (`45ddf3bca7`). These were isolated reruns of failing cases. Affected test modules and counts follow.

  - `tests/academy/test_advice.py` (2)
  - `tests/academy/test_decision_surface.py` (1)
  - `tests/academy/test_learn_evidence.py` (1)
  - `tests/academy/test_merchant_payment_recovery.py` (1)
  - `tests/academy/test_move_presentation.py` (1)
  - `tests/academy/test_payment_preview.py` (5)
  - `tests/academy/test_play_modes.py` (2)
  - `tests/academy/test_presentation.py` (3)
  - `tests/academy/test_public_commitment_parity.py` (3)
  - `tests/academy/test_shared_table.py` (1)
  - `tests/academy/test_shared_table_protocol.py` (1)
  - `tests/academy/test_stops.py` (2)
  - `tests/academy/test_table_projection.py` (1)
  - `tests/academy/test_ur_lessons_increment.py` (3)
  - `tests/belief/test_belief_agent.py` (1)
  - `tests/belief/test_learning.py` (1)
  - `tests/belief/test_player.py` (2)
  - `tests/belief/test_state.py` (1)
  - `tests/semantic/test_current_content.py` (1)
  - `tests/semantic/test_elemental.py` (1)
  - `tests/semantic/test_player_control.py` (2)
  - `tests/verify/test_seat_balance.py` (2)

- All 18 focused pass 3 checks passed. The engine-guide example and the demo passed, including the 0.2.x integer round trip. Both locally built source archives and wheels passed the release inventory checker. A fresh CPython 3.13 wheel install imported `managym.academy.launch` and rejected `import academy`.
- Next at gate/CI: debug nextest, Engine Budget, the Academy UI checks and the rest of the release platform/interpreter matrix. The broad Python run remains red on the 38 base failures above.

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
# Build the paired release artifacts as in .github/workflows/release.yml, then:
uv run --no-project python scripts/release/validate.py dist
git grep -n -i -E 'deprecat.*0\.3\.0|removed? in 0\.3\.0|removal in 0\.3\.0' -- managym academy docs AGENTS.md WORLDS.md
```

The warning scan may list dense transport and v9/v10 (MTG-433) and documentation for manabot-owned shims (MTG-435). Engine Budget still needs to confirm the pinned digests on the PR.

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
# From outside the checkout, with only the built engine wheel installed:
uv run --no-project --no-sync python -I -c 'import academy'  # ModuleNotFoundError
```

## Done when

- The demo prints exactly the commented values.
- A 0.2.x Observation capture with integer steps still loads (including
  `"step": 12` for first strike); a stored decision-memory checkpoint reads the
  exact same `0..=11` step cell as before (`FirstStrike`/`CombatDamage` = 7,
  `Cleanup` = 11).
- The only remaining engine removal notices belong to MTG-433. Documentation for live manabot shims remains under MTG-435.
- The built managym wheel and staged sdist pass `validate.py` with no `academy/`
  entry.
