# MTG-344: object names for observation and Python object records

## Problem

MTG-320 (#491, `493127545a`) typed the engine runtime as `Object`/`ObjectId`/`CardId`
but left the published observation and Python names untouched. The observation
row built for every hand, library-top, graveyard, exile, stack and battlefield
object is still `CardData` / `managym.Card`, listed in
`Observation.agent_cards` / `opponent_cards`, although it holds tokens and
copies of spells (`observation.rs::populate_cards`, `add_permanent`,
`printed_card_data` sets `is_token: !object.is_card()`).

Jack Heart, 2026-11-12: "file a task to do these in a way that uses deprecation
infra -- alias, and then break in next change. ideally we shouldnt need to
change tensor layout, just the layer below that".

Task constraint: tensor layout, buffer shapes and keys, learning-token hashed
names, serialized configuration keys, checkpoint fields and frozen evidence keep
their spelling. "If any rename would force a layout, key or vocabulary change,
stop and ask."

## Outcome

A managym user reads `obs.agent_objects` / `obs.opponent_objects`, rows of
`managym.Object`, and `stack_object.source_registry_key`. Every old Python name
still works, emits a `DeprecationWarning` naming the replacement and removal in
0.3.0, and returns the same value. Observation JSON, encoded buffers, match-state
hashes and conformance evidence are byte-identical to `493127545a`.

## Demo

The slice-1 script (`uv run python -W always scratch/observation_digest.py --show-tokens`)
plays a seeded game until a token is visible, prints the row's class and the
token names from `obs.agent_objects`, then reads `obs.agent_cards` and
`managym.Card`. Expected: `Object [...]`, then two warnings of the form
`managym.Observation.agent_cards is deprecated and removed in 0.3.0; use
managym.Observation.agent_objects`.

## Rename table (decided)

CR 109.1 object vs CR 108 card, per `docs/engine.md` "Objects and zones".

| Published today | Holds | New name | Old name |
|---|---|---|---|
| native `CardData`, Py class `Card` | any object in a visible zone | `ObjectData`, `managym.Object` | `managym.Card` alias, warns |
| `Observation.agent_cards` / `opponent_cards` (native field and Py attr) | those rows | `agent_objects` / `opponent_objects` | Py get/set alias, warns |
| `StackObjectData.source_card_registry_key` | definition of a spell or ability source, token sources included | `source_registry_key` (matches `ContinuousEffectData.source_registry_key`) | Py get/set alias, warns |
| native helpers `populate_cards`, `push_card`, `card_data`, `printed_card_data`, `CardsJson` | same rows | `populate_objects`, `push_object`, `object_data`, `printed_object_data`, `ObjectsJson` | none (not published) |

Keep `card` (holds only cards, or is the rules term):

- `CardTypeData` / `CardTypes` / `.card_types`: "card type" is the CR 205.2 term for tokens too.
- `SideboardCardData` / `SideboardCard` / `agent_sideboard`: sideboards hold cards.
- `StackTargetData.card_id`, `StackTargetKind::Card`: only `Target::GraveyardCard`
  writes them (`observation.rs` ~L1704).
- `PlayerData.known_hand`, `*_counts`, `PlayerTurnHistory.cards_drawn`, `HandSnapshot.cards`,
  `scenario_force_card_in_hand`, `enable_card_capture`, `card_definitions_json`, deck/pool APIs.

## Frozen spellings (must not move)

- Observation JSON keys `agent_cards`, `opponent_cards`, `source_card_registry_key`
  (`observation_json.rs` writes them as string literals; the derived
  `Deserialize` with `deny_unknown_fields` needs `#[serde(rename = "...")]` on each
  renamed field). `managym.decision.Observation.viewer_state["opponent_cards"]`
  and Academy parity readers read those keys and stay.
- Buffer keys `agent_cards`, `opponent_cards`, `agent_cards_valid`,
  `opponent_cards_valid`, `semantic_cards`, `card_turn_state`; layout axes and
  column names; `ObservationSpace.max_cards_per_player`. The encoder buffer
  struct fields and the layout `card::` module mirror those keys and keep them,
  so a buffer has one name; only functions and parameters that take the records
  are renamed.
- `EventEntityKind::Card` / `EventEntityKindEnum.CARD`, `EventTypeEnum.CARD_MOVED`,
  `ActionEnum.SELECT_CARD`: integer values are tensor contents, the names appear in
  protocol fixtures, and `EventEntityKind::Object = 4` already exists. Out of scope.
- `managym.decision_capture.CardReference` (`kind: Literal["card"]`), Academy
  `experience.py` wire models (`LegacyCardView`, ...), `x-etude-*`: schemas.
- `agent/record_vocabulary.rs`: untouched.

## Alias mechanism

One message format, the one `managym/__init__.py::_deprecated_alias` already uses
(removed in 0.3.0; current release `v0.2.3`):
`managym.<Old> is deprecated and removed in 0.3.0; use managym.<New>`.

- **Class.** The native module exports `Object` only. `managym/__init__.py` gains
  a module `__getattr__` (PEP 562) over a small old->new table that warns
  (`stacklevel=2`) and returns the same class, so `isinstance` holds. The stub
  has `Card = Object  # Deprecated spelling, removed in 0.3.0.`, as
  `LearningCatalog` does.
- **Attributes.** PyO3 0.22.6 (`PyErr::warn_bound`, `pyo3-0.22.6/src/err/mod.rs:743`):
  explicit `#[getter]`/`#[setter]` pairs for `agent_cards`, `opponent_cards` and
  `source_card_registry_key` call one private `warn_deprecated(py, old, new)` in
  `bindings.rs`, then read or write the renamed field. Stub: `@property` +
  `@deprecated("use ...; removed in 0.3.0")`.
- No pytest `filterwarnings` is configured, so nothing fails on a stray old use
  by default; the acceptance run passes `-W error::DeprecationWarning`.

Rejected: Python wrapper subclasses over the native records (a second
representation, and vector-env observations would bypass it); keeping both field
names in the native structs (dual write).

## Text observation

`manabot/semantic/text_observation.py` renders the same rows under
"Visible cards (separate aliases for separate copies):" with
"card power/toughness" and "Bound public card definitions:". These become
"Visible objects ...", "object power/toughness" and "Bound public definitions:";
`C<n>` aliases stay (identifiers, and `O<n>` is taken by sideboard candidates).
The text changes, so `REPRESENTATION_VERSION` moves `manabot-text-6` ->
`manabot-text-7` (`manabot/README.md:459` describes it). Retained receipts keep
their recorded version and are not regenerated.
`tests/semantic/test_text_observation.py` keys its field map by `managym.Object`.

## Forbidden outcomes

- Any change to bytes of observation JSON, encoded buffers, state hashes,
  conformance or protocol fixtures; regenerating a fixture to pass.
- Old names used anywhere in repository code outside the alias definitions and
  their test. Search `scripts` and `experiments/*.py` as well as `manabot
  managym academy tests`; concept review 3 found `scripts/bench_states.py`
  only because the earlier searches skipped `scripts/`.
- A second record type or Python-side wrapper for the same row.
- An alias without a warning or without the removal release.

## Delete — do not maintain

Native `CardData`, `PyCard`, the object-list fields and their encoder / JSON
helpers were replaced in place; no second record representation remains. Only
the three Python warning aliases survive, until 0.3.0. Nothing is left to delete
in this Task.

## State (2026-11-13): implemented and validated; delivery remains

All rename slices, the three concept-review follow-ups and the acceptance checks
are done (history: `git log e363e4058e..HEAD`). Remaining: publish the PR, and
file the linked alias-removal Task when it lands.

Decisions taken during implementation, beyond the rename table:

- Bindings over `managym.Object` rows are `row`/`rows`/`objects` in Python
  (never the builtin `object`; `object_row` where `row` is taken) and
  `object`/`objects` in Rust (`entry` where `object` is already bound). Emitted
  keys, messages, text, `C<n>` labels, capacity labels (`"agent cards"`) and
  buffer keys are unchanged.
- Kept `card` where the value is a card where it is used:
  `manabot/sim/heuristic.py` `can_kick`, `_timely`, `_worth_casting` (hand
  cards); `heuristic.py`/`competency.py`/`action_cases.py` bindings filtered to
  hand or battlefield zones; `observation.rs::card_event_data`
  (`EventEntityKind::Card`); encoder `cards_len` / `card_turn_state_len` and
  `observability.rs` `left.agent_cards` (buffer fields); sideboard loops.
- Frozen-key readers stay: `academy/authored_match_parity.py`,
  `academy/public_commitment_parity.py` (`opponent_cards` JSON key), MLX
  `batch.agent_cards`. `manabot/eval/card_query.CardData` is unrelated.
- The Python text-observation field guard excludes the deprecated properties
  explicitly (AGENTS.md "Testing" records the rule).
- `tests/verify/test_observation_parity.py::test_observation_parity` hit the
  300-second per-test limit inside NumPy comparison, with no mismatch. Split
  into twenty ten-game batches (questions.md 9).

## What the runs showed

- Baseline at `493127545a`: `uv run --no-sync python scratch/observation_digest.py`
  (seeds 0–19, default `Match`, `random.Random(seed)` legal choices;
  length-prefixed JSON plus sorted scalar buffer name/shape/dtype/bytes) ->
  3,512 decisions, SHA-256
  `14c102e45341799283111bb57adec4b2df92b1742170b866a9f2334c0295100a`.
  Every later run, including after concept review 3 with a rebuilt extension,
  gave the same count and digest.
- `... observation_digest.py --show-tokens` under `-W always` ->
  `Object ['Joo Dee, One of Many']`, deprecated list length 3, class identity
  `True`, exactly two deprecation warnings.
- `docs/engine.md` example against the installed extension ->
  `Object 7 {'Hand'}`, `0`. Native `ZoneEnum` has no `.name`; the example uses
  its string form.
- Debug `cargo test --lib canonical_json_preserves_baseline_bytes_and_hashes`
  -> 1 passed (11 baseline cases).
- Debug `--test match_state_hash_tests --test semantic_conformance_tests --test canonical_replay_tests`
  (`CARGO_BUILD_JOBS=2`, `--test-threads 2`) -> 23 passed.
- Debug cards `--test cards observability` -> 66 passed.
- `semantic_conformance fuzz --root conformance/semantic-kernel-w17-v3 --seed 24301 --cases 32 --max-commands 512`
  -> 6,345 commands, no failures.
- `uv run --extra dev pytest tests/env/test_object_record_aliases.py tests/env/test_observation.py tests/env/test_named_observation_layout.py tests/semantic/test_text_observation.py -q -W error::DeprecationWarning`
  -> 130 passed, 1 failed: a capacity test still called the deleted private
  `_encode_cards`. Moved to `_encode_object_rows` / `Object list`;
  `-k test_object_capacity_rejects_omitted_objects` -> 2 passed. The other 130
  were not rerun.
- `tests/verify/test_observation_parity.py` plus the first-64-actions capacity
  test under `-W error::DeprecationWarning` -> 42 passed (1024.97s total, each
  case under 300s).
- `tests/sim/test_heuristic.py`, `tests/semantic/test_player_control.py` and
  five Academy/verify files (`test_learn_setup`, `test_authored_match_parity`,
  `test_authored_match_authority`, `test_competency`,
  `test_checkpoint_scenarios`) -> pass under the same flag.
- `uv run --no-sync python -W error::DeprecationWarning scripts/bench_states.py duplicate-tokens --sizes 1`
  -> pass at all five checkpoints, no warnings.
- `git diff e363e4058e --stat -- conformance protocol/fixtures academy/fixtures managym/tests/fixtures tests/fixtures managym/src/agent/record_vocabulary.rs`
  -> empty.
- Ruff across changed Python files reports seven I001 import-order findings
  (`academy/server.py`, `manabot/sim/heuristic.py` among them) that reproduce at
  the base; left unchanged. `cargo fmt --all --check`, `git diff --check` -> pass.
- Compress (final pass): no code change; this plan condensed. No rerun.

## Open for delivery and later Tasks

- **Alias-removal Task, filed at landing.** `v0.2.3` is the latest tag and
  already carries the `learning_*` aliases; these three first ship in the
  release after it. If that release is 0.3.0 itself, no published release
  warns. The removal Task should state that a 0.2.x release carrying these
  warnings precedes it (questions.md 2 holds the alternative, 0.4.0).
- Engine runtime `managym/state/src/stack_object.rs::StackObject::source_card_registry_key()`
  is outside this Task (serialized key frozen by `match_state_hash_tests.rs:250`).
  Its source may be a token: a candidate for the removal Task or an engine rename.
- `is_token` is true for copies of spells and the text observation prints
  "token yes". The column and JSON key are frozen; `docs/engine.md` states the
  meaning ("not a card").
- `RELEASE_NOTES.md` is still the v0.2.3 notes, untouched.
- Historical notes still spell the old names and are read-only here:
  `experiments/milestone-1-*.md`, `.lf/wave/coverage/MEMORY.md:681`.

## Done when

- The new names are what docs, stubs, examples and repository code use.
- Each old Python name warns with its replacement and 0.3.0, and still works.
- The seeded digest and every byte-pinned test match the base commit.
- A linked Task to remove the aliases in 0.3.0 is filed when this lands.
