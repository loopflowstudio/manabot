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
`tests/semantic/test_text_observation.py` (~L520) keys its field map by
`managym.Card`.

## Consumers to move to the new names

Counts of `agent_cards` matches at kickoff (buffer-key uses included):
`managym/src` 90, `managym/cards` tests 31, `managym/tests` 15, `manabot/env` 33,
`manabot/verify` 14, `manabot/backends` 13, `manabot/sim` 5, `tests/env` 34,
`tests/verify` 21, `tests/semantic` 17, `tests/academy` 6.

- Rust: `agent/observation.rs`, `observation_json.rs`, `observation_encoder.rs`,
  `policy_history.rs`, `vector_env.rs`, `python/bindings.rs`; the exhaustive guard
  `managym/cards/tests/cards/observability.rs` (field lists at ~L2833, L2899,
  L3256 fail to compile until updated); engine and card tests.
- Python: `manabot/env/observation.py` (`_encode_cards(cards: List[managym.Card])`),
  `manabot/env/match.py`, `manabot/sim/heuristic.py`, `manabot/semantic/*`,
  `manabot/verify/*`, `managym/experience_projection.py`, `academy/server.py`
  (L388-477), tests. `manabot/eval/card_query.CardData` is unrelated and stays.
- Docs: `docs/engine.md` (deprecated-names table at "Deprecated names"; L709 and
  L827 currently say the observation names keep `card`), `managym/README.md`
  (L76, L244), `docs/observation-and-model.md` L342 (buffer keys stay; describe
  the rows as objects), `managym/__init__.pyi` docstrings.

## Forbidden outcomes

- Any change to bytes of observation JSON, encoded buffers, state hashes,
  conformance or protocol fixtures; regenerating a fixture to pass.
- Old names used anywhere in repository code outside the alias definitions and
  their test.
- A second record type or Python-side wrapper for the same row.
- An alias without a warning or without the removal release.

## Slices

1. **This slice: baseline.** Build the extension at `493127545a` and record, in
   this file, one SHA-256 over `toJSON()` and every scalar-encoder buffer for
   each decision of 20 seeded games on the default decks (throwaway script in
   `scratch/`, not committed). Before number: recorded here.
2. Native rename with serde renames; Rust tests and guards compile and pass in debug.
3. PyO3 classes, alias getters/setters, module `__getattr__`, stub;
   `tests/env/test_object_record_aliases.py`.
4. Python consumers, Academy, tests.
5. Text wording and version; docs.

Slices 2-5 land as one PR; they are not independently shippable.

## Acceptance at gate

These are gate requirements; actual results are under Implementation progress.

- `cargo test --manifest-path managym/Cargo.toml --lib canonical_json_preserves_baseline_bytes_and_hashes` -> 1 passed (11 baseline cases, fixtures untouched in `git diff --stat`).
- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml --test match_state_hash_tests --test semantic_conformance_tests --test canonical_replay_tests -- --test-threads 2` -> pass (debug).
- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/cards/Cargo.toml --test cards observability -- --test-threads 2` -> pass.
- `cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- fuzz --root conformance/semantic-kernel-w17-v3 --seed 24301 --cases 32 --max-commands 512 --failure-dir target/semantic-conformance/failures` -> no failures.
- Rebuilt extension, then the slice-1 script -> the same SHA-256 as the baseline.
- `uv run --extra dev pytest tests/env/test_object_record_aliases.py tests/env/test_observation.py tests/env/test_named_observation_layout.py tests/semantic/test_text_observation.py -q -W error::DeprecationWarning` -> pass. The alias test asserts, for `managym.Card`, `Observation.agent_cards`, `Observation.opponent_cards` and `StackObject.source_card_registry_key`: one `DeprecationWarning` containing "0.3.0" and the new name, value equal to the new name's, setters write through; and that a seeded game through the Python encoder, the text observation and the Academy projection raises none.
- `git diff d1554835be --stat -- conformance protocol/fixtures academy/fixtures managym/tests/fixtures tests/fixtures managym/src/agent/record_vocabulary.rs` -> empty (the branch's main base; `493127545a..d1554835be` itself adds `managym/tests/fixtures/payment_search/manabot_slow_step.py`), except fixtures whose only content is the text observation at version 7, listed in the PR if any.

## Done when

- The new names are what docs, stubs, examples and repository code use.
- Each old Python name warns with its replacement and 0.3.0, and still works.
- The seeded digest and every byte-pinned test match the base commit.
- A linked Task to remove the aliases in 0.3.0 is filed when this lands.

## Risks checked

- **JSON keys would follow a field rename.** Writer uses literal keys; reader is
  derived, so serde renames are required. The baseline-bytes test covers both.
- **Exhaustive observability guard** names `CardData` fields; a missed field is
  a compile error, not a silent gap.
- **Vector env** shares `PyObservation`; no separate record class.
- **Hot paths**: alias getters warn per call; all repository callers move, so
  training pays nothing.

## Docs written (2026-11-13, before implementation)

Describe the intended behavior; slice 5 reads these rather than rewriting them.

- `docs/engine.md`: "Observations" (Python example over `agent_objects`, which
  names keep `card`), "Deprecated names" (three new rows, alias semantics,
  frozen JSON/array spellings), "Names" (L~745), "Where the engine departs".
- `managym/README.md`: two `CardData` -> `ObjectData` mentions.
- `docs/observation-and-model.md`: layout row for `agent_cards`/`opponent_cards`.
- `manabot/README.md`: `manabot-text-7` and the "Visible objects" heading.

Verified the example in `docs/engine.md` after rebuilding: `Object 7 {'Hand'}`
and `0`. Native `ZoneEnum` has no `.name`; the example now uses its string form.
Stub docstrings and aliases are implemented. `RELEASE_NOTES.md` remains the
v0.2.3 notes and is untouched.

## Delete — do not maintain

Replace native `CardData`, `PyCard`, object-list fields and their object encoder /
JSON helpers in place; retain no second record representation. Only the reviewed
Python warning aliases survive until 0.3.0. Preserve card-only records, frozen
JSON keys, buffer fields/keys, token vocabulary and evidence as listed above.

## Implementation progress

- Before (`493127545a`): `uv run --no-sync python scratch/observation_digest.py`
  -> 3,512 decisions, SHA-256
  `14c102e45341799283111bb57adec4b2df92b1742170b866a9f2334c0295100a`.
  Uses seeds 0–19, default `Match`, `random.Random(seed)` legal choices;
  length-prefixed JSON and sorted scalar buffer name/shape/dtype/bytes.
- Renamed native/PyO3 records and consumers in place; warning aliases and stubs
  implemented; text now `manabot-text-7`. No frozen fixture or vocabulary diff.
- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml --lib canonical_json_preserves_baseline_bytes_and_hashes -- --test-threads 2`
  -> 1 passed in debug (0.02s test time).
- `cargo fmt --manifest-path managym/Cargo.toml --all --check` -> pass.
- Focused Ruff on aliases, stubs, encoder, projection and text-field guard -> pass.
  Ruff across changed Python files reports seven I001 import-order findings;
  checking those same files at `HEAD` reproduces all seven. Left them unchanged.
- Release extension rebuilt and installed atomically from its matching wheel.
- `uv run --no-sync pytest tests/env/test_object_record_aliases.py tests/semantic/test_text_observation.py::test_every_observed_field_has_llm_text tests/semantic/test_text_observation.py::test_joo_dee_surveil_and_copy_play_through_the_text_decision tests/env/test_named_observation_layout.py::test_named_layout_matches_native_buffers_and_python_values -q -W error::DeprecationWarning`
  -> 21 passed (8.09s), including v9/v10 native/Python parity.
- After: the same 20-game script -> 3,512 decisions and the identical SHA-256
  `14c102e45341799283111bb57adec4b2df92b1742170b866a9f2334c0295100a`.
- `uv run --no-sync python -W always scratch/observation_digest.py --show-tokens`
  -> `Object ['Joo Dee, One of Many']`, deprecated list length `3`, class identity
  `True`, and exactly the intended two deprecation warnings.
- `CARGO_BUILD_JOBS=2 cargo check --manifest-path managym/Cargo.toml -p managym-cards --test cards`
  -> pass in debug, including the exhaustive observation guard.
- Rename slices are implemented. Remaining gate suites are listed above; publication and alias-removal Task
  creation at landing remain the caller's delivery steps.
- Compress (2026-11-13): `warn_deprecated(py, class, old, new)` states the class
  once; the Python row encoder's parameter is `row`, not a shadowed builtin
  `object`; README drops the version-6 comparison. Messages unchanged. Rebuilt
  extension, then the alias, text-field-guard and named-layout parity tests with
  `-W error::DeprecationWarning` -> 20 passed (3.99s); digest script -> 3,512
  decisions, same SHA-256 `14c102e4...100a`.
- Realign after syncing onto main `d1554835be` (adds `de39d601fb`, large-board
  benchmarks and payment-scan changes): the new `managym/examples/bench_states.rs`
  read native `Observation.agent_cards`/`opponent_cards`; moved to
  `agent_objects`/`opponent_objects` (its `Output::Observation.cards` label stays).
  `CARGO_BUILD_JOBS=2 cargo check --manifest-path managym/Cargo.toml --features benchmark --examples`
  -> pass. No other old-name use outside aliases, their test, serde renames,
  frozen JSON keys and fixtures.
- Left as is, outside this Task's observation/Python scope: engine runtime
  `managym/state/src/stack_object.rs::StackObject::source_card_registry_key()`
  (serialized key frozen by `match_state_hash_tests.rs:250`). Its source may be a
  token, so it is a candidate for the alias-removal Task or an engine rename.

## Concept review (2026-11-13) and follow-up

The published surface matches the rename table: one record (`ObjectData` /
`managym.Object`), three warning aliases, frozen JSON and buffer keys. No second
representation, and `manabot.managym` is the same module object, so the class
alias reaches it too. Nothing in the model changes.

The review found `managym.Object` / `ObjectData` rows still bound to card names.
Renamed to `row`/`rows`/`objects` in Python (never the builtin `object`) and
`object`/`objects` in Rust: `managym/experience_projection.py` (`_serialize_object`,
`_serialize_permanent`, `_serialize_player`, `_permanent_names`, which now zips
permanents with battlefield rows), `manabot/env/match.py` battlefield counters,
`manabot/semantic/text_observation.py` row loops, and
`managym/src/agent/observation_encoder.rs` row loops and test helper
`make_object`. Emitted dictionary keys, output text and `C<n>` labels unchanged.

Left as card, because the value is a card where it is used:
`manabot/sim/heuristic.py` `can_kick`, `_timely`, `_worth_casting` (hand cards
considered for casting); `observation.rs::card_event_data` (writes
`EventEntityKind::Card`); encoder `cards_len` / `card_turn_state_len` (buffer
lengths under frozen keys).

Checks at the merged HEAD (`d1554835be` base) with these renames:

- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` -> pass (8m29s); cp313 extension installed atomically. `cargo fmt ... --all`, `git diff --check` -> pass.
- The alias, text-field-guard, Joo Dee text-decision and named-layout tests with `-W error::DeprecationWarning` (command under Implementation progress) -> 21 passed (10.20s).
- `uv run --no-sync python scratch/observation_digest.py` -> 3,512 decisions, SHA-256 `14c102e4...100a`, identical to baseline.
- Frozen fixture/vocabulary diff against `d1554835be` -> empty.
- Compress: after the `_permanent_names` zip (Python only, no rebuild), the same tests plus `tests/semantic/test_player_control.py` -> 27 passed (8.16s).
- Realign at `b89559f62b` (merged main `e363e4058e`, Etude 0.3.2; no `managym/` or Rust change, so the extension built above still matches): `.lf/skills/qa.md` read `.agent_cards`; now `agent_objects`. Same 27 tests -> 27 passed (7.47s); digest -> 3,512, `14c102e4...100a`. Remaining old-name text is buffer fields/keys (`encoded.agent_cards`, MLX `batch.agent_cards`), the deprecated-names table, `manabot/eval/card_query.CardData` (unrelated), and historical notes (`experiments/milestone-1-*.md`, `.lf/wave/coverage/MEMORY.md:681`, read-only here).
- Remaining gate suites (Acceptance at gate) and the alias-removal Task stay with later delivery steps.

Constraint for the alias-removal Task filed at landing: `v0.2.3` is the latest
tag and already carries the `learning_*` aliases; these three first ship in the
release after it. If that release is 0.3.0 itself, removing them there publishes
no release that warns. The removal Task should state that a 0.2.x release
carrying these warnings precedes it (questions.md 2 holds the alternative,
naming 0.4.0).

Not a naming gap in this Task: `is_token` is true for copies of spells and the
text observation prints it as "token yes". The column and JSON key are frozen
and `docs/engine.md` states the meaning ("not a card").

## Concept review 2 (2026-11-13, HEAD `ce07967654`): binding follow-up

Published surface, aliases, stubs, docs and frozen spellings match the rename
table; nothing in the model changes and no product choice is open.

The first review's local-name pass was incomplete. These non-test bindings
still call `managym.Object` rows (whole `agent_objects` / `opponent_objects`
lists, tokens and spell copies included) cards. Rename the bindings only, to
`row`/`rows`/`objects` (never builtin `object`); emitted keys, messages, text
and buffers stay byte-identical:

- `manabot/env/observation.py`: `_object_rows` (`card.name for card in objects
  if card.is_token`), and the `cards`/`card` loop variables in `encode` at
  ~L386-436 that iterate the object lists. Keep `outside`/`card` at ~L446
  (sideboard cards), the capacity labels `"agent cards"` / `"opponent cards"`
  (they name the `agent_cards` buffer capacity `cards_per_player`).
- `manabot/verify/scenarios.py`: `_card_name` -> `_object_name` (its caller
  `action_casts` resolves a spell), loop variables there, and
  `cards` / `battlefield_cards` at ~L69-77.
- `manabot/verify/choice_corpus.py` ~L74-79, `manabot/semantic/learning.py`
  ~L560-566, `academy/server.py` ~L387-392 and ~L475-480: `cards`/`card` loop
  variables over the two object lists.
- `manabot/sim/heuristic.py` ~L507: `cards()` returns `dict[int, managym.Object]`
  over all visible rows -> `objects()`; callers that then pick hand cards keep
  `card` (already recorded above).

Tests: rename only where the bound row is known not to be a card
(`tests/verify/test_observation_parity.py:731` counts Food tokens;
`tests/verify/test_native_buffer_capacities.py:519`). Other test
comprehensions over hand/graveyard rows may keep `card`.

Not changing: Rust `observability.rs` `left.agent_cards`, `owner.agent_cards`
(encoded buffer fields), `listed_card_row` only if it is used solely for cards.

Python only, so no extension rebuild. Recheck with the 27-test command above
under `-W error::DeprecationWarning`, `tests/sim/test_heuristic.py`,
`tests/verify/test_observation_parity.py`, and the digest script (expect 3,512
decisions, `14c102e4...100a`).


Implementation follow-up: renamed the reviewed object-row bindings end to end,
including `_Board.objects` and its private cache and callers, `_object_name`,
and battlefield row consumers. Sideboard/casting-only names and all emitted
strings, schema roles, capacity labels and buffer keys stay unchanged. The
turn-state loop uses `object_row` to avoid rebinding the player-state array.
Ruff format and `git diff --check` pass; Ruff check reports only the existing
I001 import-order findings in `academy/server.py` and `manabot/sim/heuristic.py`,
reproduced from HEAD. Frozen fixture/vocabulary diff against `e363e4058e` is empty.


Validation of this follow-up (2026-11-13):

- The 27-case alias/text/layout/player-control selection plus
  `tests/sim/test_heuristic.py` completed 64 successful cases under
  `-W error::DeprecationWarning`; the combined run then hit the 300-second
  limit in `tests/verify/test_observation_parity.py::test_observation_parity`,
  inside NumPy comparison, with no mismatch reported. Split its 200 games
  into twenty ten-game seed batches (questions.md 9); the assertions and total
  game count are retained.
- `uv run --no-sync pytest tests/verify/test_observation_parity.py tests/verify/test_native_buffer_capacities.py::test_first_64_legal_actions_preserve_membership_mapping_and_masks -q -W error::DeprecationWarning`
  -> 42 passed (1024.97s total, each case within the unchanged 300-second limit).
- `uv run --no-sync python scratch/observation_digest.py` -> 3,512 decisions,
  SHA-256 `14c102e45341799283111bb57adec4b2df92b1742170b866a9f2334c0295100a`,
  identical to baseline.

The Concept review 2 implementation is complete. Remaining gate suites and
publication/alias-removal follow-up remain with the caller's later steps.

Compress after the binding follow-up (2026-11-13): Academy's two
battlefield-row/permanent pairings in `academy/server.py` use `zip` (same
truncation as the index guard), and `scenarios._object_name` scans both object
lists in one loop. No emitted value changes; Python only.
`uv run --no-sync pytest tests/academy/test_learn_setup.py tests/academy/test_authored_match_parity.py tests/academy/test_authored_match_authority.py tests/verify/test_competency.py tests/verify/test_checkpoint_scenarios.py -q -x -W error::DeprecationWarning`
-> 45 passed (87.31s). The digest script does not import either file; not rerun.

## Concept review 3 (2026-11-13, HEAD `7410da93bc`): one consumer on a deprecated name

Published surface, aliases, stub, docs and frozen spellings match the rename
table. Nothing in the model changes and no product choice is open.

Next implement direction:

- `scripts/bench_states.py:97` reads `obs.agent_cards` / `obs.opponent_cards`
  on a `managym.Observation` (arrived with main's `de39d601fb`; the realign
  moved the Rust example `managym/examples/bench_states.rs` but not this
  script). This is the "old names used in repository code" forbidden outcome
  and warns on every checkpoint. Move to `agent_objects` / `opponent_objects`;
  `max_cards_per_player` stays (frozen configuration key). The earlier searches
  covered `manabot managym academy tests`, not `scripts/`; search
  `scripts experiments/*.py` too when rechecking.
- Rust closures over the object lists still bind `card`: `managym/src/decision.rs:371`
  and `managym/src/decision_context.rs:128` (the latter matches any visible
  object by id) -> `object`. `agent/env.rs:1746,1750` test closures likewise.
  Binding names only; needs an extension rebuild only if you want the tests
  rerun, and no bytes move.
- `tests/env/test_object_record_aliases.py::test_object_list_alias_reads_and_writes_same_rows`
  takes `rows = obs.agent_objects` for both seats, so the opponent case writes
  the agent's rows into the opponent list. Use `getattr(obs, new)`.

Left as is: `academy/authored_match_parity.py`, `academy/public_commitment_parity.py`
read the frozen JSON key `opponent_cards`; `manabot/eval/cards.py` `CardData`
is `card_query`'s; `heuristic.py`/`competency.py`/`action_cases.py` `c`/`card`
bindings filter to hand or battlefield zones.

Recheck: `uv run --no-sync python -W error::DeprecationWarning scripts/bench_states.py`
on its smallest state selection (see its `--help`), the alias test under
`-W error::DeprecationWarning`, and the digest script.

### Concept review 3 implementation

Moved the benchmark script to the current observation attributes and renamed the
reviewed native object-row bindings. The public-entry closure uses `entry` to
avoid shadowing its enclosing `object`. The seat-parametrized alias test now
selects `getattr(obs, new)`, places a visible opponent permanent, and clears the
current list before writing it through the deprecated setter; both seats exercise
a nonempty write. No persistence or tensor names change.

Search of `scripts/` and `experiments/` found no remaining deprecated consumers
in Python files. Other matches are frozen protocol JSON and historical Markdown.
Ruff check/format on the two changed Python files and `git diff --check` pass;
workspace Rust formatting is applied. Frozen fixture/vocabulary diff against
`e363e4058e` remains empty.

Focused validation:

- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml --lib fixed_player_observation_does_not_follow_the_acting_seat -- --test-threads 2` -> 1 passed in debug (0.06s test time).
- Root maturin release build -> pass; matching cp313 extension installed atomically.
- `uv run --no-sync pytest tests/env/test_object_record_aliases.py -q -W error::DeprecationWarning` -> 5 passed (22.89s), including both nonempty seat lists.
- `uv run --no-sync python scratch/observation_digest.py` -> 3,512 decisions, SHA-256 `14c102e45341799283111bb57adec4b2df92b1742170b866a9f2334c0295100a`, identical to baseline.
- `CARGO_BUILD_JOBS=2 cargo build --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml` -> pass.
- `uv run --no-sync python -W error::DeprecationWarning scripts/bench_states.py duplicate-tokens --sizes 1` -> pass: encoder and text consumers at all five checkpoints, with no deprecation warnings.

Concept review 3 follow-up is complete.

Compress after concept review 3: the list-alias test states its warning pattern
once (`message`); the same hand-leak closure over `opponent_objects` in
`managym/src/flow/decision.rs` (test
`learn_reveal_is_ordered_public_definition_and_known_hand_is_not_a_hand_leak`)
binds `object`, matching `agent/env.rs`. Test-only Rust, no extension rebuild.
`uv run --no-sync pytest tests/env/test_object_record_aliases.py -q -W error::DeprecationWarning`
-> 5 passed (8.65s); `CARGO_BUILD_JOBS=2 cargo check --manifest-path managym/Cargo.toml --lib --tests`
-> pass in debug; `cargo fmt --all --check` -> pass. `scripts/` and
`experiments/` Python files hold no deprecated observation attribute.

Remaining gate suites and the alias-removal Task remain later delivery steps.
