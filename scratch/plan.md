# MTG-504: separate record names, runtime program tokens and decision-memory names

## Outcome

`managym.record_vocabulary()` lists only names an encoder really hashes into an
id column. Runtime-only program tokens come from their own list,
`managym.runtime_tokens()`. Decision-memory names belong to decision memory and
carry no program token. No code selects rows by a `DecisionMemory::` or
`execution/` string prefix. Checkpoints written before the change load with the
same learned vectors and compute the same outputs.

This is a 0.3.0 change: the content of a published function changes, with no
0.2.x shim. **The PR is published and held unmerged until the 0.3.0 cut**, the
same way PR #612 is held ("Hold unmerged until ready to cut `0.3.0` so `main`
stays patch-compatible for `0.2.x`"). Delivery for this Task ends at a
published, green, held PR; it is not landed and auto-merge is not enabled.

## Base

The branch starts at `main` 6a131c4dc9, which lacks PR #636 (MTG-501, open, in
its deliver step when this plan was written). #636 adds `PROGRAM_VOCABULARY`,
`missing_program_tokens`, the debug emission guard, more `execution_names`
registrations, the docs section "Program-token vocabulary", and three of the
prefix filters this Task removes. Build on it: `lf sync main` if #636 has
merged, otherwise `lf sync MTG-501`. Do not build before one of those succeeds.
It is not stacked on #612; the two meet at the 0.3.0 cut.

## Findings that shape the design

- `record_names()` (`managym/src/agent/record_vocabulary.rs`) appends
  `execution_names`: one `("token", "execution/<kind>/<name>", token)` row per
  runtime token. Nothing in `managym/` hashes an `execution/...` string into an
  observation (`git grep 'execution/' managym/src managym/cards` finds only the
  generator). The rows exist so `_record_tables` in `manabot/policy/catalog.py`
  appends the token.
- In `_record_tables` a name with a token reads that token's embedding row. So
  an `execution/*` name key and its token key address one row, and so do a
  `DecisionMemory::*` name key and its phantom `Variant` token key. Dropping
  one key of each pair changes no row count and no parameter shape; only the
  order of appended rows moves.
- `manabot/policy/binding.py::bind_catalog` binds stored rows by key and
  refuses an old key that is neither in the running tables nor classified in
  `managym/src/cards/token_renames.rs` ("unexplained … key"). After the change
  every stored checkpoint has `execution/*` name keys with no running
  counterpart, and memory checkpoints have phantom token keys. Both must
  retire; the learned vector stays reachable through the surviving key.
- Literal `RECORDS` entries with a successor (`execution/6/kept creature` →
  `execution/6/kept permanent`) would fail with "missing successor" once the
  successor name is gone. Every literal `execution/*` entry in `RECORDS` has
  its token in `TOKENS` (`combat`, `kept creature`, `creatures each player
  sacrifices`, the three `ReplacementEffect` pairs, the four retired texts), so
  one rule covers them: the legacy name of every runtime token, current or
  historical.
- Decision memory reads its names as record names
  (`manabot/policy/decision_memory.py`: `tables.names(1, field_values)`,
  admission through `managym.decision_memory.decision_name_values()`), never
  as program tokens. Default policies must not grow by the 44 rows (agent
  guide: "Optional model capabilities select their catalog rows explicitly").
- Consumers of the changed rows: `manabot/policy/catalog.py`,
  `managym/decision_memory.py`, `managym/src/python/bindings.rs`,
  `scripts/pilot_names.rs` (built against a writer's engine by
  `scripts/pinned_pilot.py`), `managym/cards/tests/cards/observability.rs`
  (`assert_execution_tokens_registered` and the effect witness, which chain
  `record_names()` tokens), `tests/model/test_record_coverage.py` (drops the
  `execution/6/effect execution` row), `tests/model/test_effect_vocabulary.py`,
  `tests/semantic/test_pooled_mana_classes_input.py`, docs
  (`docs/engine.md#program-token-vocabulary`, `manabot/README.md`,
  `managym/cards/tests/cards/README.md`, `AGENTS.md`, `.lf/qa/engine.md`).

## Design

### Native: three lists, three owners

`managym/src/agent/record_vocabulary.rs` stays the one registration site, with
two entry points:

- `record_names() -> Vec<RecordName>`: names the fact and shared-effect
  encoders hash. No `execution/*` rows, no decision-memory names.
- `runtime_tokens() -> Vec<RuntimeToken>` (`kind: TokenKind`, `name: String`):
  the program tokens runtime projections emit that no printed span supplies.
  This is today's `execution_names` body returning tokens, sorted and
  de-duplicated, without the "skip if a record row already has it" check.

`managym/src/agent/decision_memory.rs` owns its names: a function beside
`RECORD_NAMES` returns each `DecisionMemory::<name>` as a
`RecordName { namespace: Token, token: None }`.

`PROGRAM_VOCABULARY` (from #636) becomes printed catalog ∪ `record_names()`
tokens ∪ `runtime_tokens()`, with no filter. The uniqueness test checks
`record_names()` and the memory names together, since they share the `token`
namespace.

### Python surface

- `managym.record_vocabulary()`: same signature, real record names only.
- `managym.runtime_tokens() -> list[tuple[int, str, int]]`: `(kind, name,
  value)`. New published function; `managym/__init__.pyi` and
  `tests/public_api.json` gain it (`scripts/public_api.py record`, an
  addition).
- `managym.decision_memory.decision_names() -> tuple[tuple[str, int], ...]`
  from a native read of the memory-owned list; `decision_name_values()` derives
  from it.
- `manabot/policy/catalog.py::_record_tables`: record names alias or append as
  now; then each runtime token not already present appends a row; then, only
  when `decision_memory` is on, each memory name appends a name row. No fake
  rows, no prefix test.

### `catalog_tokens()` naming (reviewed, kept)

`catalog_tokens()` keeps its name: it is the per-definition token catalog,
rows by `CardDefId`, and `CatalogTokens`, `catalog_token_value` and
`catalog_token_renames` share the word. It was renamed once already in 0.2.3.
The confusion in issue 629 was the missing second surface, which
`runtime_tokens()` now is. Its docstring and `docs/engine.md` state that the
admitted program vocabulary is printed pairs ∪ record tokens ∪ runtime tokens,
and a Python test holds a fresh policy's `record_token_keys` equal to exactly
that union with decision memory on and off.

### Legacy keys

`managym/src/cards/token_renames.rs` gains functions that the
`catalog_token_renames()` binding returns in place of the bare constants:

- records: literal `RECORDS` (now only `Modification::ControlChanged`) plus
  `("token", "execution/<kind>/<name>", None)` for every runtime token and
  every `TOKENS` entry's old name. The literal `execution/*` entries go.
- tokens: literal `TOKENS` plus `(Variant, "DecisionMemory::<name>", None)`
  for each memory name.

`binding.py` needs no new rule: its existing retire path handles both. The
superset (legacy names for tokens that never had a row) is harmless, because a
retirement only applies to a key a stored checkpoint actually has. The agent
guide's rule shortens to: removing a runtime token retires its token in
`TOKENS`; its legacy record name follows.

## Decisions made for review

Nobody was present; these were chosen and can be reversed.

1. **Hold for 0.3.0 with no 0.2.x shim.** Passed over: landing now with
   `record_vocabulary()` kept as a deprecated composite and the clean lists
   under new names. That is a second rename of a function renamed in 0.2.3 and
   a dual state until the cut.
2. **Decision-memory names leave `record_vocabulary()`** and live on decision
   memory's own surface. The brief says "Change `DecisionMemory::*` entries to
   `record(Token, …, None)` and remove the repeated `DecisionMemory::` prefix
   filters"; keeping the rows in the shared table would still need a prefix
   test in `catalog.py` and `decision_name_values`. Passed over: rows stay in
   `record_vocabulary()` with `token=None`, two prefix selections remain.
3. **`catalog_tokens()` is not renamed**; it is paired with
   `runtime_tokens()`. Passed over: `printed_tokens()` with 0.3.0 → 0.4.0
   shims across about 45 call sites.
4. **Legacy keys are generated**, not a frozen list of several hundred
   strings (card names included). Passed over: a literal list frozen at the
   change.

## Acceptance check

With the rebuilt extension installed:

```sh
uv run python -c "import managym; v=managym.record_vocabulary(); print(sum(n.startswith(('execution/','DecisionMemory::')) for _,n,*_ in v), len(managym.runtime_tokens()) > 0)"
# 0 True
git grep -nE 'starts_?with\("(DecisionMemory::|execution/)' -- managym manabot tests docs scripts
# no output
uv run --extra dev pytest tests/model/test_pinned_pilot.py tests/model/test_checkpoint_binding.py tests/model/test_effect_vocabulary.py tests/model/test_record_vocabulary.py tests/model/test_record_coverage.py tests/model/test_decision_memory.py -q
```

The pinned v9 and v11 pilots pass unchanged, with no
`scripts/pinned_pilot.py record`: the stored game's outputs are identical and
the binding report shows the `execution/*` names as retired keys. Needing
`record --reason` means the design is wrong, not that the pin should move.

## Absent and error states

- A stored key that is neither running nor classified still raises
  "unexplained … key"; the generated retirements do not widen that to
  arbitrary keys.
- A runtime projection emitting an unregistered token still panics in debug
  (#636 guard) and fails Python admission with the `(kind, 0x…)` pairs.
- A memory name reaching a policy built without decision memory fails
  admission as before; the phantom-token variant of that test goes, since the
  token no longer exists.
- A checkpoint written after the change is not promised to load on 0.2.x.

## Exclusions

- No change to observations, encoders, token kinds, hashes, policy input
  version or embedding row count.
- No rename of `record_vocabulary`, `catalog_tokens` or
  `ModelConfig.record_vocabulary`.
- Frozen evidence and the pinned pilot fixtures are not regenerated.
- Nothing from PR #612; `release/v0.3.0/NOTES.md` lives there. The release
  line goes in this PR's body: "`record_vocabulary()` lists hashed record names
  only; runtime-only program tokens are `runtime_tokens()`; decision-memory
  names are `managym.decision_memory.decision_names()`."

## What goes

- `execution_names` and the `execution/<kind>/<name>` generator in
  `record_vocabulary.rs`; the `variant(...)` registration of memory names.
- The literal `execution/*` entries of `token_renames::RECORDS`.
- Every `DecisionMemory::` prefix selection: `PROGRAM_VOCABULARY`,
  `catalog.py`, `managym/decision_memory.py`, `docs/engine.md`,
  `tests/model/test_effect_vocabulary.py` (with its `memory_pair` fixture and
  the disabled-memory-token cases).
- The `execution/6/effect execution` parameter of
  `test_a_name_missing_from_the_vocabulary_fails_the_sweep`, replaced by
  removing that token from `runtime_tokens()`.

Must survive: every learned embedding row of a stored checkpoint, the debug
emission guard, exact `(kind, 0x…)` diagnostics, the 44 memory names' hashes.

## Passes

1. **Checkpoints survive the split (could prove the plan wrong).** Sync the
   base. Native: `runtime_tokens()`, memory-owned names, clean
   `record_names()`, generated legacy keys, bindings. Python: `catalog.py`,
   `managym/decision_memory.py`, stubs. Rebuild the extension per the agent
   guide. Shows: both pinned pilots and `test_checkpoint_binding.py` pass
   unchanged; a fresh policy has the same embedding row count as before, with
   decision memory on and off.
2. **Consumers and guards.** `PROGRAM_VOCABULARY` without a filter,
   `observability.rs` helpers, `scripts/pilot_names.rs`, Python tests
   (union-equality test, record coverage, effect vocabulary, pooled mana),
   debug Rust tests for `record_vocabulary`, `token_renames`, `tokens` and the
   effect witness (`CARGO_BUILD_JOBS=2`, `--test-threads 2`).
3. **Docs and contract.** `docs/engine.md`, `manabot/README.md`,
   `managym/README.md`, cards test README, `AGENTS.md` naming and retirement
   rules, `.lf/qa/engine.md`; `scripts/public_api.py record`; PR body with the
   hold notice and release line.

## Done when

- The acceptance commands print what is shown above.
- A pre-change checkpoint, with and without decision memory, loads and plays
  its stored game to the same outputs.
- A reader of `docs/engine.md` can compute the admitted program vocabulary
  from three named lists without a string filter.
- The PR is open, green and marked held for 0.3.0.
