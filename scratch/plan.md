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

PR #636 (MTG-501) merged as `main` 624751ae71 on 2026-11-27 and is an
ancestor of the implementation. It adds `PROGRAM_VOCABULARY`, `missing_program_tokens`,
the debug emission guard, more `execution_names` registrations, the docs
section "Program-token vocabulary", and three of the prefix filters this Task
removes. Not stacked on #612; the two meet at the 0.3.0 cut.

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
- Two live renames sit only on synthetic `RECORDS` rows
  (`execution/6/kept creature` → `execution/6/kept permanent`, and the
  `creatures each player sacrifices` pair); `TOKENS` lists both old texts with
  `None` (`token_renames.rs:183-225` at 624751ae71). Before #636 the `TOKENS`
  test accepted only printed successors; #636 switched it to
  `PROGRAM_VOCABULARY`, so the renames can move onto `TOKENS`.
- `CatalogTokens` stores 32-bit hashes, not names: a retirement whose name
  string is no longer listed anywhere cannot be recomputed. Hence a frozen
  ledger, not a generator over live lists.
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
  `tests/semantic/test_pooled_mana_classes_input.py`,
  `tests/model/test_record_vocabulary.py:104` (counts `record_name_keys`
  against `len(record_vocabulary())` with decision memory on), docs
  (`docs/engine.md#program-token-vocabulary`, `manabot/README.md`,
  `managym/cards/tests/cards/README.md`, `AGENTS.md`, `.lf/qa/engine.md`).

## Design

### Native: two vocabularies, one memory list

`managym/src/agent/record_vocabulary.rs` stays the one registration site:

- `record_names() -> Vec<RecordName>`: names the fact and shared-effect
  encoders hash. No `execution/*` rows, no decision-memory names. Its `token`
  column is only an alias into the program vocabulary, never a token source.
- `runtime_tokens() -> Vec<RuntimeToken>` (`kind: TokenKind`, `name: String`):
  every program token a runtime projection can emit, sorted and de-duplicated:
  today's `execution_names` body plus every token a record name aliases
  (`Modification::AddTypes` beside its closing `/Modification::AddTypes`,
  unprinted `Keyword::*`, `CounterKind::*`, `SourceQuality::*`, subtypes). It
  may overlap printed spans; it is never filtered against the pack, so a card
  that later prints a runtime symbol changes nothing.
- One pack walk serves both functions (today each serializes every
  `ObjectDefinition`).

The admitted program vocabulary is `catalog_tokens()` ∪ `runtime_tokens()`.
`PROGRAM_VOCABULARY` is that union with no filter.

`managym/src/agent/decision_memory.rs` owns its names: a function beside
`RECORD_NAMES` returns each `DecisionMemory::<name>` with its hash and no
token. The uniqueness test checks `record_names()` and the memory names
together, since they share the `token` namespace.

### Python surface

- `managym.record_vocabulary()`: same signature, real record names only.
- `managym.runtime_tokens() -> list[tuple[int, str, int]]`: `(kind, name,
  value)`.
- `managym.decision_memory_names() -> list[tuple[str, int]]`: `(name, value)`;
  `managym.decision_memory.decision_name_values()` derives from it.
- Both new functions go in `managym/__init__.pyi` and `tests/public_api.json`
  (`scripts/public_api.py record`, additions).
- `manabot/policy/catalog.py::_record_tables`: printed rows, then each runtime
  token not already present appends a row; record names then alias an
  existing token row (asserted, never appended) or append a name row; only
  when `decision_memory` is on, each memory name appends a name row. No fake
  rows, no prefix test.
- `manabot/policy/decision_memory.py` reads names with
  `RecordNamespace.TOKEN`, not the literal `1`.

### `catalog_tokens()` naming (reviewed, kept)

`catalog_tokens()` keeps its name: it is the per-definition token catalog,
rows by `CardDefId`, sharing the word with `CatalogTokens`,
`catalog_token_value` and `catalog_token_renames`; it was renamed once already
in 0.2.3. Issue 629's confusion was the missing second surface, which
`runtime_tokens()` now is. Its docstring and `docs/engine.md` state the
two-list union, and a Python test holds a fresh policy's `record_token_keys`
equal to exactly that union with decision memory on and off.

### Legacy keys: a frozen ledger

Retirements are history, so they are not generated from lists that keep
changing. `managym/src/cards/token_renames.rs`, returned through
`catalog_token_renames()`:

- `TOKENS` carries the two live renames that today ride on `RECORDS`:
  `(Text, "kept creature", Some("kept permanent"))` and
  `(Text, "creatures each player sacrifices", Some("permanents each player
  sacrifices"))`. Leaving them `None` while deleting the `execution/6/…` rows
  would mean-initialize both vectors on pre-rename checkpoints.
- `LEGACY_EXECUTION_TOKENS`: manually registered pre-0.3.0 runtime symbols
  from structs, enums, fields and `EXECUTION_TEXT`, the 13 synthetic keys
  already retired before the split, and the play-grant condition label
  `you control a Wizard permanent`, frozen as `(kind, name)`.
- `LEGACY_DECISION_MEMORY_TOKENS`: the 44 pre-0.3.0 `DecisionMemory::*`
  names, retired as `(Variant, name, None)`, independent of later
  `RECORD_NAMES` edits.
- Record retirements: literal `RECORDS` (now only
  `Modification::ControlChanged`) plus `("token", "execution/<kind>/<name>",
  None)` for each `LEGACY_EXECUTION_TOKENS` entry and each pack card or set
  name. Token renames and live play-grant condition labels do not feed the
  record ledger.
- Tests check that retired runtime and phantom memory keys stay classified,
  that dispositions are unique by packed key and that successors resolve.

`binding.py` needs no new rule. Skipping a missing `record_name` key whose row
a `record_token` already bound was considered and rejected: real aliased names
(`flying` ↔ `Keyword::flying`) share rows too, so it would swallow an
accidental deletion of a live record name.

### Historical name tracer

`scripts/pinned_pilot.py::names_at` copies HEAD's `scripts/pilot_names.rs` into
an archived writer's crate. It detects whether that crate's
`record_vocabulary.rs` defines `pub fn runtime_tokens` (as it already inspects
`tokens.rs`) and compiles the matching probe, so `names --at <rev>` keeps
working for pre-0.3.0 writers such as `ab2b757d9`.

## Decisions made for review

Nobody was present; these were chosen and can be reversed before the 0.3.0
cut. Items 2, 4 and 5 follow the concept review's recommendations.

1. **Hold for 0.3.0 with no 0.2.x shim.** Passed over: landing now with
   `record_vocabulary()` kept as a deprecated composite and the clean lists
   under new names (a second rename of a function renamed in 0.2.3, and a dual
   state until the cut).
2. **Decision-memory names leave `record_vocabulary()` for a top-level
   `managym.decision_memory_names()`.** The brief says "Change
   `DecisionMemory::*` entries to `record(Token, …, None)` and remove the
   repeated `DecisionMemory::` prefix filters"; keeping them in the shared
   table would keep prefix tests in `catalog.py` and `decision_name_values`.
   Top level keeps every engine vocabulary on the stub-checked surface.
   Passed over: submodule `managym.decision_memory.decision_names()` over a
   private `_managym` binding; rows kept in `record_vocabulary()` with
   `token=None`.
3. **`catalog_tokens()` is not renamed**; it is paired with
   `runtime_tokens()`. Passed over: `printed_tokens()` with 0.3.0 → 0.4.0
   shims across about 45 call sites.
4. **`runtime_tokens()` is complete: admitted = `catalog_tokens()` ∪
   `runtime_tokens()`**, with record-name tokens pure aliases. Passed over: the
   old `execution_names` body only, which leaves a three-way union and repeats
   issue 629 one layer down. Costly to change once 0.3.0 publishes it.
5. **Legacy keys come from a frozen ledger** (runtime symbols and 44
   memory names as constants, with pack names derived).
   Passed over: the whole list including card names as one static table
   (must be regenerated if cards land before the cut); a generator over live
   lists (loses vectors when a runtime token graduates to print or a memory
   name is cleaned up).
6. **Decision-memory retirement is covered by a constructed pre-0.3.0
   checkpoint** in `tests/model/test_checkpoint_binding.py`; no retained
   memory checkpoint is known to pin.
7. **Pack card/set `execution/6/<name>` retirements stay derived from the
   append-only content pack until the 0.3.0 cut** (review recommendation 1a);
   only generated prose and `TOKENS` leave the feed. Passed over: freezing
   ~1,255 card/set names as a static table now and deleting
   `content_token_texts()` (must be regenerated for cards landing before the
   cut). Worth freezing at the cut if a card rename could drop a name.

## Acceptance check

With the rebuilt extension installed:

```sh
uv run python -c "import managym; v=managym.record_vocabulary(); print(sum(n.startswith(('execution/','DecisionMemory::')) for _,n,*_ in v), len(managym.runtime_tokens()) > 0, len(managym.decision_memory_names()))"
# 0 True 44
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
  "unexplained … key"; the ledger does not widen that to arbitrary keys.
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
  names are `decision_memory_names()`."

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

1. Checkpoints survive the split (221af98c25): both pinned pilots unchanged,
   probe `0 True 44`.
2. Consumers and guards (0b770d16be): name probe, union check with memory on
   and off, writer tracer at `ab2b757d9` (2,553 keys) and HEAD.
3. Docs and contract (10b956fdd5, 0b770d16be): API recorded, contract tests
   pass.
4. Frozen ledger (5301417e8): 87 inferred dispositions dropped, all 557
   `ab2b757d9` execution keys covered. Six focused debug Rust tests and ten
   Python checks pass with the rebuilt extension, including both pinned pilots
   and the constructed memory checkpoint. The optional `u1952` replay skipped.

After Pass 4: PR body with the 0.3.0 hold notice and release line; merge and
auto-merge stay off.

## Concept review dispositions

Pre-implementation review (a173411b48): all findings built in Passes 1–3.

Post-implementation review (54e1b1816b):
- `record_renames()` chains all of `TOKENS` (61 printed-only, plus every
  future retirement) into phantom `execution/*` retirements → Pass 4.
- Live `condition_text` prose feeds retirements; a wording change would make
  every pre-0.3.0 checkpoint fail with "unexplained record_name key" → Pass 4.
- Named `RECORDS` constant gone and `record_renames()` untested → Pass 4.
- Stale module headers in `record_vocabulary.rs` and
  `manabot/verify/record_coverage.py` → fixed in this realign.
- Divergent decision-memory label admission → Deferred findings.
- Review question 1 (card/set retirements) → decision 7. Question 2 (hold,
  constructed memory checkpoint, API names) → decisions 1, 2, 6 stand.

## Deferred findings

- **Unify decision-memory label admission** (pre-existing, separate Task).
  `manabot/policy/decision_memory.py:498-504` checks `kind`/`choice_kind`
  only against the running engine's `decision_name_values()`, never the
  checkpoint's `record_name_keys`; `manabot/policy/admission.py:394-398`
  checks only `record_name_keys` (accepts any `TOKEN` name such as `flying`)
  and skips models with `decision_memory="none"`. Takes one shared helper
  (membership in both) and a regression test for an incomplete checkpoint
  table on retained prefixes and a non-memory `TOKEN` name on v11 batches.

## Done when

- The acceptance commands print what is shown above.
- A pre-change checkpoint, with and without decision memory, loads and plays
  its stored game to the same outputs.
- A reader of `docs/engine.md` can compute the admitted program vocabulary
  from the printed and runtime lists without a string filter.
- The PR is open, green and marked held for 0.3.0.
