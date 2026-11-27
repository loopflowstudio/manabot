# MTG-501 · Guard runtime program-token emission

**Done when** no runtime emitter can produce a non-numeric program token that a catalog policy with `decision_memory=False` would reject, and the Rust test suite enforces this at the emission point. The public API stays unchanged. An unknown symbolic pair still fails Python admission, and the error names its `(kind, 0xword)`.

## What people said

Jack Heart's Task steer (ee8ad494), verbatim:

> 1. Do not register the 11 historical 0.2.6 GrantClock/PermanentId/PublicId/field tokens in record_vocabulary.rs (no 0.2.6 row shim; commit 8904bf12a5 already stopped emitting them and 0.2.6 rejected them anyway).
> 2. Do not add a new program_vocabulary field to managym.catalog_tokens(); keep the public API surface unchanged.
> 3. Add a cfg(debug_assertions) guard at TokenWriter::into_rules_tokens asserting every emitted non-numeric runtime program token is in catalog_tokens ∪ record_names, and fix any latent unregistered runtime tokens it surfaces in the Rust test suite (projecting storage IDs as visible references rather than registering raw ID structs).
> 4. Add a directed regression test for a departed permanent with an acquired once-per-turn trigger under a suspended frame (e.g. Koh inheriting a once-per-turn trigger), verifying last-known-characteristics emission stays within the vocabulary and emits no GrantClock/PermanentId/PublicId tokens.
> 5. Keep Python admission tests covering record_vocabulary() / catalog_tokens() acceptance and clear rejection of unknown program tokens.

The kickoff design (register 11 historical pairs and add `catalog_tokens()["program_vocabulary"]`) was replaced by this steer and is not built.

## Findings that still shape the approach

- Issue 629 came from 0.2.6 `Projection::lki` serializing a departed permanent's `granted_abilities` with `GrantClock` and its raw `PermanentId`/`PublicId` wrappers. Commit `8904bf12a5` stopped that on main: `EffectiveTrigger` skips its provenance fields and derives `clock()`. `managym/tests/match_state_hash_tests.rs` covers serialization, but no test checks the vocabulary on the LKI path.
- Every runtime program goes through `TokenWriter::into_rules_tokens` (`managym/src/cards/tokens.rs:297`). `catalog_tokens()` uses `span()` instead, so a guard that builds its vocabulary from the catalog does not recurse. `TokenWriter` holds `pack`.
- Live bug on main: `effect_execution.rs:919` calls `self.object("enters attached to", …)`, but that label is missing from `EXECUTION_TEXT`. The call hits a `debug_assert!` in debug builds, and in release the label reaches the policy unregistered. No test covers it (an Aura entering attached under a pending replacement).
- There are four separate Rust versions of the admitted vocabulary: `record_vocabulary.rs:438`, `observability.rs:2374`, `observability.rs:5651` and `token_renames.rs:277`. The one at `observability.rs:5651` walks only `managym_cards::catalog()`, so it misses tokens, back faces and fixtures. Both 5500 and 5662 wrongly exempt `TokenKind::Null`. Python (`admission.py:510`) exempts only Integer and Boolean (`NUMBER_TOKEN_KINDS`).
- `record_vocabulary.rs:91-95` gives all 44 `DecisionMemory::*` names a `Variant` token. Default policies (`decision_memory=False`) reject those tokens, so a plain `catalog ∪ record_names` set would wrongly admit them.

## Decisions made for review (from concept review, headless)

- **Guard rule (refines steer 3).** The guard checks the vocabulary a `decision_memory=False` policy admits: the printed catalog (tokens and back faces included) plus non-`DecisionMemory::*` record tokens. Only Integer and Boolean skip the lookup; Null and DefinitionRef must match exactly. It runs only when `std::ptr::eq(self.pack, default_content_pack())`, so custom test packs don't false-fail, and it reports every missing pair in a stream before panicking. The steer's literal "catalog_tokens ∪ record_names" was not used because it would wrongly admit the phantom `DecisionMemory::*` tokens.
- **1a: guard every caller of `into_rules_tokens`**, including `definition_tokens` (decision memory encodes live characteristics through it). Not chosen: 1b, which guards runtime callers only and leaves `decision_memory.rs:425` unguarded.
- **2a: leave the three unused `EXECUTION_TEXT` strings alone** ("creature type choice", "chosen creature type", "normal combat damage"). Retiring them would add six unrelated `token_renames.rs` entries.
- **3a: the `execution/*` and `DecisionMemory::*` cleanup is its own 0.3.0 outcome** (see Deferred findings).

## Docs written in place

- `docs/engine.md` (`### Program-token vocabulary` clarifying `catalog_tokens()` vs `record_vocabulary()` and exact admission rules)
- `managym/__init__.pyi` (docstrings on `catalog_tokens()` and `record_vocabulary()`; signatures unchanged)
- `managym/cards/tests/cards/README.md` (`## Effect vocabulary checks` updated for the debug `TokenWriter::into_rules_tokens` guard, shared `decision_memory=False` vocabulary rule, storage-ID projection rule, and dense+v11 admission tests)
- `docs/development.md` (effect/runtime-projection validation note referencing the debug emission guard)
- `manabot/README.md` (`### Semantic catalog dimensions` clarifying runtime program-token sources and `(kind, 0x…)` admission failures)

## Remaining: one implementation pass

1. Add `record_vocabulary::program_vocabulary()`, a `OnceLock<HashSet<(u16, i32)>>` following the guard rule. It is internal Rust, with no PyO3 or stub change.
2. Guard `TokenWriter::into_rules_tokens` under `cfg(debug_assertions)` with that set. If `observability.rs:2365` needs to aggregate across many effects, add an unchecked `#[cfg(test)]` helper.
3. Point the four Rust oracles (`record_vocabulary.rs:438`, `observability.rs:2374` and `:5651`, `token_renames.rs:277`) at `program_vocabulary()`. This drops the Null exemption and the catalog-only walk. If that lets a runtime-only rename resolve, keep the existing dispositions unless a test forces a change.
4. Register "enters attached to" in `EXECUTION_TEXT`, which adds one record row (fresh rows bind by mean imputation in `bind_catalog`). Add a directed test that executes `Projection::entry` with `attached_to`.
5. Steer 4: add a directed test with a departed permanent holding an acquired once-per-turn trigger under a suspended frame (Koh, or another TLA card that inherits one). Its LKI program must stay inside `program_vocabulary()`, and none of the 11 0.2.6 pairs from issue 629 may appear (for example `GrantClock` 0x42b9f74a, field `grantor` 0x274966b7).
6. Run the Rust suite in debug (`CARGO_BUILD_JOBS=2`, `--test-threads 2`). Fix every unregistered runtime token the guard finds: project storage IDs as visible references, never register ID structs. Record each fix here.
7. Steer 5: in `tests/model/test_effect_vocabulary.py`, check that `fact_vocabulary` with production host tables accepts every non-`DecisionMemory` `record_vocabulary()` token and the TLA printed spans on both the dense and v11 paths. An unknown pair must be rejected with its `(kind, 0xword)` (keep the existing unknown/null/definition-reference tests). After registering the new row, run `tests/model/test_pinned_pilot.py` to confirm stored-pilot predictions are unchanged.

Validation: rebuild the extension; `cargo nextest run --profile ci` in debug for managym and its cards crate; `uv run pytest -q tests/model/test_effect_vocabulary.py tests/semantic/test_learning_projection.py tests/model/test_pinned_pilot.py`; `uv run python scripts/public_api.py check`, which must be unchanged.

## Deferred findings

- **Untangle `record_vocabulary()` (0.3.0).** About 350 synthetic `("token", "execution/<kind>/<name>", …)` rows exist only to carry runtime program tokens into `catalog.py`'s `record_token_keys`. The 44 `DecisionMemory::*` names carry phantom `Variant` tokens. The fix: expose runtime program tokens directly; change `DecisionMemory::*` to `record(Token, …, None)`; build `record_token_keys` without the fake records; and bind legacy keys in `binding.py`/`token_renames.rs` so pre-0.3.0 checkpoints still load. This breaks checkpoint compatibility, so it belongs with the 0.3.0 removals as a separate Task.
- **Naming of `catalog_tokens()`.** Readers, including the author of issue 629, take it for the full vocabulary, but it holds only printed spans. Clarify the docs and the error text (runtime tokens come from `record_vocabulary()`) without changing the API. Do this in this pass if the docstring/error edit is trivial.
