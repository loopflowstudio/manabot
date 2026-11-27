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

## Implementation

- Pass 1 `f35670541`: one shared Rust vocabulary behind the debug emission guard and four checks; attachment label, `LoyaltyCost::MinusX`, `/Modification::AddTypes`/`AddColors` registered; Koh LKI regression; dense+v11 `fact_vocabulary` tests. Debug Nextest: 2,393 pass, 3 failures (then fixed; seven focused reruns passed). 57 Python tests passed on the rebuilt cp313 extension; public API unchanged.

## Pass 2 (next): close the guard's own gaps

From the post-implementation concept review (`scratch/concept-review.html`, HEAD 182634051f). Choices 1a/2a/3a adopted headless, marked for review:

1. **Restore guard coverage in `observability.rs` (1a).** `witness_pack()` (`observability.rs:2350`) clones the default pack to dodge the `ptr::eq` check, and `tokens_of()` routes through it, so ~46 witness tests run unguarded. Delete `witness_pack()`; `tokens_of()` uses `default_content_pack()`. Add a `#[doc(hidden)]` unchecked serializer (`definition_tokens_unchecked` / `rules_tokens_unchecked` in `cards/tokens.rs`) used only by `every_effect_variant_reaches_the_learning_tokens`, which still checks `program_vocabulary()` and reports all missing pairs at once. Fix any unregistered witness tokens this surfaces. Passed over: 1b (keep the clone inside that one test) — keeps a full-catalog clone hack.
2. **One out-of-catalog chosen-name rule across all surfaces (2a).** `SearchNamed` (`effect_execution.rs:1766`) masks names absent from `name_candidates()` to `Null`; `Projection::entry`'s `EntryChoice::CardName` (`effect_execution.rs:~871`, "chosen on entry"), a suspended `Effect::ExileNamed`, and `FactKind::ChosenName` (`observation_encoder.rs:1882`) still emit the raw name. Share one helper so all four encode an uncatalogued name as Null/omitted while the text keeps the spelling; fix the `SearchNamed` note (no `Some("…")` Debug wrapping, no duplicate line). Directed test: Sorcerous Spyglass (`xln`) with `scenario_naming_vocabulary()` choosing a fixture-only name, observed under a pending entry replacement and on the battlefield. Passed over: 2b (strict production, fixture-only names) — leaves the rule half-applied in a different direction and needs `naming_tests.rs` changes.
3. **Cover `effect_vocabulary` admission (3a).** Add a bounded probe in `tests/model/test_effect_vocabulary.py` for shared continuous effects: accepts the registered `/Modification::AddTypes`/`AddColors` tokens, rejects an unknown pair and a `DecisionMemory::*` token with `(kind, 0x…)` in the message. Keep `admission.py:516` wording unchanged (docs carry the `catalog_tokens()` vs `record_vocabulary()` split). Passed over: 3b (append a source hint to the error) — regex churn for little gain.

Validate: debug `cargo nextest` on `managym/cards` `tests/cards.rs` (observability, xln, tla) and the touched engine tests, `CARGO_BUILD_JOBS=2 --test-threads 2`; rebuild cp313 extension; `uv run python -m pytest tests/model/test_effect_vocabulary.py`; `uv run python scripts/public_api.py check`. Full suite to CI.

## Deferred findings

- **Untangle `record_vocabulary()` (0.3.0).** About 350 synthetic `("token", "execution/<kind>/<name>", …)` rows exist only to carry runtime program tokens into `catalog.py`'s `record_token_keys`. The 44 `DecisionMemory::*` names carry phantom `Variant` tokens. The fix: expose runtime program tokens directly; change `DecisionMemory::*` to `record(Token, …, None)`; build `record_token_keys` without the fake records; and bind legacy keys in `binding.py`/`token_renames.rs` so pre-0.3.0 checkpoints still load. This breaks checkpoint compatibility, so it belongs with the 0.3.0 removals as a separate Task.
- **Naming of `catalog_tokens()`.** Readers, including the author of issue 629, take it for the full vocabulary, but it holds only printed spans. Docs and docstrings now state the split (pass 1); renaming is a 0.3.0 API question alongside the `record_vocabulary()` untangling. Review finding 4 (synthetic `execution/*` rows, four repeated `DecisionMemory::` prefix filters) is this same 0.3.0 outcome.
