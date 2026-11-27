# MTG-501 · Admit granted-trigger program tokens

**Done when** a catalog-backed policy can prepare continuation facts carrying the complete 0.2.6 granted-trigger clock, and a caller of `managym.catalog_tokens()` can enumerate the exact program symbols accepted for the world, including TLA card programs and runtime-only records. A genuinely unknown symbolic pair still reports its kind and word.

## What the release did

I reproduced [issue 629](https://github.com/loopflowstudio/manabot/issues/629) with the published 0.2.6 wheels on CPU. Full-word continuation facts (kind 14, frame 1) passed through `fact_vocabulary` with lookup buffers from `catalog_tables("world", records=True, decision_memory=False)`. The error printed the issue's eight pairs in the same order. The complete clock payload has **eleven** missing pairs. `_require_keys` sorts them and prints the first eight. Each row below names the serializer input verified with the wheel's `catalog_token_value`.

| Token kind | Word | Spelling |
|---|---|---|
| Structure (1) | 0x15d1e4cc | `/PublicId` |
| Structure (1) | 0x1a65bcdc | `/PermanentId` |
| Structure (1) | 0x1e2e1716 | `PermanentId` |
| Structure (1) | 0x42b9f74a | `GrantClock` |
| Structure (1) | 0x4a2e5134 | `/GrantClock` |
| Structure (1) | 0x7069dc33 | `PublicId` |
| Field (2) | 0x169280b1 | `timestamp` |
| Field (2) | 0x274966b7 | `grantor` |
| Field (2) | 0x463aa4fe | `static_slot` |
| Field (2) | 0x4d7d0f4e | `ability_index` |
| Field (2) | 0x7771479d | `clock` |

At tag `v0.2.6`, `Projection::lki` serializes the departed permanent's `granted_abilities` into “last known characteristics” (`managym/src/agent/effect_execution.rs`). An `EffectiveTrigger` with a clock serializes all four `GrantClock` fields and both identity wrappers (`managym/state/src/ability.rs`). The fact encoder writes that ordered stream under `ContinuationProgram`. Neither printed definitions nor runtime registration supplied the eleven pairs. The wheel **does** admit `(6, 0x274966b7)` through `fact_vocabulary`. That text label introduces a projected reference and shares a word with the struct field. The two have different meanings and require distinct embedding rows.

Commit `8904bf12a5` already changed new emission. Today `EffectiveTrigger::clock()` derives the rules clock from skipped provenance fields. The existing test in `managym/tests/match_state_hash_tests.rs` checks that an acquired trigger has a clock in game logic and omits it in serialization. Historical fact rows still carry the old spellings. A current random self-play run could pass while admission of those rows remains broken.

## One implementation pass

1. Start with the failing continuation in `tests/model/test_effect_vocabulary.py`. Feed a representative 0.2.6 grant occurrence, including nonzero numeric operands, through the production host tables and `fact_vocabulary`. Cover dense split words and the v11 full-word path. Require valid embedding rows for every token, including distinct rows for text and field `grantor`. This fails on the existing registration even if attention and sharding are disabled.
2. Register the eleven historical structure/field spellings by kind in `execution_names` in `managym/src/agent/record_vocabulary.rs`. Record aliases, sorted lookup keys and learned rows then follow the existing `manabot/policy/catalog.py` path for fresh policies and the key-based checkpoint binder in `manabot/policy/binding.py`. Numeric operands continue to use the numeric kind rows. New observations keep the current viewer-safe projection. Two otherwise equal acquired triggers with different grantor IDs or clock values must yield identical learning tokens today.
3. Expose the finite inventory as **`catalog_tokens()["program_vocabulary"]`**, a sorted, deduplicated list of `(kind, value)` pairs. The native agent vocabulary module can compute it from the printed catalog already in hand and every non-null `RecordName.token`. Use this one union in the PyO3 binding and in place of the locally assembled unions in the runtime and exhaustive effect witnesses (`managym/cards/tests/cards/observability.rs`). The public stub documents the field as an additive `NotRequired` key for existing typed producers. Record the public API addition. Describe span traversal and vocabulary membership separately in the content contract.
4. Verify the exported inventory against independent inputs: the non-null pairs in `record_vocabulary()`, actual spans for every TLA definition/program, and the incident fixture. Admit the entire inventory with decision memory enabled. With memory disabled, check the incident, TLA spans and ordinary runtime records while retaining the existing optional-row exclusion. Keep existing unknown-symbol, arbitrary integer/boolean, null and definition-reference tests. Exercise the stored-policy path with the pinned pilot so a lookup fix cannot silently change learned predictions.

**Decision made for review.** The public catalog gains a membership field. Its existing `token_kind` and `token_value` arrays encode occurrences in printed card order. On the 0.2.6 wheel both the final program offset and the array length are 106,079. Appending the runtime vocabulary there would put card names and runtime labels in a stream whose consumers expect printed rules alone (`tests/semantic/test_learning_projection.py`), and sorting those pairs into the base embedding would grow even the hashed or memory-disabled models. The inventory is computed at catalog construction. Policy rows keep their capability filter and existing keyed allocation. In particular, diagnostics looking up a runtime symbol use the new pair field. Historical files and native game identities require no rewrite.

## Headless demo

After rebuilding the checkout's extension, run the following from the repository root:

```sh
uv run pytest -q tests/model/test_effect_vocabulary.py tests/semantic/test_learning_projection.py tests/model/test_pinned_pilot.py
uv run python scripts/public_api.py check
```

The first command should show the 0.2.6 occurrence passing on both transports, the full native inventory admitted for an enabled policy, strict rejection of a missing registration, intact printed-card spans and unchanged pinned predictions. The second should report a compatible published surface. The native runtime/effect witnesses additionally exercise real serializer output against the same exported inventory. No training launch is needed to observe the failure or its repair.
