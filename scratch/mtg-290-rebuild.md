# MTG-290 rebuild on v9-only main

Direction (2026-11-12): PR #468 conflicted with main after #464 ("Standardize on
policy input v9") and 92560a1e0 (managym published as its own package). Jack Heart
chose "Rebuild on main": keep what still applies, drop version selection.

Branch state: commit 7a190564d merged origin/main, taking main's side in all 22
conflicted files. The remaining PR diff (`git diff origin/main --stat`) holds this
PR's non-conflicting pieces, written for v1–v9; they do not compile against main yet.

Keep, rebuilt for main's single v9 encoding:
- `managym/src/agent/observation_layout.rs` as the one native authority for named
  columns of the v9 encoding (plus main's `policy_history` event variant if it
  still exists). Native encoder writes and the observability guard
  (`managym/cards/tests/cards/observability.rs`) use those names; keep numeric
  ABI assertions and Python/native parity checks.
- Python `managym.observation_layout()` (history option only if main keeps it),
  typed in `managym/__init__.pyi` (the stub moved there from `manabot/managym/`).
  Torch-free discovery; the jace column-24 `definition` lookup works by name.
- Engine guide docs: reading arrays by name and the "Card definition IDs" section
  (IDs renumber when the catalog grows; join through
  `content_pool("world")["definitions"]`; features from `catalog_tokens()`).
  Place them where main's engine docs now live (`docs/engine.md` and/or the new
  `docs/engine-python.md` for engine-only consumers); `docs/observation-and-model.md`
  links to them.
- Renames with deprecated aliases (warn, removed in 0.3.0, tracked by MTG-294):
  `learning_catalog`→`catalog_tokens`, `LearningCatalog`→`CatalogTokens`,
  `learning_token_value`→`catalog_token_value`,
  `learning_record_names`→`record_vocabulary`. Aliases live in
  `managym/__init__.py` (and stubs) now that the engine package moved.
- AGENTS.md naming rule (re-add; main's AGENTS.md won the conflict): "Name a thing
  for what it is where it is produced, not for how one consumer uses it. managym
  produces observations, encodings and card tokens for any consumer, so its API
  says `catalog_tokens` and `record_vocabulary`; manabot may name the same value by
  its use. Rename a published name with a deprecated alias that warns and states
  its removal release; serialized configuration keys, checkpoint fields and frozen
  evidence keep their spelling."

Drop (no longer meaningful on v9-only main):
- `recommended_encoding_version`, `latest_encoding_version`, the
  `encoding_version=` / `policy_input_version=` keyword and alias, v1–v8 layouts,
  the reworded v6/v7 rejection error and the "choosing a version" docs.
- MTG-294 should then cover only the `learning_*` aliases; update its notes.

Open owner question from the earlier pass (v9 as fresh default) is settled by main.
Delete `scratch/` entirely in the final commit before landing.

## Status (implement pass, 2026-11-12)

Built on main's v9-only encoder:
- `observation_layout.rs` has no version argument: `observation_layout(policy_history)`.
  Encoder, continuation, shared-effect and history writes index by layout names.
  Compress pass deleted every column `*_INDEX` alias (encoder, continuation,
  observability guard, cr_602 test now use `observation_layout::<axis>::<NAME>`);
  `*_DIM` stay as buffer-width names equal to `<axis>::WIDTH`.
- Python `managym.observation_layout(*, policy_history=False)`; renames with warning
  aliases in `managym/__init__.py(i)`; docs in `docs/engine.md#encoded-observations`,
  linked from `docs/engine-python.md` and `docs/observation-and-model.md`.
- MTG-294 comment already narrows it to the `learning_*` aliases.

Checks: `cargo test --lib agent::` 50 passed; cards `observability` 58 passed;
clippy `-D warnings` clean; layout + rename-affected pytest (10 files) 151 passed,
layout test 8 passed; academy parity + MPS 22 passed; guide example run verbatim.

Compress checks: `cargo test --lib agent::` 50 passed; cards `observability` 58
passed; rules `cr_602_policy` 1 passed; clippy `-D warnings` clean.

Steer (2026-11-12), Jack Heart: "i dont need to review again". Ship the rebuild at
demo: remove scratch/, mark ready, enable squash auto-merge.

Acceptance against the Task brief: (1) layout readable from Python, tied to native
widths and the encoder's values (`tests/env/test_named_observation_layout.py`), and
column 24 is `agent_permanents.columns.definition`; (2) default-version guidance and
the version-naming rejection error are moot on v9-only main, where any other version
raises `UnsupportedPolicyInput` naming `ac601834d`. Realign (2026-11-12): branch
contains origin/main 92560a1e0; no stale `*_encoding_version` references remain;
`LearningCatalog` is a stub-only TypedDict alias, as `LearningCatalog` was stub-only
on main.

Remaining: gate; delete `scratch/` in the final commit before landing.

Concept review (2026-11-12): no change. One concept (`observation_layout()`, from the
column table the native encoder writes through) plus renames by the producing-side
naming rule. Considered and kept: `manabot/env/observation.py` keeps its own index
constants as the independent reference encoder, checked by native/Python parity
(the design's stated choice); `policy_history=` matches the encoder option it
describes, so renaming it to suit the new naming rule would only churn the API.
