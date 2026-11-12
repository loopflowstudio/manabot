# MTG-320: runtime object kinds by zone

## Rename pass (steers d57fc17e, 0779d3e9, e3b2a142, dbf13009; 2026-11-12)
- Merged current MTG-298 branch (6320ad43c); `PutIntoGraveyard` now reads `moved.is_card()`;
  `become_copy_of` already leaves kind alone. Four card tests moved off the removed `is_token` field.
- Design written to docs/engine.md "Objects and zones" (CR term -> type table, "What is built",
  token moment). Built: record `Object` in `state::game_object`, `ObjectId`/`ObjectVec`,
  `GameState::objects`, `object_to_permanent`, `PublicId` for the old u32 id, locals/fields renamed
  with serde wire names kept. Details and exclusions in questions.md.
- Wire check: `semantic-kernel-w17-v2` conformance reproduces (state hashes byte-identical). A
  misplaced serde attribute (on `ObjectLki.counters`) was the one drift found, by diffing regenerated
  traces against a base-revision build.
- Checks: `cargo nextest run --workspace --profile ci` 1166 passed, 4 timed out at load ~250-380;
  those 4 pass alone without the limit (conformance 508 s, base revision 345 s at the same load).
- clippy `--workspace --all-targets -D warnings` clean. Rebuilt cp313 extension; pytest env, belief,
  semantic, verify, academy + world-change list: 907 passed, 1 failed
  (`tests/verify/test_input_v10.py::test_v10_vector_buffers_reach_the_model`: "policy forward
  requires an observation from admit_observation"; no Python differs from main on this branch, so
  inherited), 1 deselected (`test_real_population_learning_is_held_out_and_beats_p0` hit the 5-min
  per-test limit at load ~260; trains a model, touches nothing renamed).
- Left for the PR body as remaining: Card/CardId store split, new object per zone change, CR 400.7
  carry-over record, bench numbers (Engine Budget in CI).

Docs: `docs/engine.md#objects-and-zones` (rename written 2026-11-12: `GameObject`/`ObjectId`/`GameState::objects`, card-only naming rule, departure "record named Card" dropped), `managym/README.md` (Architecture pointer; policy-input paragraph says `ObjectId`). Build must match these names.

Built:
- `state::card::ObjectKind { Card, Token, SpellCopy }`, private `kind` on the runtime object record
  (`Card`), set at construction (`from_definition` reads the row's `is_token`; `with_kind` for token
  copies and spell copies). `is_card()`; `resolve_into_token()` in `commit_zone_move` when a spell copy
  enters the battlefield (CR 111.12).
- Runtime readers moved off the definition flag: SBA (tokens off battlefield, spell copies off stack,
  CR 704.5d-e), predicates/layers (`token` filter = kind Token), cast_permission exile grants,
  "card" selection in decision.rs, return-to-battlefield guard in resolution.rs.
- Deleted: `show()` re-setting `is_token` on a token's back face; `prepare_token_copy` writing
  `is_token` into the copy's definition. `CardDefinition.is_token` now only marks token catalog rows.
- Byte-identity: observation `is_token` = `!is_card()`; match-state hash `copied_characteristics` for
  non-card objects carries `is_token: true` (v16 bytes unchanged, no version bump).
- Test: Azula instant copy is `SpellCopy` and ceases off the stack; existing permanent-copy test checks Token.

Checks: `cargo nextest run --workspace --profile ci` 1058 passed; rebuilt cp313 extension; pytest
belief/verify/semantic/search-command-evidence/academy authored+parity+study 444 passed.
Remaining for gate: clippy; CLAUDE.md world-change pytest list; Engine Budget in CI (digest should not move).

Upstream (origin/main 92e476217, checked 2026-11-12): this branch stacks on MTG-298 (`ac213b488`),
which is not on main yet; its merge base with main is `01d5abd2c`. Main has since added two runtime
readers of the definition flag that must move to the kind when this branch meets main:
- `TriggerCondition::PutIntoGraveyard` in `flow/triggers.rs` matching: `!moved.is_token` ->
  `moved.is_card()` ("card put into a graveyard"; a dying token doesn't match).
- `Card::become_copy_of` in `state/src/card.rs` saves and restores `is_token` around the copy; with the
  private `kind` untouched by copying, that save/restore goes away.
After the sync, re-grep runtime `is_token` reads (`git grep -n is_token -- managym/src managym/state/src`)
and rerun the debug nextest. Main's new cards define no emblem and cast no copy of a card, so the
"none yet" rows in `docs/engine.md#objects` stay true.

Compress: only import tidying (`ObjectKind` in retarget.rs, predicate.rs). Kept `kind() == ObjectKind::Token`
over a `Card::is_token()` method: it would share a name with the deref'd row flag `CardDefinition.is_token`.
No runtime `Card` reads that flag any more (grep: remaining reads are rows or observation `CardData`).
`cargo build --tests` ok.

Concept review (2026-11-12): no concept change. The model reads as the rules do: a definition (card or
token row) says what a thing is, `ObjectKind` says what each game object is, zones say where. One
authority per fact: `kind` for runtime questions, the row's `is_token` only for catalog rows and the
frozen wire/hash bytes. Emblem and copy-of-a-card stay "none yet". Keeping the record named `Card` is
a naming refactor, not a product choice; it stays as recorded in questions.md. The only outstanding
work is the main sync above (two readers to move to `is_card()`/kind), already written as direction.
