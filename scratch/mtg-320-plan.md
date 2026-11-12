# MTG-320: runtime object kinds by zone

Docs: `docs/engine.md#objects-and-zones`, `managym/README.md` Architecture pointer (match the build).

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
