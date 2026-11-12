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
