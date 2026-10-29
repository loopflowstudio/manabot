# TLA cards — direct Rust cards and one world (accepted 2026-10-28)

Task 892c47 updates PR394 and stops at demo; merge, auto-merge, paid compute,
training and follow-up launches are not authorized. Direction: Jack Heart's
accepted "Direct Rust cards and worlds" in
[tla-rust-authoring-direction.md](tla-rust-authoring-direction.md). The earlier
Rust → cards.json → Python compiler → IR JSON → include_str! pipeline and its
review/recovery notes are history at `544ae312c:scratch/892c47-author-tla-cards.md`.
D1/f871a8 keeps parser, construction, `admit_deck` and consumer input; the
`content_pool`/`pool_content`/`Eligibility` API is unchanged.

## Implemented (2026-10-28)

1. **In-process compile** (`2b34e4964`): `managym/src/semantic/compile.rs` lowers the
   typed definitions to IR in memory (references, token closure, target roles,
   opcodes, hashes, qualification check). Proven identical to every removed
   checked-in IR (incl. ir/source hashes) and eligibility before deletion.
   Deleted: `content/semantic/current/generated/*`, `export_cards.rs`,
   `scripts/compile_current_content.py`, `manabot/semantic/content.py`.
   Python reads `managym.semantic_ir_json(pack_key)`; `CURRENT_IR_PATHS` →
   `CURRENT_PACK_KEYS`. `compiler.py` remains only for frozen v1/w7–w11 checks.
2. **Set files and keys**: `cardsets/` holds one file per printed set (tla, stx,
   mh1, bro, blb, ktk, afr, m20, iko, eld, zen, war, fin, lci, soi, arb, ons,
   lea) plus `tokens.rs`; keys are `<set>.<card>` / `token.*`. Set attribution
   for uncertain cards was checked on Scryfall/retail listings. Qualification
   hashes were refreshed only after a test proved every reviewed closure
   byte-identical once the rename is undone. `pool` now holds every non-token
   card (research cards included; still unsupported); the unused `tla` preset
   and `native_roots` are gone.
3. **One default pack**: native duplicates (elemental/sacrifice/temur files,
   STX Learn spells, basics, Lightning Bolt) deleted; the default pack registers
   all typed definitions. `cards/` holds the language (`authoring.rs`),
   `content_pack.rs` (from `cardsets/alpha.rs`) and `native.rs`: cards still
   lacking a typed op (Ancestral Recall, Pyroclasm, Man-o'-War, Waterfall
   Aerialist ward, Proft's Eidetic Memory, legacy vanilla/keyword creatures,
   Crossroads fixture). Legacy `CardDefId` prefix preserved by slot order.
4. **World**: `managym/src/world.rs` owns `WORLD_VERSION` (w16), `supports_card`,
   `unsupported_cards` (Python `managym.unsupported_cards`). WORLDS.md gains w16
   and "run an old world from its git revision".

## Remaining

- Migrate `cards/native.rs` into set files (needs new ops: target-player draw,
  damage each creature, defender, ward, conditional combat trigger) — not done.
- Gate/CI: full debug cargo, Clippy, world-change Python suite, conformance
  receipts (semantic-kernel corpus is still w15 — pre-existing failure),
  authored-match receipt (pre-existing stale keys), `test_semantic_cards`
  zero-logit failure (seen with identical IR; not shown pre-existing).
- Demo: human review of the set layout and world check; `scratch/demo.html`
  describes the superseded pipeline and needs replacing.

## Delete — do not maintain

Done in compress: `current/presentation.json` and its frozen-IR text fallback
(lowering IR without `text_box` now fails closed; the optional-payment test uses
current IR); `compiler.py`'s pool/token/text/enters_tapped additions (reverted to
main — only frozen sources use it). Next: `cards/native.rs` entries as their ops
land in set files.

PR394 updated at `6f8b2231e` (title/body rewritten; auto-merge unset).
Check: debug `cargo test --no-fail-fast` — only the pre-existing w15 conformance pair
fails (two stale test refs fixed and re-run); clippy clean; rebuilt cp312 extension;
103 Python consumer tests pass, `test_optional_payments` fails on a pre-existing schema-7 assertion.
Compress: semantic_ir/conformance/content_pack/lib semantic debug tests, test_temur
(schema assertion now tracks the current schema), test_current_content, test_policy_catalog pass.
