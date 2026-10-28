# TLA content implementation

Accepted design: [tla-content-handoff.md](tla-content-handoff.md), Jack Heart,
2026-10-28. One cohesive PR, review at demo; no merge or later Task launches.

Delete — do not maintain: removed `cardsets/tla.rs`, TLA blocks in elemental.rs,
sacrifice.rs and temur.rs, `register_complexity_presentation`, executable registry
lookup in `compile_content_pack`, and exclusive native/compiled parity assertions.
Kept directed rules witnesses, non-TLA registrations, token closure, historical
sources and all authored setup counts. No alternate current TLA executor remains.
Also removed repeated per-pack inventory/identity rows from generated eligibility
and `PolicyCatalogSpec`'s setup/schema fields. IR owns pack membership and IDs;
qualification manifests retain only eligible keys and exact IR hashes. Policy
binding uses the pool descriptor directly, without starting a game. No remaining
deletion target in this cut; preserve the public importer API and all identities.

Implemented: one current definition source, setup projections, shared lowering,
embedded presentation, closure-bound qualification and typed native/Python pool
interface. Added 24 complete cards, migrated Avatar Enthusiasts/Glider Kids, and
qualified those 26 plus five basics. All nine fixed token templates are explicit.
World w16 / learning schema 8 are fresh; w15 remains reserved by MTG-177.
Durable `docs/rules/tla/` retains 286 card owners, 294 faces, 9 fixed tokens,
Copy and the explicit Treasure exclusion, plus final research source pins.
All T1 cards have programs; preserved programs without complete independent
qualification remain unsupported for arbitrary pool entry.

Five final-research T1 deferrals: Foggy Swamp Hunters → T3 (conditional keyword),
Pirate Peddlers/Zhao → T4 (`another` sacrifice matcher), Vindictive Warden → T4
(untargeted damage), Rocky Rebuke → T4 (invalid victim retained while source legal,
pending MTG-177). Its positive-only draft was removed, evidence retained in
`scratch/rocky-rebuke-positive.rs`; no runtime repair was copied.

Parallel importer f871a8 consumes `content_pool` / `pool_content` under the
agreed contract and received checkpoints 73865361f and 1e0d3c998. It owns
native deck admission, parsing and consumer inputs. Final content checkpoint
includes the land correction, qualified pool policy binding and consumer proof.

Check: focused debug Rust card tests (16 retained + Avatar 5 + Glider 4),
descriptor and extended Rumble witnesses passed; release native rebuild and all
6 `tests/semantic/test_current_content.py` checks passed (generation, all learning
projections, stale card/token qualification, preserved setup counts, and real
Command/fork/viewer-safe Academy path). Formatting/lint passed. Gate owns complete
debug/clippy, Python current-world/arena/model/choice suites and new versioned
conformance/Learn/browser receipts; preserve old evidence. New test collection
and release source inventory checks are also gate-owned. Publication follows gate,
then demo. No training or broad suites run here.

Check: `cargo test --lib semantic::pool::tests` (2 passed), release wheel rebuilt
with `uv run --extra play maturin build` and installed atomically;
`uv run --extra dev pytest tests/semantic/test_current_content.py tests/model/test_policy_catalog.py -q`
(12 passed). All 10 native descriptors, 8 authored policy bindings and 11 IR/
coverage projections retain their exact identities and decisions. Eligibility
shrinks from 19,881 lines to 445; gate retains the broader checks above.
