# TLA Rust authoring — implementation, 2026-10-28

Accepted: Jack Heart's [Rust-first revision](tla-rust-authoring-direction.md)
supersedes JSON authoring. This remains Task 892c47 / draft PR394, one cohesive
PR, stopping at demo without merge. D1/f871a8 retains admission/parser ownership.
The active Graph owns subsequent gate, publication and demo steps.

## Design and deletion cut

Typed Rust card definitions in `managym/src/cardsets/` own current characteristics,
programs and presentation. An offline Rust exporter serializes this source; the
existing Python compiler resolves symbolic references, validates, and emits the
checked-in native/learning IR. There remains one runtime lowering path. Ordinary
setup decklists, independent qualification decisions and frozen evidence remain data.

Delete — do not maintain: hand-authored `current/definitions.json`,
`native_roots.json`, `token_templates.json`, their loader paths, authoring docs,
parity source-manifest reference, and the JSON-only unregistered-definition smoke. Independent Avatar Enthusiasts/Glider Kids rules witnesses and the shared
Command/learning/Academy proof remain on the surviving Rust source.
Replace them with one generated catalog exported from Rust, including membership.
Preserve exact program/source hashes where behavior is unchanged, all historical
artifacts, setup counts, independent tests and the public ContentPool API.
No stringly JSON macros or runtime Oracle parsing in set definitions.
The deletion cut is complete; no predecessor authoring paths remain to maintain.
Shared typed helpers express the ten draw lands, five basic lands, single targets,
mana-only costs, modal mana alternatives, prowess and unconditional firebending.
Program constructors live beside their type; named mechanics share their trigger
and duration contracts while retaining each card's program key. Keep conditional
firebending, unusual target ranges and costs
explicit, and retain optional-field presence because source hashes distinguish
omitted fields from empty or false values. The reduction preserves catalog bytes
and all downstream identities.

Main resolution repair `7d7736c6a` is integrated at `f674bcd8f`. Allies at Last
now has independent outcomes on that integration. World w16/schema 8 remain;
five other T1 holdouts retain T2/T3 prerequisites. Full TLA is not this PR.

Implemented: 140 typed Rust definitions with 95 native roots and nine fixed
tokens export byte-equivalent source values; all prior IR and coverage identities
are unchanged. JSON authoring/dependent loader paths are removed. Source and
membership export to checked-in `generated/cards.json`; the CLI compiles Rust
before generating/checking every projection, and CI provisions Rust for that step.
Allies at Last now qualifies on integrated resolution semantics (86 cards total).
Five other T1 holdouts retain their reviewed prerequisites.

Remaining: Graph-owned gate/publication/demo. Gate owns full debug/Clippy, current-world
arena/evaluation/monitoring/model suites and choice-navigation/Learn/browser
receipts. The existing draft PR394 is updated by the publication step, without merge.
Prior implementation and evidence detail: `f674bcd8f:scratch/892c47-author-tla-cards.md`.

Check: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 --test rules_tests tla_content` — 37 passed, plus the focused `prowess_resolves_above_the_noncreature_spell` test; `CARGO_PROFILE_DEV_OPT_LEVEL=1 uv run python -m scripts.compile_current_content --check` — catalog and projections unchanged; Rustfmt/whitespace passed; cp312 release extension rebuilt and atomically installed, pool import retains w16 and 86 eligible cards. Reuse implementation's 13 focused Python passes and source-pin checks; full affected suites and browser acceptance remain with gate.
