# PR D: import decklist files into shared human/bot play

Accepted by Jack Heart on 2026-10-28 as the separate parallel decklist PR within an eight-PR TLA delivery. Automatically implement, test, refine and publish one cohesive PR, then stop for Jack Heart's review. No automatic merge, paid compute, training or follow-up launches. Source exploration: loopflow://manabot/task/0d9ba3. This brief governs implementation; copied research is evidence, not a competing research-only instruction.

Outcome: a person can supply a decklist file and use a valid novel deck combination with Academy and manabot through one native setup/admission contract. Begin with a qualified supported pool; expanding TLA support expands eligibility without changing the parser. Curated-only matchup matching is not an acceptable final path. A parser disconnected from playable setup is not completion.

Define a documented, small text dialect supporting counts/card names and main/sideboard sections, retaining original source and line references. Normalize into named typed structures, resolve against the selected versioned pool, and return actionable line-level unknown/ambiguous/unsupported/count/format issues without silently substituting cards or discarding source. Card legality and format validation are shared domain behavior, never browser-only checks. No network card lookup at runtime. Both consumers use the same admitted setup/content identity. Preserve source on errors so correction is possible.

Inspect existing deck construction rules and expose them as an explicit named, versioned ruleset. Specific tournament-format selection is not settled: do not assert Standard legality or silently invent tournament restrictions. Use an explicitly labeled development format based on established repository rules for the initial end-to-end proof; isolate construction policy so review can select a product format without changing parsing or native rules. Main/sideboard parsing does not authorize a sideboarding UI, retail draft/sealed collation, Commander or multiplayer. Deliver a concise Academy file-import flow and ordinary manabot configuration/CLI entry point.

Parallel PR1 owns content authoring, semantic IR/runtime lowering, definition eligibility and content identity. This Task owns parsing, normalized deck models, shared pool/format admission, and setup entry points. Consume a small typed interface of resolved definition IDs, versioned content/world identity and eligibility; do not build a second rules registry or rewrite cardsets. Agree contract details early and integrate current reviewed upstream changes; initial fixtures can use the existing qualified supported pool. Admission must reject missing programs/faces/tokens and unsupported cards rather than falling back to native generic definitions. Existing curated setups remain presets.

Proof: import a valid deck combination absent from authored setups and start it through both consumers; show the same resolved identities and legal action authority. Exercise malformed lines, bad quantities, ambiguous/unknown names, known unsupported TLA cards, sideboard/format violations, and retained editable source. Preserve Command/reconnect/replay/fork binding and viewer safety. No policy training is necessary. Keep model/checkpoint compatibility explicit; do not silently accept a checkpoint incompatible with the chosen content/observation schema.

Overlap check 2026-10-28: PR389, Task 19d240, rewrites shared evaluation and simulation consumers including manabot/sim/native_game.py, match.py, serving.py, semantic/policy.py and Academy watch/villain. PR383, MTG-175, owns adaptive curriculum changes in vector_env.rs, vector_env_bindings.rs and sim/rollout.py; its reserved curriculum decisions stay there. Recheck these before editing overlapping entry points. MTG-177 owns target-resolution changes; this importer must not repair those independently.

Use repository conventions and validation.md. All Python through uv; debug Rust with two Cargo jobs and atomic extension replacement where needed; frontend generation/check/build before browser proofs. No artificial load, random search, heavy benchmarks, paid services, frozen-evidence rewrites or new training. Include a simple real import demo and focused independent error/admission tests in the final review.

## Implementation — 2026-10-28

Content contract coordinated with Task 892c47: native `content_pool(pack_key)`
provides world/content/compiled identity and eligible/unsupported definition rows.
That Task owns eligibility and closure; importer owns setup admission and parsing.
PR389 and PR383 remain open; no edits planned in their reserved files.

Delete — do not maintain: replace explicit-pack authored-setup-only selection
in `semantic/content.rs` with shared qualified pool admission. Preserve implicit
preset selection and checkpoint exact-world binding. No card authoring deletions
belong to this Task.

Remaining: typed parser/source diagnostics; native construction admission; CLI
file options; Academy import/editor and retained setup identity; focused checks.
Historical full-TLA/audit notes and duplicate handoff preserved in
`.lf/tmp/decklist-evidence/decklist-research-20261028.tar.gz`; source exploration 0d9ba3,
base c5d406e3d2a701525c15c8500a30f7da7896cc40.
