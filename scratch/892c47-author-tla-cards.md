# TLA content implementation — reconciled 2026-10-28

Accepted outcome: [tla-content-handoff.md](tla-content-handoff.md), Jack Heart,
2026-10-28. One cohesive PR for review at demo, no merge or later Task launches.
The eight-PR allocation remains in [tla-delivery-batches.md](tla-delivery-batches.md).

Implemented at `2a44b763c`: current definitions and setup projections lower to one
native runtime and learning source. All nine fixed token templates are explicit.
World w16 and learning schema 8 are fresh; frozen evidence and setup counts stay
unchanged. Qualification hashes cover transitive card/token closure. IR owns IDs
and membership; generated eligibility retains inventory and qualified keys per IR
hash. Policy binding reads the descriptor without starting a game.

## Delete — do not maintain

Completed: handwritten TLA registrations in cardsets/tla.rs and TLA portions of
elemental.rs, sacrifice.rs and temur.rs; registry-only presentation lookup in
semantic/content.rs; exclusive native/compiled equality tests. Preserve non-TLA
rules and independent directed witnesses on the surviving typed source.
Attack legality reuses can_tap's haste-aware restriction; temporary haste never
clears sickness. Existing typed look/select carries Rallier's required reveal.
The repeated Python pack-path tables and policy receipt comparison are removed:
CURRENT_IR_PATHS selects explicit current catalogs and BoundSemanticPack checks
pack key plus both source/IR hashes for every compiled binding. Policy retains
its required-receipt and full-definition coverage checks. The required-receipt
guard rejects null or malformed provenance before generic observation binding,
which still permits uncompiled manifests. Receipt tests mutate the native descriptor
at that boundary rather than replacing a successfully validated binding.
Directed card tests use effective_pt for paired assertions; learning tests check
the complete shared compiler vocabulary separately from each pack's used opcodes.
Pool descriptors now create implemented entries directly and add only missing
inventory entries; both public accessors share the same runtime lookup. Current
projections assemble their closure once. Qualification mutation tests share one
fixture while preserving direct-card and transitive-token cases. Surviving
Lotus Cobra and Treasure registrations reuse the existing modal-mana constructor,
retaining color order and payment behavior. No deletion targets remain in this cut.

## Clause review

All 60 previously unqualified T1 cards now have a disposition in
`docs/rules/tla/t1-clause-review.json`, checked against final pinned Oracle/rules
and release notes. 54 gain complete printed-clause evidence (existing independent
witnesses plus focused supplements); the qualified pool grows from 31 to 85.
Rallier now reveals its selected card. Wartime Protestors' temporary haste now
enables attacks and tap-symbol costs without erasing summoning sickness.

Six stay unsupported with concrete shared prerequisites, retaining T1 completion
ownership rather than silently handing qualification to T7:
- Accumulate Wisdom: ordered private library remainder (T2).
- Suki, Kyoshi Warrior: CDA power outside the battlefield (T3).
- Water Tribe Rallier: effective library power filtering, demonstrated by Suki;
  reveal/payment/ordinary selection fixed or witnessed (T3 authority, T2 consumer).
- Cat-Gator and Lightning Strike: planeswalker/battle damage-target classes under
  CR 115.4, absent from CreatureOrPlayer (T3 characteristics/target authority).
- Allies at Last: illegal victim still receives power damage when another target
  remains valid (MTG-177; same primitive as Rocky Rebuke).

The 91-card T1 allocation is reviewed, not fully delivered. Prerequisite owners
are existing plan boundaries, not newly launched work or a cyclic dependency on
landing the entirety of T1. Eligibility remains closed until accepted integration
and remaining whole-card witnesses. No partial card is qualified.

Five earlier corrections remain: Foggy Swamp Hunters to T3; Pirate Peddlers, Zhao,
Vindictive Warden and Rocky Rebuke to T4. MTG-177's last observed status reports
gate running; retained PR354 does not establish acceptance of its later w15 repair.
No resolution repair was copied. Full-file ownership still covers 286 identities
and 294 faces. All ten final raw source byte digests were rechecked this pass.

Parallel f871a8 retains admission/parser/consumer ownership. Its agreed pool API
and checkpoints remain intact; latest compressed integration and joined consumer
proof are D1 work. No Task, Graph, publication or merge was launched here.

Qualification evidence includes 102 passing exact-filter debug witnesses plus
Rallier reveal and haste controls (19 new directed tests), and verified source pins.
Check: `cargo test --manifest-path managym/Cargo.toml --test rules_tests rules::tla_content` — 35 passed; `uv run --extra dev pytest tests/semantic/test_current_content.py tests/model/test_policy_catalog.py -q` — 23 passed after atomic release-extension rebuild; `uv run --extra dev pytest tests/semantic/test_learning_projection.py tests/semantic/test_learning_projection_env.py -q` — 12 passed; generator `--check`, Rust format, Ruff and `git diff --check` passed.
Receipt follow-up: `uv run --extra dev pytest tests/model/test_policy_catalog.py -q -k 'receipt or catalog_uses_actual'` — 7 passed after three malformed-receipt cases reproduced the cleanup regression; focused Ruff and staged/unstaged whitespace checks passed.
Reduction check: `cargo test --manifest-path managym/Cargo.toml --lib semantic::pool::tests` — 2 passed; `cargo test --manifest-path managym/Cargo.toml --test rules_tests rules::sacrifice_cards` — 21 passed; after atomic release-extension rebuild, `uv run --extra dev pytest tests/semantic/test_current_content.py tests/model/test_policy_catalog.py -q` — 27 passed and all 10 native pool descriptors match their pre-reduction snapshots exactly; generator `--check`, Rust format, Ruff and whitespace checks passed.
Gate retains full debug Rust/Clippy, current-world
arena/evaluation/monitoring, model suite, choice-navigation,
Learn/browser proofs on new receipts, CI collection and release-source checks.
Publication and demo remain later; historical receipts are unchanged.
