# MTG-286 decisions and evidence — 2026-11-11

Sync resolution: retain this branch's active TLE notes when main deletes the
inherited release-notes file; main's release and licensing changes apply unchanged.

Jack Heart: "lets file a task for covering all of TLE".
Inherited MTG-285 release notes remain in this file at checkout base
`4c041d6cefe829c5b1b52273ba19991d663ee26b`; they do not direct this Task.
The live plan is `scratch/mtg-286-define-every-tle-card-so-tla-limited-pools.md`.

Reversible assumptions: retain the existing 1v1 development world with correct
no-commander behavior. Full Commander/multiplayer setup remains a demo question,
not a prerequisite or a reason to narrow the 242-name inventory. Retain the
shipping guide's planned-support labels and Scout's Warning example until the
complete implementation demonstrates them.

Earlier documentation-only work: the inventory matched 242 fetched names,
8 defined and 234 missing; syntax/fence/link checks and `git diff --check`
passed. No runtime or corpus proof was claimed. `lf open --host desktop --file
docs/engine.md` reported "No Desktop Files view attached; nothing opened."
Original uncommitted notes were retained before compression at
`/var/folders/m6/r3tllnrs1yq7yfbwm680tss40000gn/T/mtg286-compress-ju2m728q/`.
Realign observed `origin/main` at `4c041d6cefe829c5b1b52273ba19991d663ee26b`,
no Wave or PR, and no further Task comments beyond the claim. Runtime sources
then matched that base; this implementation pass changes them.

## Implementation evidence

The new `managym/cards/tests/fixtures/tle-inventory.jsonl` preserves all 242
names, Oracle IDs/paragraphs, TLE printing IDs and first-printing IDs/sets.
Sources: [TLE search](https://api.scryfall.com/cards/search?q=set%3Atle&unique=cards&order=name)
and Scryfall exact-name print searches ordered by release ascending, fetched
2026-11-12 UTC. Requests initially hit HTTP 429; retained successful responses
and sequential slower exact-name searches completed the fetch. Existing eight
cards keep their definition printing IDs. Noxious Gearhulk uses KLD rather than
its simultaneous MPS edition; Three Dreams records its earlier PSAL printing.
This is external source data, not a complete clause allocation or admission list.
SHA256: `334e003532fcc4d3426086da033a02356715faeb7c892279978ba40a394a327d`.

The Cabbage Merchant is implemented with counted tap costs and both printed
triggers. The first competing-cost test exposed an offered Food that was
reserved for its own activation; checking reserved tap availability before
selection fixed it. The test now proves five Foods pay the Food activation
through two Merchant activations, whereas four cannot. Source-tapped behavior,
three/four Food counts, waterbend exclusion, and Command replay/stale rejection
also pass. Observation continuation remains acting-viewer-only, consistent
with the existing privacy contract; committed permanent taps are public.

Validation (focused, debug unless stated):

- `cargo test -j 2 --manifest-path managym/Cargo.toml -p managym-cards --test cards merchant_ -- --test-threads=2` — 12 passed: seven TLE behavior tests, one new learning-token test, four existing Merchant-named tests.
- Same binary, filters `boiling_rock_rioter_exiles_with_itself_and_casts_your_exiled_allies`, `barrels_of_blasting_jelly_filters_one_mana_once_each_turn`, `committed_optional_payment_resumes_after_two_nested_mana_decisions`, `every_trigger_condition_and_ability_reaches_the_learning_tokens`, `badgermole_cub_earthbends_and_adds_green_when_a_creature_taps_for_mana` — 5 passed.
- `cargo test -j 2 --manifest-path managym/Cargo.toml -p managym --lib registered_mana_activations_consume_finite_payment_resources -- --test-threads=2` — 1 passed.
- Required root `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — built cp313 macOS arm64 wheel; atomically replaced the matching `_managym.cpython-313-darwin.so`.
- `uv run --extra dev --extra play pytest tests/verify/test_observation_parity.py -k merchant_tap_count -q` — 4 passed (v5, v6, v8, v9); native allocation, preallocated buffers, Python reference and LLM text agree at both tap selections. Foods are created by real Unlucky Cabbage Merchant casts from the existing priority-root fixture, not injected token deck entries.
- Changed Python Ruff check/format, workspace formatting and `git diff --check` passed. Offline inventory validation matched the original 242-name list, unique IDs and 9 current literal definitions; 233 remain missing. Scratch stays within both context limits.

The new production counterexample is recorded in the plan: Heartbeat needs
actual produced types, either player's triggering sources and output identity
that `ManaChoice` currently discards. Animal Attendant needs spent-mana entry
provenance absent from current pools. The implement instruction says:
"If implementation turns up a counterexample that breaks the slice, the data
model, the deletion path or the overall design, stop dependent work and revise
the design or send it back to review." This stops dependent production work;
it does not remove all-card scope or mark the Task complete. No Graph transition,
publication, corpus fetch or corpus admission proof occurred.

Remaining: full clause allocation; production/observation contract review;
233 definitions and their mechanics; vector-run/Academy pending-choice proofs;
world integration/current receipts; full gate suites and the 1,000-event corpus
admission proof. The artifact corpus is in the manifest, not initially local;
fetch through Doppler at that acceptance step. Frozen evidence stays unchanged.

## Compression — 2026-11-12 UTC

Pre-edit uncommitted notes retained at
`/var/folders/m6/r3tllnrs1yq7yfbwm680tss40000gn/T/mtg286-compress-xijd7yuo/`.
The inventory table now points to the complete JSONL instead of duplicating
242 names; baseline evidence and all external identity bytes are retained.
The plan describes the surviving resource model and shared selected-tap witness;
all-card scope, production-design blockers and acceptance obligations remain.

Validation of compression:

- `cargo test -j 2 --manifest-path managym/Cargo.toml -p managym-cards --test cards merchant_ -- --test-threads=2` — 12 passed in debug after the final code edits; the same binary's `boiling_rock_rioter` and `committed_optional_payment_resumes_after_two_nested_mana_decisions` filters each passed.
- `cargo test -j 2 --manifest-path managym/Cargo.toml -p managym --lib registered_mana_activations_consume_finite_payment_resources -- --test-threads=2` — 1 passed.
- `cargo fmt --manifest-path managym/Cargo.toml --check`, changed-Python Ruff and `git diff --check` — passed. Offline comparison verified all 242 names against the historical table and the unchanged JSONL SHA256.
- Root `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed; matching cp313 macOS arm64 extension installed atomically. `uv run --extra dev --extra play pytest tests/verify/test_observation_parity.py -k merchant_tap_count -q` — 4 passed (v5/v6/v8/v9). Full affected suites remain with gate.
