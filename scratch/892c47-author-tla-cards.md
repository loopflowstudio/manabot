# TLA Rust authoring — reviewed, 2026-10-28

Jack Heart accepted clear typed Rust set files as the single card-authoring source,
minimal configuration, and checked-in generated representations. This supersedes
JSON authoring; see [the accepted direction](tla-rust-authoring-direction.md).
Task 892c47 updates existing PR394 and stops at demo. Merge, auto-merge, paid
compute, training and follow-up Task launches are not authorized.

## Implemented

Commit `0a0d22577` contains 140 typed definitions in `managym/src/cardsets/`,
95 native roots and nine fixed tokens. A Rust exporter produces checked-in
`generated/cards.json`; the existing compiler validates symbolic programs and
emits the native/learning IR. Set files own rules, characteristics and presentation.
Shared typed helpers remove repeated land-cycle, cost, target and mechanic code.
The removed hand-authored definitions/native-roots/token-templates JSON and their
loaders are not maintained. There is one native runtime. Setups and independent
qualification evidence remain data. Historical artifacts remain frozen.

The Rust migration preserved prior source values and IR/coverage identities.
Current execution is w16/schema 8. Main's resolution repair `7d7736c6a` was merged
at `f674bcd8f`; independent Allies at Last witnesses clear its prior blocker.
The native pool now admits 86 cards. ContentPool/Eligibility/pool_content and
Python exports remain stable for D1/f871a8, which retains parser, construction,
admit_deck and consumer-input ownership. Authored availability is not arbitrary
pool admission. T1 retains five unsupported cards with explicit prerequisites:
Accumulate Wisdom (T2 ordering), Suki and Water Tribe Rallier (T3 characteristics),
and Cat-Gator/Lightning Strike (T3 target classes). All 286 card identities and
294 faces retain their accepted batch owners; neither T1 nor full TLA is complete.

## Review and recovery

Reconciliation and concept review confirmed the accepted Rust ownership model;
no new authoring choice is required. A stale-output diagnostic incorrectly named
the historical compiler. The current catalog, IR, coverage and eligibility paths
now name `uv run python -m scripts.compile_current_content`; historical callers
retain their original command. Isolated stale-output/recovery checks passed for
all four current artifact kinds and historical behavior.

Generation is sequential: a later failure can leave derived files updated.
Repair the source/evidence, regenerate completely, then check freshness before
rebuilding consumers. Rust export failure precedes catalog writing; missing
references fail compilation; edited qualified clauses reject stale qualifications.
Do not mechanically refresh qualification hashes instead of reviewing behavior.
Native pool loading rejects eligibility/IR mismatch. No atomic-generation promise
or additional registry is needed.

The original Graph stopped on an ignored `.lf/tmp` sync-pathspec bug, then an
agent-startup routing error. Manual sync succeeded. Direct realign and concept
review subsequently completed. Their results, rather than the stopped route,
support the remaining publication assessment. No Graph schedules the next steps.
The selected engine Wave GOAL/MEMORY are absent; none was edited.

## Evidence and publication boundary

Retained checks at `0a0d22577`: 37 focused TLA debug tests plus prowess ordering,
13 Python consumer checks, Rust-export freshness, formatting, source pins and an
atomically rebuilt cp312 extension. Reconciliation verified ten raw source pins.
The current pass ran 28 current-content/coverage tests, isolated generation
recovery checks, Ruff and whitespace checks. A real native descriptor read returned
w16, 86 eligible cards, eligible Allies at Last/Water Tribe Captain, and unsupported
Water Tribe Rallier. The baseline Command/fork/learning/Academy proof serializes
real observations; it is not an observed browser walkthrough.

Focused evidence and completed review satisfy the selected pre-demo publication
boundary. Update PR394 around Rust authority, 86 qualified cards, five holdouts,
and these proof limits; then present demo here. Gate/CI still owns full debug
Rust/conformance and Clippy, arena/evaluation/monitoring/model suites, choice/Learn
and browser receipts, CI collection and release-source checks. Exact commands are
in `docs/development/validation.md`. D1 integration proofs and its named-format
choice remain D1/full-file work, not content-publication approval requirements.

Check: `uv run --no-sync pytest tests/semantic/test_current_content.py tests/semantic/test_coverage_gate.py -q` — 28 passed; isolated recovery, Ruff and whitespace passed. Broader gate/CI checks remain unclaimed.
