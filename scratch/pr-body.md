Card vocabulary previously grew dense counter storage and boolean keyword fields, while card definitions lived inside the engine crate. Sparse counters, enum-backed keywords and separate card/rules crates let the catalog grow without adding storage to every permanent or rebuilding the engine for an isolated card check.

## What changes

- Move rules types into `managym-state` and card sets/tests into `managym-cards`, preserving `managym::state` and one definition per card. Update CI selection, release packaging and live rules-audit references for the moved files.
- Preserve counter serialization and serialize only present keywords. Evergreen observation columns and the Python boolean keyword surface remain unchanged; exhaustive guards require an explicit representation for future keywords.
- Replace earthbend's animation flag and crew's specialized modification with continuous-effect records. Use conditional `TypeGrant` for War Balloon and Phoenix Fleet Airship, including source-relative namesake counting. Preserve main's derived-cache invalidation behavior and T5 copy/characteristic semantics.
- Rebind mutable w17 conformance, Academy fixtures and parity receipts. Content identity changes and match-state hash contract becomes 12; learning schema remains 10. Earthbend records allocate IDs. Frozen worlds, checkpoints and historical receipts remain unchanged.

## Checks

- After merging main's cache and performance-CI changes: full debug and release Rust suites each passed 882 tests (2 ignored); formatting and all-target/all-feature clippy passed.
- Exact Python Unit Tests invocation, including `tests/infra`, the clean-machine verifier and `deploy/play/tests`: 103 passed. Website: 21 passed. Live rules-inventory validation passed; the pinned raw TXT was unavailable for source-based regeneration, and frozen audit evidence was retained.
- Rebuilt and atomically installed the CPython 3.12 extension; authored parity, native/Python observation parity and static-token checks passed all 28 tests.
- Before the cache merge: Academy's 7 permanent-note tests, Svelte check/build and 10 isolated headless browser checks passed. The recovery CI run also passed Academy UI, clean-machine Play, protocol/semantic conformance, Rust, Python unit and Academy integration before a newer documentation head superseded it.

Local validation used macOS. Current-head CI, including Engine Budget, is pending; no passing performance-budget result is claimed. The keyword bitset currently supports 64 parameterless variants (14 are defined); the separate capacity-extension proposal is not included. The first non-evergreen keyword still needs a fact representation and Python API migration. This PR is for review against main; auto-merge is not enabled, and T6 integration follows this crate move.
