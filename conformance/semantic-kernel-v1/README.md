# Semantic kernel conformance v1

This checked evidence covers the curated UR Lessons versus GW Allies boundary
from `content/semantic/v1/two_deck.source.json`. The source currently resolves
to 41 UR cards and 40 GW cards; the harness reads it directly rather than
copying another deck constant.

The readable `reference/explicit-step-v1` executor publishes and applies every
singleton action. The production `optimized/skip-trivial-v1` executor collapses
those forced actions. After every external semantic command, the harness
compares the deterministic match hash, legal prompt, pending choice, event
boundaries, and terminal result. Both paths share low-level rules and card
primitives, so this is not a second Comprehensive Rules implementation.

## Reproduce

```bash
cargo test --locked --manifest-path managym/Cargo.toml --test semantic_conformance_tests
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- check --root conformance/semantic-kernel-v1
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- fuzz --root conformance/semantic-kernel-v1 --seed 24301 --cases 32 --max-commands 512 --failure-dir target/semantic-conformance/failures
```

`check` never rewrites evidence. A reviewed semantic change can regenerate the
checked replay receipts and summary with:

```bash
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- record --root conformance/semantic-kernel-v1
```

`replay` accepts either a checked corpus receipt or a fuzz-failure receipt. Fuzz
failures contain the exact source/content and executor identities, game seed,
choice seed, semantic command tape, last matching checkpoint, failing step, and
error; reproduction requires every field to match. Reproduce one with:

```bash
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- replay target/semantic-conformance/failures/<case>.json
```

Or replay a checked case directly:

```bash
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- replay conformance/semantic-kernel-v1/replays/ur-vs-gw-5eed.json
```

The Phase matrix is an offline source comparison pinned exactly to
`553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d`. It records normalized matches,
known mismatches, and explicit exclusions. Blocking CI does not clone or build
Phase and makes no whole-engine parity claim.
