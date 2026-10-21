# Semantic kernel conformance — w5

Recorded on 2026-10-20 after MTG-122 registered three vanilla research cards
(Savannah Lions, Glory Seeker, Hill Giant) for its simple L1 pool. Adding them
changes the general content pack's digest, so [w4](../semantic-kernel-w4-v1/)
can no longer replay; it remains unchanged as the earlier receipt.

This corpus uses w4's manifest: the same four seed/seat probes and command
caps. Against w4, all four games make the same 567 commands with the same
actions, action-space and choice digests and outcomes; only the content digest
and the state hashes that include it differ. The Phase matrix is copied
unchanged from w4.

```bash
cargo test --locked --manifest-path managym/Cargo.toml --test semantic_conformance_tests
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- check --root conformance/semantic-kernel-w5-v1
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- fuzz --root conformance/semantic-kernel-w5-v1 --seed 24301 --cases 32 --max-commands 512 --failure-dir target/semantic-conformance/failures
```

`check` reproduces the receipts without rewriting them. For another accepted
rules change, create a separate corpus before running `record`.
