# Semantic kernel conformance — w4

Recorded on 2026-09-29 after ETU-88 introduced mandatory cleanup, the legend
rule and three registered UR cards. This corpus uses the same four seed/seat
probes and command caps as [v2](../semantic-kernel-v2/). All four games finish,
with 567 commands total. Earlier corpora and their receipts remain unchanged;
their identities do not certify w4.

The harness uses the source main decks with empty sideboards and the general
content pack. It compares explicit singleton execution with trivial-step
collapsing. It does not test the candidate deck, complete authored setup or
live browser parity. The candidate's eight complete-game Command/replay tests
and directed card evidence are described in the
[ETU-88 record](../../docs/rules/ur-lessons-increment.md).

The Phase matrix is copied unchanged from v2. Its pinned comparison is
historical evidence, not a new upstream review or coverage of the new cards.

```bash
cargo test --locked --manifest-path managym/Cargo.toml --test semantic_conformance_tests
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- check --root conformance/semantic-kernel-w4-v1
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- fuzz --root conformance/semantic-kernel-w4-v1 --seed 24301 --cases 32 --max-commands 512 --failure-dir target/semantic-conformance/failures
```

`check` reproduces the receipts without rewriting them. For another accepted
rules change, create a separate corpus before running `record`.
