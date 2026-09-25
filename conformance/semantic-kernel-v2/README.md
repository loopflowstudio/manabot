# Semantic kernel conformance v2

This corpus records the post-Learn rules at the same four seeded scheduling
probes as v1. The frozen [v1 receipts](../semantic-kernel-v1/) remain unchanged;
current source digests, Learn action names and checkpoint hashes are not
retroactive evidence for that earlier world.

The harness deliberately uses the source's main decks with empty sideboards
and the general content pack. It compares explicit singleton execution with
trivial-step collapsing, not complete authored-setup or live/replay parity.
Sideboard retrieval is covered separately by `learn_setup_tests` and the
`flow::decision::learn_tests` tests. The Phase matrix is copied unchanged from
v1; its pinned source comparison is not a new upstream review.

```bash
cargo test --locked --manifest-path managym/Cargo.toml --test semantic_conformance_tests
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- check --root conformance/semantic-kernel-v2
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- fuzz --root conformance/semantic-kernel-v2 --seed 24301 --cases 32 --max-commands 512 --failure-dir target/semantic-conformance/failures
```

`check` reproduces all four receipts without rewriting them. To record a
reviewed rules change, create another versioned corpus before running `record`;
do not relabel historical evidence as current.
