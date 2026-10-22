# Semantic kernel conformance — w7

Recorded on 2026-10-20 for MTG-161. Eight terminal games contain 1,107 commands:
four inherited UR/GW seed probes use the general native content pack; four
Elemental probes use the independently compiled `elemental-combat-v1` pack in
both seat orders. Each tape binds its own source and content digests. The
aggregate evidence's source/content descriptor identifies the inherited
UR/GW comparison; it does not replace the per-tape bindings. The manifest
explicitly binds w7, and checkpoints use match-state hash v4.

The two Elemental validation rosters split all 42 owned cards across seats,
with five copies of each basic per roster. They exercise admission without
adopting the final ten product decks. Full native/compiled definition parity
and directed card behavior are checked separately; see
[Elemental contracts](../../docs/rules/elemental-combat.md).

The harness compares explicit singleton execution with trivial-step collapse.
It shares the same rules primitives and is not an independent Magic engine.
The inherited 15-row Phase matrix retains its pinned historical source
comparison; it makes no new external-engine claim for Elemental cards.
Earlier corpora and frozen receipts remain unchanged and require their own
historical runtimes. This corpus does not certify release-browser behavior,
training quality or strength.

```bash
cargo test --locked --manifest-path managym/Cargo.toml --test semantic_conformance_tests
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- check --root conformance/semantic-kernel-w7-v1
cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- fuzz --root conformance/semantic-kernel-w7-v1 --seed 24301 --cases 32 --max-commands 512 --failure-dir target/semantic-conformance/failures
```

Bounded fuzzing cycles through both inherited and both compiled Elemental
orientations. Failure receipts retain the actual orientation's source and
content identities. `check` reproduces evidence without rewriting it. Create a
separate corpus for the next accepted rules world before running `record`.
