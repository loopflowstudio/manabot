# gate (repo override)

Polish and validate only the current branch scope.

## Required checks for Rust managym work

When `managym/Cargo.toml` or `managym/src/**` changes:

```bash
cd managym
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Notes:
- `cargo test --all-features` can fail in environments without Python dev/link symbols; treat that as environment setup, not core engine correctness.
- Record any unavailable tooling (e.g., missing `pytest`) explicitly in the gate summary.
