# gate (repo override)

Polish and validate only the current branch scope.

## Required checks for website work

When `website/**` changes, run from `website/`:

```bash
uv run --python 3.12 --extra test pytest tests/ -q
```

Keep coverage for parseable blog-only sitemap XML, the root redirect, and the
Fly health probe's plain-text `/healthz` response. A URL substring check alone
does not prove that a sitemap is valid XML.

## Required checks for clean-machine startup work

Run the launcher and verifier tests in `tests/etude/test_play_launcher.py` and
`tests/etude/test_clean_machine_verifier.py`, then `./scripts/verify-clean-machine`
with the documented empty-artifact precondition. Preserve existing local build
artifacts outside the checkout before the proof and restore them afterward.
Use unused ports as described in `.lf/directions/e2e-ports.md`.

Keep the external 60,000 ms playable-state budget, offline reload, and session
identity assertions intact. Check the receipt's host and cache conditions:
a local macOS pass does not certify the Ubuntu CI performance profile.
Proof-runner startup should overlap installation after npm is ready, and both
launcher failure and timeout must reap the browser runner.

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
