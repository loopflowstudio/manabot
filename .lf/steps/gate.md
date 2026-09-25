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
Frontend service startup and its first HTTP request must overlap a cold native
build, with cleanup on either service's preparation failure. Run the launcher
contract tests when changing startup ordering; a warm local launch cannot
expose serialized installation and frontend compilation.

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

## Local challenger and shared history

For changes to the bounded challenger runner, include
`uv run pytest tests/sim/test_train_challenger.py -q`. Reject nonfinite wall
limits before creating output or launching a worker; never run training just
to test malformed budget arguments.

For history changes, exercise `game-history.spec.ts` against an isolated built
SPA/ASGI instance using `.lf/directions/e2e-ports.md`. Include empty results,
visible request errors and retry, alongside shared/private replay and feedback.
Record local timing separately from production coverage. Preserve failures in
frozen advice evidence as explicit gate blockers; do not regenerate fixtures
or relax historical checkpoint validation to make the suite green.

When history permissions or replay projections change, run the release prompt
matrix too. Match terminal traces by the live table's `attempt_id`; replay
responses omit the deal seed even for participants.
On macOS, `--ignore-snapshots` can verify gameplay and trace lookup, but only
the pinned Linux profile certifies committed visual references.
