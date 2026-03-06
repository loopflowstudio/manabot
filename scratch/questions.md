# Open Questions / Blockers

- `pytest tests/env/` could not be executed in this sandboxed run:
  - `pytest` binary is not installed in PATH.
  - `uv run pytest tests/env/` failed due sandbox/runtime issues (`Operation not permitted` on default UV cache and then a uv runtime panic when redirected to `/tmp`).
- Assumption: Rust test suite (`cargo test`) is sufficient for this headless implementation pass; Python env tests should be run in a fully provisioned local dev environment.
