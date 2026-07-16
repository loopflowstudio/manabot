Never add "Co-Authored by:" tags in commits

## Commands

- **All Python commands run through uv.** This repo's environment is
  uv-managed; bare `python`, `uvicorn`, `pytest`, `maturin` etc. will miss the
  venv or pick the wrong interpreter. Always `uv run <cmd>` (or
  `.venv/bin/<cmd>` when a script must reference the interpreter directly).
  Never emit an un-uv'ed Python command in docs, scripts, or instructions.
- Python is pinned to 3.12 (PyO3 caps at 3.13; the venv is 3.12). Fresh venvs:
  `uv venv --python 3.12`.
- After changing Rust under `managym/src`, rebuild the extension:
  `cd managym && uv run maturin build --release -i ../.venv/bin/python`, then
  place the cp312 `.so` from the wheel at `managym/_managym.cpython-312-darwin.so`.
- Play against the bot: `./scripts/play` (installs locked local dependencies,
  starts backend + frontend, and stops both on Ctrl-C).

## Testing

- **CI runs `cargo test` in debug, so validate in debug before landing.**
  `cargo test --release` alone is not enough: the engine guards its invariants
  with `debug_assert!`, which compiles out of release entirely, so a test can
  pass green in release and still fail CI. `create_token` is the live example —
  it `debug_assert!`s that the name resolves to a token definition, so passing
  an ordinary card name is release-silent and debug-fatal. Clippy does not
  catch this either.
