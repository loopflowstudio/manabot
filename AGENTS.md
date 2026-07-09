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
- Play against the bot: `uv run scripts/play.py` (starts backend + frontend,
  Ctrl-C stops both). 