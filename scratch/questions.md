# Open Questions

- Previous note about missing `wave/sps/` was incorrect — the directory exists. The `lf ops ingest` fast-path failed because it resolved the wrong absolute path (looked in `manabot/` instead of `manabot.sps/` worktree).
- Python test execution is currently blocked in this sandbox: package dependencies (numpy/torch/gymnasium/pytest) are unavailable and network access is restricted, so only Rust checks and Python syntax compilation were run locally.
