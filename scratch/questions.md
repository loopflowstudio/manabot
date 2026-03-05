# Questions

## Wave name mismatch

The worktree is `manabot.mtr` (wave = `mtr`) but there's no `wave/mtr/` directory.
The recent commit history and content suggest this worktree works on the `rules` wave (`wave/rules/`).
Assumed `mtr` maps to `rules` and proceeded with picking from `wave/rules/`.

This caused `lf ops ingest` to fail. Either rename the wave directory or the worktree to match.

## Verification tooling missing in this environment

`cargo test` was run and passes for `managym`.

The design-doc "done when" Python checks could not be run here because `pip`
and `pytest` are not installed in this sandbox (`zsh: command not found` for
both commands). Training smoke remains unverified in this environment.
