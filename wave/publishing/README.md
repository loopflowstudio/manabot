# Publishing

## Vision

Ship `managym` and `manabot` as installable PyPI packages. Today both
require cloning the monorepo and building from source. A researcher or
tinkerer should be able to `pip install managym` and start writing agents.

### Not here

- Rust crate publishing (crates.io) — managym's Rust API isn't stable
- Docs site or API reference
- Package manager integration beyond pip/uv

## Packages

### managym

Pure Rust engine with PyO3 bindings. Ships as a binary wheel.

- **Build**: maturin (standard for PyO3 projects)
- **Wheels**: manylinux (x86_64, aarch64), macOS (arm64, x86_64)
- **CI**: GitHub Actions matrix builds wheels per platform, uploads to PyPI
- **Versioning**: semver, independent of manabot

Open questions:
- Windows support? Probably not initially.
- Minimum Python version? 3.10 matches current `requires-python`.
- How much of the Python-facing API is stable? The observation types
  and env interface are settling, but card-level details are still
  shifting.

### manabot

Pure Python. Depends on `managym` from PyPI (replaces `pip install -e managym`).

- **Build**: hatchling (already configured)
- **Dependency**: `managym>=0.2.0` in `pyproject.toml`
- **Versioning**: semver, coordinated loosely with managym

Open questions:
- Should manabot be installable without managym (with a degraded mode)?
  Probably not — managym is the whole point.
- wandb is currently required. Should it be optional for people who
  just want to run inference?

## Sprints

### 01: managym on PyPI

- Switch managym build to maturin
- CI workflow: build wheels on push to main, publish on tag
- Test: `pip install managym` in a clean venv, import works
- Trusted publisher setup on PyPI

### 02: manabot on PyPI

- Add `managym>=X.Y` to manabot's dependencies
- Remove the `pip install -e managym` step from README
- CI workflow: publish on tag
- Test: `pip install manabot` in a clean venv, `manabot train --help` works

### 03: Version coordination

- Decide on compatibility policy between managym and manabot versions
- Add CI check that manabot's tests pass against the published managym
  (not just the local editable install)
