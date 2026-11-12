# v0.2.2

The rules engine is now its own distribution. `managym` holds the compiled
engine and `import managym`; `manabot` is a pure-Python training library that
requires exactly `managym==0.2.2`. Both are released together at one version.

Install the engine alone, without Torch, Gymnasium or Etude:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python --only-binary :all: 'managym==0.2.2'
```

Install training, which brings the matching engine:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python --only-binary :all: --torch-backend cpu 'manabot==0.2.2'
uv run --no-project --python .venv/bin/python manabot --help
```

Engine wheels target ordinary CPython 3.12 and 3.13 on Linux x86-64
(glibc 2.28+) and Apple Silicon (macOS 14+); no Rust toolchain is needed.
Use `--python 3.12` to select the other supported interpreter. See the
[engine installation guide](https://github.com/loopflowstudio/manabot/blob/v0.2.2/docs/engine-python.md)
and the [installation guide](https://github.com/loopflowstudio/manabot/blob/v0.2.2/docs/development.md#install)
for extras, backend selection and dependency troubleshooting.

## Imports

`import managym` is the engine. `from manabot import managym` and
`manabot.managym` remain a compatibility alias to the same classes and
submodules, so existing code keeps working.

## Upgrade from 0.2.0 or 0.2.1

Earlier `manabot` wheels bundled the engine and own `managym/__init__.py`.
With pip, uninstall first: `pip uninstall manabot`, then install 0.2.2.
Removing the old wheel after the new engine is installed (what pip does in a
one-step upgrade) deletes that file. If `import manabot` then reports missing
`managym` package files, run `pip install --force-reinstall 'managym==0.2.2'`.
uv upgrades in one step without this problem.

## Release workflow

One tag builds both packages at the same version: four engine wheels, one
pure-Python training wheel and two source archives. Both inventories are
validated before publishing the engine, then training. After both PyPI uploads,
the workflow creates a GitHub Release from these notes and adds both PyPI
links. Uploads are not atomic across projects; completion checks the retained
seven-file inventory and hashes. A complete engine upload with matching hashes
can resume a missing training upload; partial files within either package or
conflicting bytes stop for inspection.

The current world is still open **w17**. Package versions do not override
world, content or checkpoint bindings. The
[0.2.1 release notes](https://github.com/loopflowstudio/manabot/blob/v0.2.1/release/v0.2.1/NOTES.md)
cover the API migrations since 0.2.0.
