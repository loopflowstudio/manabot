# Versioned Python releases

Manabot ships as one PyPI distribution, **`manabot`**, containing both the Python
training library and the native engine at **`manabot.managym`**. There is no
separate managym distribution, dependency, publisher or upload. `import managym`
remains a compatibility alias to the same module and native/protocol types.
New code can use `from manabot import managym`.

Releases use `X.Y.Z`, tagged `vX.Y.Z`; Python metadata and the bundled Rust crate
use the same version. The first planned release is **0.2.0**, including Etude SDK
0.2.0 from [#349](https://github.com/loopflowstudio/manabot/pull/349). Package
versions do not replace native world, content, observation or checkpoint
identities. An upgrade does not make old checkpoints compatible. These are
research releases, not certification of playing strength or full rules coverage.

## Install a pinned release

After publication, outside a checkout:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python --torch-backend cpu 'manabot==0.2.0'
uv run --no-project manabot --help
uv run --no-project manabot train
uv run --no-project python -c 'from manabot import managym; print(managym.WORLD_VERSION)'
```

For Linux NVIDIA training, replace `--torch-backend cpu` with
`--torch-backend cu128`. Torch `==2.10.0` allows its `+cpu` and `+cu128` builds.
A plain PyPI install uses PyTorch's platform default; select the backend
explicitly to avoid an unintended CUDA installation. CUDA needs a compatible
GPU and driver. CI validates CPU execution, not GPU execution.
See [PyTorch installation](https://pytorch.org/get-started/locally/).

| Component | Supported targets |
|---|---|
| Python | CPython 3.12 (`>=3.12,<3.13`), no free-threaded ABI |
| Linux | x86-64, glibc 2.28+; CPU or CUDA 12.8 Torch |
| macOS | Apple Silicon, macOS 14+ for the training stack; CPU Torch |
| Other targets | Windows, Intel Mac, Linux ARM and musl are not supplied or release-tested |

Every Manabot wheel contains the compiled engine, so installation needs no Rust
toolchain. There is one wheel per supported platform and one source archive.
Building the sdist requires Rust and a C linker; it retains Rust sources and
semantic content in their original relative layout. Other source-build targets
are unsupported. Use `--only-binary :all:` to fail rather than attempt a native
build when no matching wheel exists.

The distribution includes training/evaluation modules, engine decision/protocol
models, runtime templates, semantic schemas/IR/source data, and Academy's Python
API, fixtures and curated pack metadata. Optional extras:

- `manabot[academy]`: FastAPI/Uvicorn for the backend. The browser frontend,
  Node build, card art and hosted services are not distributed; use the
  repository's `./scripts/play` for the complete browser experience.
- `manabot[evaluation]`: Matplotlib for evaluation visualizations. Ordinary
  checkpoint evaluation is included in the base distribution.
- `manabot[reports]`: Matplotlib and Etude's S3 support for reports.
- `manabot[artifacts]`: boto3 for artifact storage; credentials are separate.
- `manabot[notebook]`: interactive notebook tooling.

Core library calls, built-in training presets, simulation and JSON `--regime`
inputs are package interfaces. `--regime-name` loads `experiments/recipes.py`
and requires a source checkout. Experiment drivers, source-bundle remote
submission, research receipt generation and repository/data-ledger utilities
also require their source tree and retained artifacts. Never change a running
experiment's pinned source or controller to install a release.

## Build and verify

From a clean checkout with uv, CPython 3.12 and Rust:

```bash
uv run --no-project python scripts/release/check.py
uv run --no-project python scripts/release/prepare.py .runs/release/staging
uv build --no-sources .runs/release/staging --out-dir dist
uv run --no-project python scripts/release/validate.py dist
uv run --no-project --with twine twine check --strict dist/*
uv run --no-project --with pytest pytest --confcutdir=tests/infra tests/infra/test_release.py -q
```

Use a fresh staging path and empty output directory for each candidate. Staging
refuses to overwrite an earlier build and copies tracked files with their
current bytes; commit and review before tagging. It excludes experiment drivers,
build products and frontend dependencies. Both `uv build` and CI's
`uv run maturin build --sdist` build wheels from the source archive, so checkout files
cannot conceal missing content.

The root project is also the development environment. Its uv groups select
separate Torch CPU/CUDA indexes. Staging adds the public runtime requirement
from `[tool.manabot.release]` to the release metadata; published artifacts must
use this preparation step. The resulting sdist has complete runtime requirements
and can be rebuilt directly with a standard Python build frontend. `uv sync`
builds the bundled extension for development; existing explicit
`uv run --extra play maturin develop --manifest-path managym/Cargo.toml` commands also use the root
Manabot project and produce the nested extension.

Clean-install proof, with absolute paths:

```bash
uv venv --python 3.12 /tmp/manabot-release-venv
uv pip install --python /tmp/manabot-release-venv/bin/python --only-binary :all: \
  --torch-backend cpu "$PWD"/dist/manabot-*.whl
cp scripts/release/smoke.py /tmp/manabot-release-smoke.py
cd /tmp
export PATH="/tmp/manabot-release-venv/bin:$PATH"
unset PYTHONPATH
uv run --no-project --python /tmp/manabot-release-venv/bin/python manabot-release-smoke.py
```

This verifies that no separate managym distribution is installed, both import
paths share the same engine, semantic catalogs bind, Torch gradients work,
and the CLI completes a tiny CPU training update outside the checkout. CI
repeats this on Linux/macOS, then installs Academy/reports/artifacts/evaluation
extras and imports their entry points. It contacts no experiment services.

## Publish

`.github/workflows/release.yml` builds and validates relevant PRs, manual runs
and version tags. Only tags publish. Tag builds reject inconsistent versions,
a commit not on main, existing release versions and registry/network failures.
Both platform installations and the artifact inventory must pass. The publishing
job alone receives OIDC `id-token: write`; no PyPI API token is stored, and
PyPI's action produces attestations for uploaded files.

Configure one trusted publisher for `manabot`:

| PyPI field | Value |
|---|---|
| Project name | `manabot` |
| Owner | `loopflowstudio` |
| Repository name | `manabot` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Jack Heart reported submitting the pending publisher on 2026-10-21. Before tagging,
verify the publisher record and configure the GitHub `pypi` environment with
required reviewers and version-tag deployment restrictions. The public PyPI
JSON endpoint returned 404 during preparation; that does not establish account
ownership or publisher configuration. The environment API also returned 404;
its protection setup remains unverified.
See [pending publisher setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

For each release, update `pyproject.toml`, Cargo.toml/Cargo.lock, `uv.lock`,
examples and release notes. Review and merge the release PR, confirm the complete
workflow is green, then create/push `vX.Y.Z` at that reviewed main commit.
Review the `pypi` job's artifacts and approve it through the repository release
procedure. PR publication alone does not authorize a tag or PyPI upload.

Failed builds/smokes upload nothing. An interrupted upload fails visibly;
`skip-existing` is deliberately disabled. Inspect which files reached PyPI,
retain artifacts/digests and prepare a new patch version if necessary. Do not
blindly retry publication, delete a release or move a used tag. If no files
reached PyPI, resolve the cause and rerun the reviewed tag workflow. After
upload, install from PyPI without local artifacts on both targets, confirm the
version and attestations, and retain the release URL and check results.
