# Versioned Python releases

Manabot and managym ship together as `X.Y.Z`, tagged `vX.Y.Z`. Manabot pins
`managym==X.Y.Z`; Python metadata and the Rust crate use the same version.
The first planned public release is **0.2.0**, including Etude SDK 0.2.0
(the migration landed in #349). Package versions do not replace native world,
content, observation, or checkpoint compatibility identities. No old checkpoint
is made compatible by a package upgrade. These are research releases, not
certification of playing strength or full Magic rules coverage.

## Install a pinned release

After 0.2.0 is published, outside a checkout:

```bash
uv venv --python 3.12
# CPU; backend selection applies to Torch, not to the rest of PyPI.
uv pip install --python .venv/bin/python --torch-backend cpu 'manabot==0.2.0'
uv run --no-project manabot --help
uv run --no-project manabot train
```

For Linux NVIDIA training, replace `--torch-backend cpu` with
`--torch-backend cu128`. The package requires Torch `==2.10.0`; its `+cpu` and
`+cu128` builds both satisfy that public version. A plain PyPI install uses
PyTorch's platform default, so select the backend explicitly to avoid a large
unintended CUDA installation. CUDA needs a compatible NVIDIA GPU and driver;
release CI checks CPU execution and dependency installation, not GPU execution.
See [PyTorch installation](https://pytorch.org/get-started/locally/).

Supported binary release targets:

| Component | Supported targets |
|---|---|
| Python | CPython 3.12 only (`>=3.12,<3.13`), no free-threaded ABI |
| Linux | x86-64, glibc 2.28+; CPU or CUDA 12.8 Torch |
| macOS | Apple Silicon, macOS 14+ for the training stack; CPU Torch |
| Other platforms | Windows, Intel Mac, Linux ARM and musl are not release-tested or supplied as wheels |

managym contains the compiled engine, its Python decision/protocol models, and
NumPy/Pydantic requirements. It needs no Rust toolchain when a matching wheel
exists. Its sdist includes Rust sources and the sibling semantic content tree;
building it requires Rust and a C linker. Source builds on other targets are
unsupported. Use `--only-binary :all:` to fail instead of attempting a native
build on an unsupported machine.

`manabot` contains the training/evaluation Python modules, runtime templates,
semantic schemas/IR/source data, and Academy's Python API, fixtures and curated
pack metadata. Optional extras:

- `manabot[academy]`: FastAPI/Uvicorn for the Academy backend. The browser
  frontend, Node build, card art and hosted services are not distributed on PyPI;
  use the repository's `./scripts/play` for the complete browser experience.
- `manabot[evaluation]`: Matplotlib for evaluation visualizations. Ordinary
  checkpoint evaluation is included in the base package.
- `manabot[reports]`: Matplotlib and Etude's S3 support for reports.
- `manabot[artifacts]`: boto3 for artifact storage. Credentials are configured
  separately; installing the extra allocates no compute or storage.
- `manabot[notebook]`: interactive notebook tooling.

Core library calls, `manabot train` with built-in presets, `manabot sim`, and
JSON `--regime` inputs are package interfaces. `--regime-name` loads
`experiments/recipes.py` and therefore requires a source checkout. Preregistered
experiment runners, source-bundle remote submission, research receipt generation,
and repository/data-ledger utilities also require their source tree and retained
artifacts; they are not portable wheel-only workflows. Never update a running
experiment's pinned checkout or controller to install a release.

## Build and verify

Run from a clean checkout with uv, CPython 3.12 and Rust:

```bash
uv run --no-project python scripts/release/check.py
uv run --no-project python scripts/release/prepare.py .runs/release/staging
uv build --no-sources .runs/release/staging/manabot --out-dir dist
uv build --no-sources .runs/release/staging/managym --out-dir dist
uv run --no-project python scripts/release/validate.py dist
uv run --no-project --with twine twine check --strict dist/*
uv run --no-project --with pytest pytest --confcutdir=tests/infra tests/infra/test_release.py -q
```

Use a fresh staging path and empty output directory for each candidate. Staging
refuses to overwrite an earlier build. It copies tracked files with their current
bytes, so commit and review changes before tagging. It excludes experiments,
build products and frontend dependencies. Native source archives retain the
`managym/` + `content/` relationship required by Rust `include_str!` and
`include_bytes!`. Both `uv build` and CI's `maturin build --sdist` build wheels
from their source archives, rather than accidentally reading the checkout.

The root project is also the development environment: its uv groups select
separate Torch CPU and CUDA indexes. `prepare.py` adds the public Torch runtime
requirement from `[tool.manabot.release]` to the staged Manabot metadata. Release
artifacts must use this preparation step; direct root builds are development
artifacts. The resulting published sdist has the complete runtime requirements
and can be rebuilt directly with a standard Python build frontend.

For a clean install smoke (use absolute paths to the artifacts and script):

```bash
uv venv --python 3.12 /tmp/manabot-release-venv
uv pip install --python /tmp/manabot-release-venv/bin/python --only-binary :all: \
  --torch-backend cpu --find-links "$PWD/dist" "$PWD"/dist/manabot-*.whl
cp scripts/release/smoke.py /tmp/manabot-release-smoke.py
cd /tmp
export PATH="/tmp/manabot-release-venv/bin:$PATH"
unset PYTHONPATH
uv run --no-project --python /tmp/manabot-release-venv/bin/python manabot-release-smoke.py
```

This admits compiled semantic catalogs, checks Torch gradients, starts the CLI,
and completes a tiny CPU training update outside the repository. CI repeats it
on Linux and macOS, then installs the Academy, reports, artifacts and evaluation
extras and imports their entry points. It does not contact experiment services.

## Publish

`.github/workflows/release.yml` builds and validates on relevant PRs, manual
runs, and version tags. PR/manual runs do not upload to PyPI. Tag builds reject
inconsistent versions, tags whose commit is not on `main`, already published
versions, and registry/network failures. All platform installs must pass before
publication. The publishing job alone receives OIDC `id-token: write`; no PyPI
API token is stored. PyPI's action produces attestations for the uploaded files.

One-time setup by the repository/PyPI owner, before the first tag:

1. Confirm control of both `manabot` and `managym` on PyPI. On 2026-10-21,
   both public JSON endpoints returned HTTP 404 (no published projects/releases).
   This does **not** establish ownership or guarantee the names are available.
2. Register a pending trusted publisher for **each** name (or an existing
   project's publisher if it has since been created): owner `loopflowstudio`,
   repository `manabot`, workflow `release.yml`, environment `pypi`.
3. Create the GitHub `pypi` environment with required reviewers and version-tag
   deployment restrictions. Confirm both publisher records and reviewer policy
   before approving the first publishing job. No secrets or running services
   need to change.

See PyPI's [pending publisher setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [existing project setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
Public registry reads cannot confirm an account's publisher configuration.
The GitHub environment API also returned 404 for `pypi` during preparation;
the environment setup is not yet verified.

For each release, update both Python versions, Cargo.toml/Cargo.lock and Manabot's
exact managym dependency; update `uv.lock`, installation examples and release notes.
Review and merge the release PR, confirm the complete release workflow is green,
and create/push the matching `vX.Y.Z` tag at that reviewed main commit. Review
the pending `pypi` job's artifacts and approve it through the repository release
procedure. A PR publication alone does not authorize a tag or PyPI upload.

Uploads are not transactional across projects. Failed builds/smokes upload
nothing. A partial PyPI upload fails visibly; `skip-existing` is deliberately
not enabled. Do not rerun publishing blindly, delete a release, or move a used
tag. Inspect which files exist, retain the workflow artifacts and digests, and
prepare a new patch version of the pair if necessary. If no files reached PyPI,
resolve publisher configuration and rerun the reviewed tag workflow. After an
upload, test installation from PyPI without `--find-links` on both targets,
confirm both versions and attestations, and record the release URL and checks.
