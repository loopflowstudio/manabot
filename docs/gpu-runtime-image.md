# Lock-bound GPU runtime image

The startup comparison keeps the scientific source, all 94 selected Linux wheels,
and the native managym build unchanged. The candidate moves dependency installation
and Rust toolchain installation into an image. This does not establish a speedup:
provider allocation and image acquisition count toward startup time too.

Build from an independently verified bundle receipt and its admitted source:

```bash
uv run python -m experiments.runners.gpu_runtime_image \
  --bundle /path/to/locked-wheels.tar --receipt /path/to/bundle-receipt.json \
  --source /path/to/frozen-source --tag ghcr.io/loopflowstudio/manabot-training:LOCK \
  --out .runs/runtime-build --push
```

`--source` needs `uv.lock` and `pyproject.toml`. The receipt supplies `sha256`,
`bytes` and `manifest_sha256`; URLs are unnecessary. The existing bundle verifier
checks archive membership, every wheel hash against the source lock, the admitted
lock identity, and exact hashed requirements before Docker runs. Buildx requires
Linux amd64 support. Without `--push`, the command loads a local test image.
Use existing Docker registry authentication; credentials never enter the context,
image, manifest or repository. The generated context contains only the verified
wheels, metadata, Dockerfile and admission program.

The recipe pins Linux amd64 Rust/Debian, Python and uv inputs by digest. It contains
one Torch/CUDA installation and no stock PyTorch environment. Wheel archives are
mounted at build time and discarded, avoiding another large image layer. The uv
cache retains the small project build dependencies needed for offline source
installation. It never contains the dependency bundle or credentials.

Rebuild on any dependency lock, project metadata, base-image digest, toolchain,
Dockerfile or admission-program change. `runtime.json` binds these inputs and the
original bundle; `runtime.sha256` is its launch identity. `image.json` records
Docker's resulting image digest. Source-only changes can reuse this runtime only
when their lock and project metadata match. Native source is compiled each start.
A new lock requires a newly verified bundle, even if versions appear unchanged.

Deploy the **digest reference** from `image.json`, never its mutable tag, in
Etude's `MachineTemplate.image`. The existing RunPod registry configuration owns
private-image access. Pass `--prebuilt-runtime SHA256` to the existing
`client_install_canary` wrapper, where SHA256 comes from `runtime.sha256`. Its
setup invokes the image's `/opt/manabot/runtime/bin/python` and admission program.
An explicit manifest identity, matching source metadata, CPython 3.12/Linux amd64,
the pinned uv/Rust versions and every installed dependency version are required
before source installation. There is no dependency-download fallback.

The source install runs `uv sync --offline --locked`; the native build runs
`uv run --no-sync maturin develop` on the selected source. Source installation,
native build and GPU health timings are retained in `dependency-setup.json`.
The ordinary setup path remains the comparison baseline. Both use the same
1,200-second setup deadline, CUDA admission, two-update recipe, output courier,
independent rental guardian and launch freshness guard. A candidate failure
stops training and remains evidence; do not retry without diagnosis.

Keep a separate Task ledger, Etude state directory and controller. Reserve cost
before admission, inspect the complete fresh provider inventory, and verify
provider deletion after each attempt. Count allocation/image acquisition, client
bootstrap, source delivery, setup, CUDA admission, first update and recovered
checkpoint. Report cached, demonstrably uncached or unknown acquisition according
to provider evidence; container setup timing alone is insufficient.
