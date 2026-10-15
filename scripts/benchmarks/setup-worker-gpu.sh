#!/usr/bin/env bash
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/worker-gpu-env.sh"
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.98.1
fi
rustup toolchain install 1.98.1 --profile minimal
uv sync --locked --python 3.12 --extra dev --extra play
uv run --no-sync maturin develop --release --features python --manifest-path managym/Cargo.toml
uv --version
rustc --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
