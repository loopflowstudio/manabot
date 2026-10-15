#!/usr/bin/env bash
set -eu
export PATH=/root/.local/bin:/root/.cargo/bin:$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled
export CARGO_BUILD_JOBS=2
export UV_CACHE_DIR=/tmp/fantasia-gpu-validation-uv
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.98.1
fi
rustup toolchain install 1.98.1 --profile minimal
export RUSTUP_TOOLCHAIN=1.98.1
uv sync --locked --python 3.12 --extra dev
uv run --no-sync maturin develop --release --features python --manifest-path managym/Cargo.toml
uv --version
rustc --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
