#!/usr/bin/env bash
# Source in the job shell so setup, native builds and training share these values.
export PATH="${CARGO_HOME:-/root/.cargo}/bin:/root/.local/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
<<<<<<< HEAD
export WANDB_MODE=disabled
=======
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
export CARGO_BUILD_JOBS=2
export UV_CACHE_DIR=/tmp/fantasia-gpu-validation-uv
export RUSTUP_TOOLCHAIN=1.98.1
