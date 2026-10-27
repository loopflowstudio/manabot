#!/bin/sh
# Install the locked CUDA policy environment and verified prebuilt engine.
set -eu
uv sync --locked --python 3.12 --no-install-project --extra evaluation
native=manabot/managym/_managym.cpython-312-x86_64-linux-gnu.so
cp "$ETUDE_INPUTS_DIR/native.so" "$native.pending"
mv "$native.pending" "$native"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$ETUDE_OUTPUT_DIR/gpu.csv"
