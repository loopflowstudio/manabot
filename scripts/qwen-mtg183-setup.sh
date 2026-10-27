#!/bin/sh
# Prepare once; the game reuses this environment, cache and loaded model.
set -eu
uv sync --locked --python 3.12 --no-install-project --extra llm
native=manabot/managym/_managym.cpython-312-x86_64-linux-gnu.so
cp "$ETUDE_INPUTS_DIR/native.so" "$native.pending"
mv "$native.pending" "$native"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$ETUDE_OUTPUT_DIR/gpu.csv"
uv run --no-sync python -m manabot.llm.cache \
    --config configs/llm/qwen3.5-9b-reasoning.json \
    --manifest configs/llm/qwen3.5-9b-cache.json \
    --out "$ETUDE_OUTPUT_DIR/cache.json"
