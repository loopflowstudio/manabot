#!/bin/sh
# Keep the game environment and the vLLM server's pinned dependencies separate.
set -eu
sh scripts/qwen-mtg183-setup.sh
uv venv --python 3.12 .venv-vllm
uv pip sync --python .venv-vllm/bin/python --require-hashes configs/llm/vllm-0.17.1-linux.lock
