#!/bin/bash
set -euo pipefail

source /app/.venv/bin/activate

cargo test --manifest-path /app/managym/Cargo.toml
pytest /app/tests/env/ /app/tests/agent/ -v
