#!/bin/sh
# Rebuild the managym extension and install it atomically (CLAUDE.md).
set -e
uv run --extra dev --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels >/dev/null 2>&1
W=$(ls -t managym/target/wheels/manabot-*-cp312-*.whl | head -1)
T=$(mktemp -d)
unzip -q -o "$W" 'manabot/managym/_managym*.so' -d "$T"
cp "$T"/manabot/managym/_managym.cpython-312-darwin.so manabot/managym/_managym.cpython-312-darwin.so.new
mv manabot/managym/_managym.cpython-312-darwin.so.new manabot/managym/_managym.cpython-312-darwin.so
echo rebuilt
