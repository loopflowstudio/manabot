#!/bin/sh
# Rebuild the release extension and swap it in atomically.
set -e
cd "$(dirname "$0")/../.."
rm -f managym/target/wheels/*.whl
uv run maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels >/dev/null 2>&1
rm -rf /tmp/whl892 && mkdir /tmp/whl892
(cd /tmp/whl892 && unzip -q "$OLDPWD"/managym/target/wheels/*.whl)
cp /tmp/whl892/manabot/managym/_managym.cpython-312-darwin.so manabot/managym/_managym.cpython-312-darwin.so.new
mv manabot/managym/_managym.cpython-312-darwin.so.new manabot/managym/_managym.cpython-312-darwin.so
echo installed
