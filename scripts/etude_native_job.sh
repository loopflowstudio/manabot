#!/bin/sh
# Build exact source on the remote machine before running an Etude-owned job.
set -eu
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$HOME/.cargo/bin:$PATH"
if [ "$(uname -s)" = Linux ]; then
  command -v cargo >/dev/null || { curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.98.1; }
  rustup toolchain install 1.98.1 --profile minimal
  export RUSTUP_TOOLCHAIN=1.98.1
fi
uv sync --locked --python 3.12 --extra play
uv run --locked maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels
uv run --locked python -c 'from pathlib import Path; from zipfile import ZipFile; import sysconfig; wheel=max(Path("managym/target/wheels").glob("*.whl"),key=lambda p:p.stat().st_mtime); z=ZipFile(wheel); name=next(n for n in z.namelist() if n.endswith(".so")); Path("managym/_managym"+sysconfig.get_config_var("EXT_SUFFIX")).write_bytes(z.read(name))'
uv run --locked python -m manabot.training.source_bundle
exec uv run --locked python -m "$@"
