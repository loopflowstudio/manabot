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
uv run --locked maturin develop --release --manifest-path managym/Cargo.toml --features python
uv run --locked python -m manabot.infra.source
exec uv run --locked python -m "$@"
