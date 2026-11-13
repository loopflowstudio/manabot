"""Prepare the pinned Manabot engine and Python environment on an Etude machine."""

RUST_TOOLCHAIN = "1.98.1"
# Etude machine preparation shared by every job that runs repository code.
TOOLCHAIN_ENV = (
    f'export PATH="$HOME/.cargo/bin:$PATH" RUSTUP_TOOLCHAIN={RUST_TOOLCHAIN}'
)


def machine_setup(extras: str) -> str:
    """Setup script installing the pinned Rust toolchain and locked venv.

    `extras` is appended verbatim to `uv sync` (e.g. `" --extra dev"`). Etude runs
    setup once per attempt, and attempts sharing a machine share `$HOME/.cargo`, so
    a `flock` serializes the toolchain install. `uv sync` stays outside it: each
    attempt has its own source directory, and uv and cargo lock their own caches.
    Linux machines supply `flock`; macOS has none, so a local attempt installs
    unserialized.
    """
    return (
        "set -eu\n"
        "{\n"
        "if command -v flock >/dev/null; then flock 9; fi\n"
        'test -x "$HOME/.cargo/bin/rustup" || curl --proto =https --tlsv1.2 -sSf '
        "https://sh.rustup.rs | sh -s -- -y --profile minimal "
        f"--default-toolchain {RUST_TOOLCHAIN}\n"
        f"{TOOLCHAIN_ENV}\n"
        f"rustup toolchain install {RUST_TOOLCHAIN} --profile minimal\n"
        '} 9>"$HOME/.manabot-rust-setup.lock"\n'
        f"uv sync --locked --python 3.13{extras}\n"
    )
