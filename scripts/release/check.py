"""Reject inconsistent versions across the paired Python and Rust release."""

import argparse
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def check(tag: str | None) -> None:
    bot = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = bot["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("release versions must be X.Y.Z")
    for path, table in (
        ("managym/pyproject.toml", "project"),
        ("managym/Cargo.toml", "package"),
        ("managym/state/Cargo.toml", "package"),
        ("managym/cards/Cargo.toml", "package"),
    ):
        if tomllib.loads((ROOT / path).read_text())[table]["version"] != version:
            raise ValueError(f"release versions must match: {path}")
    if f"managym=={version}" not in bot["dependencies"]:
        raise ValueError("manabot must require its exact managym version")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"expected tag v{version}, got {tag}")
    notes = ROOT / "release" / f"v{version}" / "NOTES.md"
    if not notes.is_file() or not notes.read_text().strip():
        raise ValueError(f"release notes missing or empty: {notes.relative_to(ROOT)}")
    print(f"validated release v{version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    check(parser.parse_args().tag)
