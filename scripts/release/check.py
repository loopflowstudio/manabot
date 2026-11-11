"""Reject inconsistent release versions and tags."""

import argparse
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def check(tag: str | None) -> None:
    bot = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    cargo = tomllib.loads((ROOT / "managym/Cargo.toml").read_text())["package"]
    version = bot["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("release versions must be X.Y.Z")
    if cargo["version"] != version:
        raise ValueError("Manabot and its bundled Rust crate versions must match")
    if any(dependency.split("=")[0] == "managym" for dependency in bot["dependencies"]):
        raise ValueError("managym must be bundled, not a separate dependency")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"expected tag v{version}, got {tag}")
    notes = ROOT / "release" / f"v{version}" / "NOTES.md"
    if not notes.is_file() or not notes.read_text().strip():
        raise ValueError(f"release notes missing or empty: {notes.relative_to(ROOT)}")
    print(f"validated release v{version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    args = parser.parse_args()
    check(args.tag)
