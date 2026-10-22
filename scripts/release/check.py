"""Reject inconsistent release versions, tags, or already-published artifacts."""

import argparse
import json
from pathlib import Path
import re
import tomllib
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


def check(tag: str | None, unpublished: bool) -> None:
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
    if unpublished:
        for name in ("manabot",):
            try:
                with urlopen(
                    f"https://pypi.org/pypi/{name}/json", timeout=30
                ) as response:
                    data = json.load(response)
            except HTTPError as error:
                if error.code == 404:
                    continue
                raise
            if version in data["releases"]:
                raise ValueError(
                    f"{name} {version} already exists; do not overwrite/retry uploads"
                )
    print(f"validated release v{version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    parser.add_argument("--unpublished", action="store_true")
    args = parser.parse_args()
    check(args.tag, args.unpublished)
