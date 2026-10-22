"""Stage one self-contained Manabot release project with its bundled Rust engine.

The development uv groups select CPU/CUDA indexes. Published metadata declares
the public Torch version instead, leaving backend selection to installers.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def prepare(destination: Path) -> None:
    """Copy tracked source bytes into a fresh release tree, preserving Rust paths."""
    destination.mkdir(parents=True, exist_ok=False)
    files = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    for name in filter(None, files):
        path = Path(name)
        if not (ROOT / path).is_file():
            continue
        if path.parts[0] not in {
            "manabot",
            "academy",
            "content",
            "managym",
        } and name not in {"pyproject.toml", "README.md", "docs/experiment-metrics.md"}:
            continue
        if name.startswith("academy/ui/") and not name.startswith(
            "academy/ui/src/lib/packs/"
        ):
            continue
        if name.startswith("managym/libmanagym.dylib.dSYM/"):
            continue
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = config["tool"]["manabot"]["release"]["runtime-dependencies"]
    metadata = (
        (ROOT / "pyproject.toml")
        .read_text()
        .replace(
            "dependencies = [\n",
            "dependencies = [\n"
            + "".join(f"    {json.dumps(dep)},\n" for dep in dependencies),
            1,
        )
    )
    (destination / "pyproject.toml").write_text(metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    prepare(parser.parse_args().destination.resolve())
