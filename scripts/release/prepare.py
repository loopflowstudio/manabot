"""Stage self-contained release projects without changing the development checkout.

Source uv groups select different Torch indexes. The published Manabot project
instead declares the public Torch version, leaving backend selection to installers.
Native sources need their sibling content tree preserved in wheels' source archive.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def prepare(destination: Path) -> None:
    """Write fresh, minimal projects; refuse to overwrite previous artifacts."""
    destination.mkdir(parents=True, exist_ok=False)
    files = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    bot = destination / "manabot"
    native = destination / "managym"
    for name in filter(None, files):
        path = Path(name)
        if path.parts[0] in {"manabot", "academy", "content"} or name in {
            "pyproject.toml",
            "README.md",
            "docs/experiment-metrics.md",
        }:
            if name.startswith("academy/ui/") and not name.startswith(
                "academy/ui/src/lib/packs/"
            ):
                continue
            _copy(path, bot)
        if (
            path.parts[0] == "content"
            or name.startswith(("managym/src/", "managym/tests/", "managym/examples/"))
            or (path.parent == Path("managym") and path.suffix in {".py", ".pyi"})
            or name in {"managym/Cargo.toml", "managym/Cargo.lock", "managym/README.md"}
        ):
            _copy(path, native)

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = config["tool"]["manabot"]["release"]["runtime-dependencies"]
    bot_config = (
        (ROOT / "pyproject.toml")
        .read_text()
        .replace(
            "dependencies = [\n",
            "dependencies = [\n"
            + "".join(f"    {json.dumps(dep)},\n" for dep in dependencies),
            1,
        )
    )
    (bot / "pyproject.toml").write_text(bot_config)
    native_config = (
        (ROOT / "managym/pyproject.toml")
        .read_text()
        .replace('readme = "README.md"', 'readme = "managym/README.md"')
        .replace('python-source = ".."', 'python-source = "."')
    )
    native_config += '\nmanifest-path = "managym/Cargo.toml"\ninclude = [{ path = "content/semantic/**/*.json", format = "sdist" }]\n'
    (native / "pyproject.toml").write_text(native_config)


def _copy(relative: Path, destination: Path) -> None:
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    prepare(parser.parse_args().destination.resolve())
