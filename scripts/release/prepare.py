"""Stage the paired Python release projects and their runtime assets.

The development uv groups select CPU/CUDA indexes. Published metadata declares
the public Torch version instead, leaving backend selection to installers.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def prepare(destination: Path, site: Path) -> None:
    """Copy tracked source bytes into a fresh release tree, preserving Rust paths."""
    if not (site / "index.html").is_file():
        raise ValueError("Academy site must contain index.html; build academy/ui first")
    destination.mkdir(parents=True, exist_ok=False)
    # The engine project needs its crate and shared content beneath one root,
    # preserving include_str! paths without parent-directory sdist entries.
    engine = destination / "managym"
    source_files: dict[str, str] = {}
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
        } and name not in {
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "docs/experiment-metrics.md",
            "protocol/experience-v1.schema.json",
        }:
            continue
        if name.startswith("academy/ui/"):
            continue
        if name.startswith("managym/libmanagym.dylib.dSYM/"):
            continue
        target = (
            engine if path.parts[0] in {"managym", "academy"} else destination
        ) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
        # Source receipts use installed paths, not staging paths. Bind
        # Python and protocol bytes; runtime fingerprints bind the native engine.
        if name == "protocol/experience-v1.schema.json" or (
            path.suffix == ".py"
            and (path.parts[0] == "manabot" or path.parent == Path("managym"))
        ):
            source_files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    shutil.copytree(site, engine / "managym/academy/site")
    metadata = (engine / "managym/pyproject.toml").read_text()
    metadata = metadata.replace(
        'manifest-path = "Cargo.toml"', 'manifest-path = "managym/Cargo.toml"'
    )
    metadata = metadata.replace('python-source = ".."', 'python-source = "."')
    (engine / "pyproject.toml").write_text(metadata)
    (engine / "managym/pyproject.toml").unlink()
    shutil.copy2(engine / "managym/README.md", engine / "README.md")
    shutil.copytree(destination / "content", engine / "content")
    source = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": source_files,
    }
    (destination / ".manabot-source.json").write_text(json.dumps(source, indent=2))
    project = destination / "pyproject.toml"
    project.write_text(
        project.read_text().replace(
            "[tool.hatch.build.targets.wheel.force-include]",
            '[tool.hatch.build.targets.wheel.force-include]\n".manabot-source.json" = ".manabot-source.json"',
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.destination.resolve(), args.site.resolve())
