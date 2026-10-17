"""Admit a lock-bound image, then install source and compile its native engine.

This stdlib entry point is baked into the image. The launch configuration pins
its runtime manifest digest independently of the image's self-description.
Admission precedes mutation; missing or incompatible dependencies never trigger
a download. The source-specific project build reuses the image's uv build cache.
"""

import argparse
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import TypedDict, cast


class Package(TypedDict):
    name: str
    version: str


class Runtime(TypedDict):
    format: int
    allowed_lock_sha256: list[str]
    pyproject_sha256: str
    recipe_sha256: str
    packages: list[Package]


ROOT = Path("/opt/manabot")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def installed_packages() -> dict[str, str]:
    return {
        dist.metadata["Name"].lower().replace("_", "-"): dist.version
        for dist in distributions()
    }


def admit(source: Path, manifest: Path, expected: str) -> Runtime:
    """Reject stale source, toolchain or dependencies before installing anything."""
    if digest(manifest) != expected:
        raise ValueError("Prebuilt runtime identity differs from the launch")
    value = json.loads(manifest.read_text())
    if not isinstance(value, dict) or value.get("format") != 1:
        raise ValueError("Unsupported runtime manifest")
    # The externally pinned digest admits the build-generated manifest schema.
    runtime = cast(Runtime, value)
    if (
        platform.python_implementation() != "CPython"
        or sys.version_info[:2] != (3, 12)
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
    ):
        raise ValueError("Runtime requires CPython 3.12 on Linux x86_64")
    if digest(source / "uv.lock") not in runtime["allowed_lock_sha256"]:
        raise ValueError("Runtime does not admit this source lock")
    if digest(source / "pyproject.toml") != runtime["pyproject_sha256"]:
        raise ValueError("Runtime does not admit this project metadata")
    installed = installed_packages()
    for package in runtime["packages"]:
        if installed.get(package["name"]) != package["version"]:
            raise ValueError(f"Runtime dependency differs: {package['name']}")
    for command, prefix in (
        (["uv", "--version"], "uv 0.8.22"),
        (["rustc", "--version"], "rustc 1.98.1"),
    ):
        actual = subprocess.check_output(command, text=True).strip()
        if actual.split(" (", 1)[0] != prefix:
            raise ValueError(f"Runtime toolchain differs: {command[0]}")
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity")
    parser.add_argument("--seal", action="store_true", help="Build-time validation")
    args = parser.parse_args()
    manifest = ROOT / "runtime.json"
    source = Path.cwd()
    if not args.seal and not args.identity:
        parser.error("--identity is required at runtime")
    began = time.monotonic()
    admit(source, manifest, digest(manifest) if args.seal else args.identity)
    if args.seal:
        return
    environment = ROOT / "runtime"
    if Path(sys.prefix) != environment:
        raise ValueError("Run admission with the image runtime interpreter")
    if (source / ".venv").exists() or (source / ".venv").is_symlink():
        raise ValueError("Refusing to replace an existing source environment")
    output = Path(os.environ["ETUDE_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {"admission_seconds": time.monotonic() - began}
    receipt: dict[str, object] = {
        "runtime_sha256": args.identity,
        "source_lock_sha256": digest(source / "uv.lock"),
        "python": sys.version,
        "timings": timings,
        "phase": "admitted",
    }

    def record() -> None:
        path = output / "dependency-setup.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(path)

    record()
    (source / ".venv").symlink_to(environment, target_is_directory=True)
    env = dict(os.environ)
    env.update(
        UV_PROJECT_ENVIRONMENT=str(environment),
        UV_CACHE_DIR=str(ROOT / "uv-cache"),
        UV_OFFLINE="1",
        UV_PYTHON_DOWNLOADS="never",
        VIRTUAL_ENV=str(environment),
    )
    for name, command in (
        (
            "source_install",
            [
                "uv",
                "sync",
                "--offline",
                "--locked",
                "--extra",
                "dev",
                "--extra",
                "play",
                "--python",
                sys.executable,
            ],
        ),
        (
            "native_build",
            [
                "uv",
                "run",
                "--no-sync",
                "maturin",
                "develop",
                "--release",
                "--features",
                "python",
                "--manifest-path",
                "managym/Cargo.toml",
            ],
        ),
        (
            "gpu_health",
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv",
            ],
        ),
    ):
        receipt["phase"] = name
        record()
        start = time.monotonic()
        subprocess.run(command, env=env, check=True)
        timings[name + "_seconds"] = time.monotonic() - start
    receipt["phase"] = "completed"
    record()


if __name__ == "__main__":
    main()
