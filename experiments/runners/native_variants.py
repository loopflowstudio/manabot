"""Build and select immutable native controls for a sequential GPU experiment.

Both variants use the same checkout/toolchain; only vector_env.rs is replaced by
its hash-checked historical source for the cloned-config control. Build receipts
bind every Rust input and the resulting library. Selection atomically replaces
the library for a fresh child process, never overwriting a loaded library inode.
Production imports no experiment selector. Run preparation in an idle checkout.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Literal
import zipfile

from pydantic import BaseModel, ConfigDict

NativeVariant = Literal["cloned", "borrowed"]


class NativeBuild(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: NativeVariant
    library: str
    library_sha256: str
    wheel_sha256: str
    sources: dict[str, str]
    command: list[str]


class NativeBuilds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit: str
    rustc: str
    interpreter: str
    reference_commit: str
    reference_sha256: str
    builds: dict[NativeVariant, NativeBuild]


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _native_sources() -> dict[str, str]:
    paths = [
        Path("managym/Cargo.toml"),
        Path("managym/Cargo.lock"),
        Path("managym/pyproject.toml"),
        *sorted(Path("managym/src").rglob("*.rs")),
    ]
    return {str(path): file_digest(path) for path in paths}


def _source_commit() -> str:
    # Etude's exact source archive has no .git directory. The worker separately
    # validates every admitted file after preparation restores the source.
    if config := os.environ.get("ETUDE_CONFIG"):
        commit = json.loads(Path(config).read_text())["source"]["commit"]
    else:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    if not isinstance(commit, str) or re.fullmatch(r"[a-f0-9]{40}", commit) is None:
        raise ValueError("native builds require an exact source commit")
    return commit


def install(directory: Path, build: NativeBuild) -> Path:
    """Select verified bytes before starting a child that imports managym."""
    source = directory / build.library
    if (
        Path(build.library).name != build.library
        or not build.library.startswith("_managym.")
        or not build.library.endswith(".so")
        or not source.is_file()
        or file_digest(source) != build.library_sha256
    ):
        raise ValueError("native library differs from build receipt")
    # Variant directories retain the ABI filename from the wheel, so install
    # never guesses a Python version or silently selects a foreign ABI.
    target = Path("managym") / build.library
    temporary = target.with_suffix(".install")
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    return target


def prepare(out: Path) -> NativeBuilds:
    """Compile two variants; restore original source/library even on failure."""
    out.mkdir(parents=True, exist_ok=False)
    source = Path("managym/src/agent/vector_env.rs")
    original = source.read_bytes()
    libraries = list(Path("managym").glob("_managym*.so"))
    if len(libraries) != 1:
        raise ValueError("prepare requires exactly one installed native library")
    target = libraries[0]
    original_library = target.read_bytes()
    reference = Path("experiments/data/worker-scaling/vector-env-e41edb26.rs")
    receipt = json.loads(reference.with_suffix(".json").read_text())
    if file_digest(reference) != receipt["sha256"]:
        raise ValueError("native reference source differs from receipt")
    builds: dict[NativeVariant, NativeBuild] = {}
    try:
        for variant in ("borrowed", "cloned"):
            source.write_bytes(
                original if variant == "borrowed" else reference.read_bytes()
            )
            directory = out / variant
            directory.mkdir()
            command = [
                "uv",
                "run",
                "--no-sync",
                "maturin",
                "build",
                "--release",
                "--features",
                "python",
                "-i",
                sys.executable,
                "--manifest-path",
                "managym/Cargo.toml",
                "--out",
                str(directory / "wheels"),
            ]
            inputs = _native_sources()
            with (directory / "build.log").open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=300,
                )
            if _native_sources() != inputs:
                raise ValueError("native source changed during compilation")
            wheels = list((directory / "wheels").glob("*.whl"))
            if len(wheels) != 1:
                raise ValueError("expected one native wheel per variant")
            with zipfile.ZipFile(wheels[0]) as wheel:
                members = [name for name in wheel.namelist() if name.endswith(".so")]
                if len(members) != 1 or Path(members[0]).name != target.name:
                    raise ValueError("native wheel ABI differs from installed library")
                library = directory / target.name
                library.write_bytes(wheel.read(members[0]))
            builds[variant] = NativeBuild(
                variant=variant,
                library=target.name,
                library_sha256=file_digest(library),
                wheel_sha256=file_digest(wheels[0]),
                sources=inputs,
                command=command,
            )
    finally:
        source.write_bytes(original)
        temporary = target.with_suffix(".restore")
        temporary.write_bytes(original_library)
        temporary.replace(target)
    changed = {
        key
        for key in builds["borrowed"].sources
        if builds["borrowed"].sources[key] != builds["cloned"].sources[key]
    }
    if changed != {str(source)}:
        raise ValueError("native variants differ outside the declared source file")
    result = NativeBuilds(
        commit=_source_commit(),
        rustc=subprocess.check_output(["rustc", "--version"], text=True).strip(),
        interpreter=sys.version,
        reference_commit=receipt["commit"],
        reference_sha256=receipt["sha256"],
        builds=builds,
    )
    (out / "builds.json").write_text(result.model_dump_json(indent=2) + "\n")
    install(out / "borrowed", builds["borrowed"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.out)
    print(
        json.dumps(
            {
                "commit": result.commit,
                "libraries": {
                    key: build.library_sha256 for key, build in result.builds.items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
