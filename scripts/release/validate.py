"""Validate release inventory and dependency metadata before trusted publishing.

The source archive defines one metadata contract; every wheel must match it.
"""

import argparse
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
import tarfile
import zipfile


@dataclass(frozen=True)
class _Metadata:
    name: str
    version: str
    python: str
    requirements: frozenset[str]


def _metadata(data: bytes) -> _Metadata:
    metadata = Parser().parsestr(data.decode())
    return _Metadata(
        name=metadata["Name"],
        version=metadata["Version"],
        python=metadata["Requires-Python"].replace(" ", ""),
        requirements=frozenset(metadata.get_all("Requires-Dist", [])),
    )


def validate(directory: Path, complete: bool) -> None:
    wheels = sorted(directory.glob("*.whl"))
    archives = list(directory.glob("*.tar.gz"))
    if len(archives) != 1 or not wheels:
        raise ValueError("expected one Manabot source archive and platform wheels")
    (source,) = archives
    root = source.name.removesuffix(".tar.gz")
    with tarfile.open(source) as archive:
        package_info = archive.extractfile(f"{root}/PKG-INFO")
        if package_info is None:
            raise ValueError("source metadata missing")
        expected = _metadata(package_info.read())
        if (
            f"{root}/content/semantic/current/learning_schema.json"
            not in archive.getnames()
        ):
            raise ValueError(f"missing content in {source.name}")

    if expected.name != "manabot" or root != f"manabot-{expected.version}":
        raise ValueError("source archive name differs from Manabot metadata")
    if expected.python != ">=3.12,<3.14":
        raise ValueError("expected CPython 3.12 and 3.13 support")
    requirements = {
        requirement.replace(" ", "").replace('"', "'")
        for requirement in expected.requirements
    }
    if not {
        "torch>=2.10.0;sys_platform=='linux'",
        "torch>=2.10.0;sys_platform!='linux'",
        "etude-sdk>=0.2.15",
    }.issubset(requirements) or any(
        requirement.startswith("managym") for requirement in requirements
    ):
        raise ValueError("Manabot must declare Torch/Etude minimums and bundle managym")

    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            files = archive.namelist()
            (metadata_file,) = [
                name for name in files if name.endswith(".dist-info/METADATA")
            ]
            if _metadata(archive.read(metadata_file)) != expected:
                raise ValueError(f"source and wheel metadata differ: {wheel.name}")
            if not any(
                path.startswith("manabot/managym/_managym.") and path.endswith(".so")
                for path in files
            ):
                raise ValueError(
                    "Manabot wheel is missing its bundled native extension"
                )
            if "content/semantic/current/learning_schema.json" not in files:
                raise ValueError("Manabot wheel is missing semantic runtime data")

    if complete:
        expected_wheels = {
            f"manabot-{expected.version}-cp{python}-cp{python}-{target}.whl"
            for python in ("312", "313")
            for target in ("manylinux_2_28_x86_64", "macosx_11_0_arm64")
        }
        if {path.name for path in wheels} != expected_wheels:
            raise ValueError(
                "expected four wheels: CPython 3.12/3.13 on Linux x86_64 and macOS arm64"
            )
    print(
        f"validated {len(wheels)} wheels and one source archive for {expected.version}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    validate(args.directory, args.complete)
