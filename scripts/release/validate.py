"""Validate both release inventories before either trusted-publisher upload.

Each source archive defines its wheel metadata contract. The packages share a
version but have disjoint code ownership: the native engine lives in managym.
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
    expected: dict[str, _Metadata] = {}
    for source in archives:
        root = source.name.removesuffix(".tar.gz")
        with tarfile.open(source) as archive:
            package_info = archive.extractfile(f"{root}/PKG-INFO")
            if package_info is None:
                raise ValueError("source metadata missing")
            metadata = _metadata(package_info.read())
            if (
                metadata.name not in {"managym", "manabot"}
                or root != f"{metadata.name}-{metadata.version}"
            ):
                raise ValueError("source archive name differs from package metadata")
            if metadata.name in expected:
                raise ValueError("duplicate source archive")
            expected[metadata.name] = metadata
            if (
                f"{root}/content/semantic/current/learning_schema.json"
                not in archive.getnames()
            ):
                raise ValueError(f"missing content in {source.name}")
    if set(expected) != {"managym", "manabot"}:
        raise ValueError("expected both managym and manabot source archives")
    engine, bot = expected["managym"], expected["manabot"]
    if engine.version != bot.version:
        raise ValueError("package versions must match")
    for metadata in expected.values():
        if set(metadata.python.split(",")) != {">=3.12", "<3.14"}:
            raise ValueError("expected CPython 3.12 and 3.13 support")
    if engine.requirements != frozenset({"numpy>=1.24.0", "pydantic>=2.12.0"}):
        raise ValueError(
            "managym must declare only its lightweight runtime dependencies"
        )
    requirements = {r.replace(" ", "").replace('"', "'") for r in bot.requirements}
    if not {
        f"managym=={bot.version}",
        "torch>=2.15.0;sys_platform=='linux'",
        "torch>=2.15.0;sys_platform!='linux'",
        "etude-sdk>=0.2.16",
    }.issubset(requirements):
        raise ValueError(
            "manabot must declare Torch/Etude minimums and its exact managym version"
        )
    seen: set[str] = set()
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            files = archive.namelist()
            (metadata_file,) = [n for n in files if n.endswith(".dist-info/METADATA")]
            metadata = _metadata(archive.read(metadata_file))
            if metadata != expected.get(wheel.name.split("-")[0]):
                raise ValueError(f"source and wheel metadata differ: {wheel.name}")
            seen.add(metadata.name)
            native = [p for p in files if p.endswith((".so", ".dylib", ".pyd"))]
            if metadata.name == "managym":
                if len(native) != 1 or not native[0].startswith("managym/_managym."):
                    raise ValueError("managym wheel must own its native extension")
                if any(p.startswith(("manabot/", "academy/")) for p in files):
                    raise ValueError("engine wheel contains training or Academy")
                if "managym/Cargo.lock" not in files:
                    raise ValueError("managym wheel is missing its provenance lockfile")
            elif native or any(p.startswith("managym/") for p in files):
                raise ValueError(
                    "manabot wheel must be pure Python without a second engine"
                )
            elif "content/semantic/current/learning_schema.json" not in files:
                raise ValueError("manabot wheel is missing semantic runtime data")
    if seen != set(expected):
        raise ValueError("expected wheels for both packages")
    if complete:
        expected_wheels = {
            f"managym-{bot.version}-cp{python}-cp{python}-{target}.whl"
            for python in ("312", "313")
            for target in ("manylinux_2_28_x86_64", "macosx_11_0_arm64")
        } | {f"manabot-{bot.version}-py3-none-any.whl"}
        if {p.name for p in wheels} != expected_wheels:
            raise ValueError(
                "expected four wheels for managym and one pure-Python manabot wheel"
            )
    print(f"validated {len(wheels)} wheels and two source archives for {bot.version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    validate(args.directory, args.complete)
