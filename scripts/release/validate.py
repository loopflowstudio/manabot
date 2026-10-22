"""Validate release inventory and dependency metadata before trusted publishing."""

import argparse
from email.parser import Parser
from pathlib import Path
import tarfile
import zipfile


def validate(directory: Path, complete: bool) -> None:
    wheels = sorted(directory.glob("*.whl"))
    archives = sorted(directory.glob("*.tar.gz"))
    if len(archives) != 2 or not wheels:
        raise ValueError("expected both source archives and platform wheels")
    versions: set[str] = set()
    names: set[str] = set()
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            files = archive.namelist()
            (metadata_file,) = [
                name for name in files if name.endswith(".dist-info/METADATA")
            ]
            metadata = Parser().parsestr(archive.read(metadata_file).decode())
            name = metadata["Name"]
            version = metadata["Version"]
            versions.add(version)
            names.add(name)
            if name == "manabot":
                requirements = metadata.get_all("Requires-Dist", [])
                if (
                    f"managym=={version}" not in requirements
                    or "torch==2.10.0" not in requirements
                ):
                    raise ValueError("Manabot runtime dependencies are incomplete")
                if (
                    "content/semantic/w11/generated/elemental_world.ir.json"
                    not in files
                ):
                    raise ValueError("Manabot wheel is missing semantic runtime data")
            elif name == "managym":
                if not any(
                    path.startswith("managym/_managym.") and path.endswith(".so")
                    for path in files
                ):
                    raise ValueError("managym wheel is missing its native extension")
            else:
                raise ValueError(f"unexpected wheel {wheel.name}")
    if names != {"manabot", "managym"} or len(versions) != 1:
        raise ValueError("expected a matching Manabot/managym version pair")
    (version,) = versions
    expected_sources = {f"{name}-{version}.tar.gz" for name in names}
    if {path.name for path in archives} != expected_sources:
        raise ValueError("source archive names differ from wheel versions")
    for source in archives:
        with tarfile.open(source) as archive:
            root = source.name.removesuffix(".tar.gz")
            files = set(archive.getnames())
            if (
                f"{root}/content/semantic/w11/generated/elemental_world.ir.json"
                not in files
            ):
                raise ValueError(f"missing content in {source.name}")
    if complete:
        if len(wheels) != 3:
            raise ValueError("expected one Python wheel and two native wheels")
        for target in ("manylinux_2_28_x86_64", "macosx_11_0_arm64"):
            if not any(f"cp312-cp312-{target}" in path.name for path in wheels):
                raise ValueError(f"missing CPython 3.12 wheel for {target}")
    print(f"validated {len(wheels)} wheels and both source archives for {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    validate(args.directory, args.complete)
