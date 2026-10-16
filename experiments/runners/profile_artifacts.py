"""Package a finished profiling study for bounded artifact transport.

Timing metadata sorts before model payloads. The complete evidence is a single
gzip tar stream split into bounded objects; concatenating parts reconstructs it.
Packaging happens after training and never changes a measured run's exports.
"""

import argparse
from collections.abc import Buffer
from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import BinaryIO


@dataclass(frozen=True)
class Artifact:
    name: str
    bytes: int
    sha256: str


class _Parts(io.RawIOBase):
    def __init__(self, destination: Path, limit: int) -> None:
        self.destination = destination
        self.limit = limit
        self.names: list[Path] = []
        self.current: BinaryIO | None = None
        self.used = 0
        super().__init__()

    def writable(self) -> bool:
        return True

    def write(self, data: Buffer) -> int:
        if self.closed:
            raise ValueError("write to closed artifact stream")
        view = memoryview(data)
        length = len(view)
        while view:
            if self.current is None:
                path = self.destination / (
                    f"10-training-evidence.tar.gz.part-{len(self.names):05d}"
                )
                self.current = path.open("xb")
                self.names.append(path)
                self.used = 0
            take = min(len(view), self.limit - self.used)
            self.current.write(view[:take])
            self.used += take
            view = view[take:]
            if self.used == self.limit:
                self.current.close()
                self.current = None
        return length

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def _artifact(path: Path) -> Artifact:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return Artifact(path.name, path.stat().st_size, digest)


def package(root: Path, destination: Path, *, part_bytes: int = 64 * 1024**2) -> None:
    """Retain all files; a failed package has no complete artifact index."""
    if part_bytes <= 0:
        raise ValueError("part_bytes must be positive")
    root = root.resolve()
    if destination.resolve().is_relative_to(root):
        raise ValueError("Destination must be outside the study")
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    if not (root / "manifest.json").is_file():
        raise ValueError("Finished study manifest is required")
    if any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in paths):
        raise ValueError("Study evidence must contain only local regular files")
    destination.mkdir(parents=True, exist_ok=False)
    metadata = destination / "00-training-metadata.tar.gz"
    with tarfile.open(metadata, "w:gz", compresslevel=1) as archive:
        for path in paths:
            if path.suffix in {".json", ".jsonl", ".log", ".txt"}:
                archive.add(path, arcname=f"training-evidence/{path.relative_to(root)}")
    with _Parts(destination, part_bytes) as parts:
        with gzip.GzipFile(
            fileobj=parts, mode="wb", compresslevel=1, mtime=0
        ) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|") as archive:
                for path in paths:
                    archive.add(
                        path, arcname=f"training-evidence/{path.relative_to(root)}"
                    )
    index = {
        "schema": 1,
        "metadata": asdict(_artifact(metadata)),
        "complete_archive_parts": [asdict(_artifact(p)) for p in parts.names],
        "reconstruction": "Concatenate complete_archive_parts in this order; decode gzip tar",
    }
    (destination / "01-artifact-index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package(args.study, args.out)


if __name__ == "__main__":
    main()
