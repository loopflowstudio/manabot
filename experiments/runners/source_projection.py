"""Package training sources without the repository's historical experiment data.

Only ``experiments/data/`` is omitted. Code, recipes, content, native sources,
build inputs, tests and other assets keep their original bytes and tar metadata.
The receipt lists retained and omitted file hashes and identifies both archives.
This is an explicit source projection, not lossless compression of the full tar.
Full source archives remain the authoritative retained provenance.
"""

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path, PurePosixPath
import tarfile

from pydantic import BaseModel


class SourceProjection(BaseModel):
    commit: str
    source_tar_bytes: int
    source_tar_sha256: str
    projected_tar_bytes: int
    projected_tar_sha256: str
    bytes: int
    sha256: str
    retained_files: dict[str, str]
    omitted_files: dict[str, str]


@dataclass(frozen=True)
class _Member:
    mode: int
    kind: bytes
    linkname: str
    size: int
    digest: str | None


def _members(archive: Path) -> dict[str, _Member]:
    result: dict[str, _Member] = {}
    with tarfile.open(archive) as source:
        for item in source:
            path = PurePosixPath(item.name)
            if path.is_absolute() or ".." in path.parts or item.name in result:
                raise ValueError("unsafe or duplicate source archive member")
            digest = None
            if item.isfile():
                stream = source.extractfile(item)
                assert stream is not None
                with stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
            result[item.name] = _Member(
                item.mode, item.type, item.linkname, item.size, digest
            )
    return result


def project_source(source: Path, destination: Path, *, commit: str) -> SourceProjection:
    """Create and verify a new gzip tar, rejecting any unexpected membership change."""
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("an exact source commit is required")
    original = _members(source)
    retained = {
        name: value
        for name, value in original.items()
        if name != "experiments/data" and not name.startswith("experiments/data/")
    }
    if not retained or retained == original:
        raise ValueError(
            "source projection must retain sources and omit historical data"
        )
    with (
        tarfile.open(source) as archive,
        destination.open("xb") as output,
        gzip.GzipFile(
            fileobj=output, mode="wb", compresslevel=1, mtime=0, filename=""
        ) as zipped,
        tarfile.open(fileobj=zipped, mode="w|") as projected,
    ):
        for item in archive:
            if item.name not in retained:
                continue
            if item.isfile():
                stream = archive.extractfile(item)
                assert stream is not None
                with stream:
                    projected.addfile(item, stream)
            else:
                projected.addfile(item)
    if _members(destination) != retained:
        raise ValueError("projected source changed retained members")
    with source.open("rb") as stream:
        original_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with destination.open("rb") as stream:
        compressed_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    projected_digest = hashlib.sha256()
    projected_bytes = 0
    with gzip.open(destination, "rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            projected_bytes += len(chunk)
            projected_digest.update(chunk)
    return SourceProjection(
        commit=commit,
        source_tar_bytes=source.stat().st_size,
        source_tar_sha256=original_digest,
        projected_tar_bytes=projected_bytes,
        projected_tar_sha256=projected_digest.hexdigest(),
        bytes=destination.stat().st_size,
        sha256=compressed_digest,
        retained_files={
            name: value.digest
            for name, value in retained.items()
            if value.digest is not None
        },
        omitted_files={
            name: value.digest
            for name, value in original.items()
            if name not in retained and value.digest is not None
        },
    )
