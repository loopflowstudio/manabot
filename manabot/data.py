"""Experiment data kept in content-addressed S3 rather than in git.

Results that runs write under ``experiments/data/`` are evidence, not source, so
git keeps only ``experiments/data.manifest.json``: each repository-relative path
with the sha256 and size of its bytes. The bytes live once per digest at
``<store>sha256/<digest>`` (``S3ArtifactStore`` owns the S3 protocol).

A machine-wide cache, shared by every worktree, holds one read-only file per
digest at ``$MANABOT_DATA_CACHE/<digest>`` (default
``${XDG_CACHE_HOME:-~/.cache}/manabot/sha256``). It keeps S3 reads within budget:
``publish`` fills it from the local file and reads nothing back, and ``fetch``
downloads a digest only when the cache lacks it, so each machine reads each
digest from S3 at most once. Worktree files are clones or copies of cache files,
never hardlinks, so editing one cannot change the cache.

``fetch`` restores manifest paths byte-for-byte and never overwrites a local file
whose bytes differ. ``publish`` is the only command that uploads. AWS
credentials come from the standard SDK chain.

    uv run --extra artifacts python -m manabot.data fetch
    uv run --extra artifacts python -m manabot.data fetch experiments/data/etu126
    uv run --extra artifacts python -m manabot.data publish experiments/data/new-run
"""

import argparse
from collections.abc import Callable, Iterable
import ctypes
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field

from manabot.infra.artifacts import S3ArtifactStore, StoredArtifact, split_s3_uri

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "data.manifest.json"
DATA = PurePosixPath("experiments/data")


class DataFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class DataManifest(BaseModel):
    """Repository-relative path to the bytes that belong there."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    store: str
    files: dict[str, DataFile]

    def stored(self, entry: DataFile) -> StoredArtifact:
        bucket, prefix = split_s3_uri(self.store)
        key = "/".join(filter(None, (prefix.rstrip("/"), "sha256", entry.sha256)))
        return StoredArtifact(
            uri=f"s3://{bucket}/{key}", sha256=entry.sha256, bytes=entry.bytes
        )

    def select(self, paths: Iterable[str]) -> list[str]:
        """Manifest paths equal to or under each requested path; all when none.

        A request that names nothing in the manifest is an error, not an empty set.
        """
        requested = [_relative(path) for path in paths]
        if not requested:
            return sorted(self.files)
        selected: set[str] = set()
        for request in requested:
            matches = [
                path
                for path in self.files
                if path == request or path.startswith(request.rstrip("/") + "/")
            ]
            if not matches:
                raise ValueError(f"no manifest entry at or under {request}")
            selected.update(matches)
        return sorted(selected)


class DataUnavailable(RuntimeError):
    """The store cannot be read from here: no boto3, credentials, access or network.

    Bytes that arrive but fail verification raise ValueError instead.
    """


class FetchReport(BaseModel):
    """Paths installed by a fetch, and how many digests S3 had to supply."""

    model_config = ConfigDict(frozen=True)
    fetched: list[str]
    present: list[str]
    from_cache: int
    downloaded: int
    downloaded_bytes: int


def load_manifest(path: Path = MANIFEST) -> DataManifest:
    return DataManifest.model_validate_json(path.read_bytes())


def file_entry(path: Path) -> DataFile:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return DataFile(sha256=digest, bytes=path.stat().st_size)


def cache_directory() -> Path:
    configured = os.environ.get("MANABOT_DATA_CACHE")
    if configured:
        return Path(configured).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    return base / "manabot" / "sha256"


def fetch_data(
    paths: Iterable[str] = (), *, manifest: DataManifest | None = None
) -> FetchReport:
    """Install missing manifest paths from the cache, downloading only cache misses.

    Every installed file has been verified against its digest. A corrupt cache
    file raises; it is never installed or silently replaced.
    """
    manifest = manifest or load_manifest()
    selected = manifest.select(paths)
    missing = _missing(selected, manifest)
    cache = cache_directory()
    absent = sorted(
        {
            manifest.files[path]
            for path in missing
            if _cached(cache, manifest.files[path]) is None
        },
        key=lambda entry: entry.sha256,
    )
    if absent:
        _download(absent, manifest, cache)
    for path in missing:
        cached = _cached(cache, manifest.files[path])
        assert cached is not None  # every absent digest was just downloaded
        _install(cached, ROOT / path)
    return FetchReport(
        fetched=missing,
        present=[path for path in selected if path not in missing],
        from_cache=len({manifest.files[path] for path in missing}) - len(absent),
        downloaded=len(absent),
        downloaded_bytes=sum(entry.bytes for entry in absent),
    )


def publish_data(paths: Iterable[str], *, manifest_path: Path = MANIFEST) -> list[str]:
    """Upload local files under experiments/data and record them in the manifest.

    S3 verifies each upload's SHA-256 and never overwrites a digest; nothing is
    read back. The same local bytes fill the cache. Returns the paths whose
    manifest entry was added or changed.
    """
    manifest = load_manifest(manifest_path)
    files: list[str] = []
    for request in paths:
        relative = _relative(request)
        if not PurePosixPath(relative).is_relative_to(DATA):
            raise ValueError(f"experiment data lives under {DATA}/: {request}")
        local = ROOT / relative
        if local.is_dir():
            files.extend(
                item.relative_to(ROOT).as_posix()
                for item in sorted(local.rglob("*"))
                if item.is_file()
            )
        elif local.is_file():
            files.append(relative)
        else:
            raise ValueError(f"no local file or directory at {request}")
    store = S3ArtifactStore()
    cache = cache_directory()
    entries = dict(manifest.files)
    changed: list[str] = []
    for path in files:
        entry = file_entry(ROOT / path)
        if entries.get(path) == entry:
            continue
        store.publish(
            ROOT / path, manifest.store, sha256=entry.sha256, size=entry.bytes
        )
        if _cached(cache, entry) is None:
            _admit(cache, entry, lambda partial: shutil.copyfile(ROOT / path, partial))
        entries[path] = entry
        changed.append(path)
    updated = manifest.model_copy(update={"files": dict(sorted(entries.items()))})
    manifest_path.write_text(updated.model_dump_json(indent=1) + "\n")
    return changed


# -- local files ---------------------------------------------------------------


def _missing(selected: list[str], manifest: DataManifest) -> list[str]:
    """Manifest paths that are absent locally; raises if present bytes differ."""
    missing: list[str] = []
    for path in selected:
        target = ROOT / path
        if not target.exists():
            missing.append(path)
        elif file_entry(target) != manifest.files[path]:
            raise ValueError(
                f"{path} differs from experiments/data.manifest.json; "
                "move it aside or publish it"
            )
    return missing


def _cached(cache: Path, entry: DataFile) -> Path | None:
    path = cache / entry.sha256
    if not path.exists():
        return None
    if file_entry(path) != entry:
        raise ValueError(f"cached experiment data is corrupt; delete {path}")
    return path


def _admit(cache: Path, entry: DataFile, write: Callable[[Path], object]) -> None:
    """Write a digest's bytes to a temporary file, verify, then publish read-only."""
    cache.mkdir(parents=True, exist_ok=True)
    partial = cache / f".{entry.sha256}.{uuid.uuid4().hex}"
    try:
        write(partial)
        if file_entry(partial) != entry:
            raise ValueError(f"bytes for {entry.sha256} differ from the manifest")
        partial.chmod(0o444)
        partial.replace(cache / entry.sha256)
    finally:
        partial.unlink(missing_ok=True)


def _install(cached: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".fetch-{uuid.uuid4().hex}"
    try:
        _clone_or_copy(cached, partial)
        partial.chmod(0o644)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def _clone_or_copy(source: Path, target: Path) -> None:
    """APFS copy-on-write clone where available, else a byte copy; never a link."""
    if sys.platform == "darwin":
        clonefile = ctypes.CDLL(None, use_errno=True).clonefile
        if clonefile(os.fsencode(source), os.fsencode(target), 0) == 0:
            return
    shutil.copyfile(source, target)


# -- the store -----------------------------------------------------------------


def _download(entries: list[DataFile], manifest: DataManifest, cache: Path) -> None:
    try:
        # boto3 is the optional `artifacts` extra; cached or present data needs none.
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as error:
        raise DataUnavailable("fetching needs `uv run --extra artifacts`") from error
    try:
        store = S3ArtifactStore()
        for entry in entries:
            reference = manifest.stored(entry)
            _admit(cache, entry, lambda partial: store.download(reference, partial))
    except BotoCoreError as error:  # no credentials, endpoint or connection
        raise DataUnavailable(str(error)) from error
    except ClientError as error:
        if error.response["ResponseMetadata"]["HTTPStatusCode"] not in (401, 403):
            raise
        raise DataUnavailable(str(error)) from error


def _relative(path: str) -> str:
    """Repository-relative POSIX path for a path given from the root or cwd."""
    candidate = Path(path)
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    return Path(os.path.normpath(absolute)).relative_to(ROOT).as_posix()


def main(argv: list[str] | None = None) -> int:
    # argparse rather than Typer: tests/conftest.py imports this module, and some
    # CI environments deliberately omit Typer.
    parser = argparse.ArgumentParser(
        description="Fetch or publish experiment data in experiments/data.manifest.json"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "fetch", help="Download missing files by digest; verify files already present"
    )
    fetch.add_argument(
        "paths", nargs="*", help="Files or directories; everything when omitted"
    )
    publish = commands.add_parser(
        "publish", help="Upload to the S3 store and update the manifest to commit"
    )
    publish.add_argument("paths", nargs="+", help="Files or directories to upload")
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            report = fetch_data(args.paths)
            print(
                f"fetched {len(report.fetched)} files; {len(report.present)} already "
                f"present; {report.from_cache} digests from the cache, "
                f"{report.downloaded} downloaded ({report.downloaded_bytes} bytes)"
            )
        else:
            changed = publish_data(args.paths)
            for path in changed:
                print(f"published {path}")
            print(f"{len(changed)} manifest entries added or changed")
    except (DataUnavailable, ValueError) as error:
        print(f"{args.command} failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
