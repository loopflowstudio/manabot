"""Experiment data kept in content-addressed S3 rather than in git.

Results that runs write under ``experiments/data/`` are evidence, not source, so
git keeps only ``experiments/data.manifest.json``: each repository-relative path
with the sha256 and size of its bytes. The bytes live once per digest at
``<store>sha256/<digest>`` (``S3ArtifactStore`` owns the S3 protocol).

``fetch`` restores manifest paths byte-for-byte and never overwrites a local file
whose bytes differ. ``publish`` is the only command that uploads: it stores new
or changed local files and records them in the manifest. AWS credentials come
from the standard SDK chain.

    uv run --extra artifacts python -m manabot.data fetch
    uv run --extra artifacts python -m manabot.data fetch experiments/data/etu126
    uv run --extra artifacts python -m manabot.data publish experiments/data/new-run
"""

import argparse
from collections.abc import Iterable
import hashlib
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Literal

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

    def stored(self, path: str) -> StoredArtifact:
        entry = self.files[path]
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
    model_config = ConfigDict(frozen=True)
    fetched: list[str]
    present: list[str]
    fetched_bytes: int


def load_manifest(path: Path = MANIFEST) -> DataManifest:
    return DataManifest.model_validate_json(path.read_bytes())


def file_entry(path: Path) -> DataFile:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return DataFile(sha256=digest, bytes=path.stat().st_size)


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


def fetch_data(
    paths: Iterable[str] = (), *, manifest: DataManifest | None = None
) -> FetchReport:
    """Download missing manifest paths, verifying each digest before it appears."""
    manifest = manifest or load_manifest()
    selected = manifest.select(paths)
    missing = _missing(selected, manifest)
    if missing:
        _download(missing, manifest)
    return FetchReport(
        fetched=missing,
        present=[path for path in selected if path not in missing],
        fetched_bytes=sum(manifest.files[path].bytes for path in missing),
    )


def publish_data(paths: Iterable[str], *, manifest_path: Path = MANIFEST) -> list[str]:
    """Upload local files under experiments/data and record them in the manifest.

    The store never overwrites a digest and reads every upload back. Returns the
    paths whose manifest entry was added or changed.
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
    entries = dict(manifest.files)
    changed: list[str] = []
    for path in files:
        entry = file_entry(ROOT / path)
        if entries.get(path) == entry:
            continue
        store.publish(
            ROOT / path, manifest.store, sha256=entry.sha256, size=entry.bytes
        )
        entries[path] = entry
        changed.append(path)
    updated = manifest.model_copy(update={"files": dict(sorted(entries.items()))})
    manifest_path.write_text(updated.model_dump_json(indent=1) + "\n")
    return changed


def _download(paths: list[str], manifest: DataManifest) -> None:
    try:
        # boto3 is the optional `artifacts` extra; reads of present data need none.
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as error:
        raise DataUnavailable("fetching needs `uv run --extra artifacts`") from error
    try:
        store = S3ArtifactStore()
        for path in paths:
            target = ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            # Download beside the target so the final rename is atomic.
            descriptor, partial = tempfile.mkstemp(dir=target.parent, prefix=".fetch-")
            os.close(descriptor)
            try:
                store.download(manifest.stored(path), Path(partial))
                Path(partial).replace(target)
            finally:
                Path(partial).unlink(missing_ok=True)
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
                f"fetched {len(report.fetched)} files ({report.fetched_bytes} bytes); "
                f"{len(report.present)} already present"
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
