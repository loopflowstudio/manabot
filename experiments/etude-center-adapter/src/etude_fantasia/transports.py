"""Restricted local Git/files and GET-only S3 transports; no traversal or cache."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from etude_client.collector import ObjectRef
from etude_client.errors import EtudeError
from etude_client.models import MAX_BYTES, sha256

from .values import obj, string


class LocalSource:
    """Explicit revision reads committed objects; default reads only enumerated files.

    Dirty working bytes have no revision attribution. No git operation writes.
    """

    def __init__(self, root: Path, revision: str | None = None) -> None:
        self.root = root.resolve()
        self.pinned = revision is not None
        result = self.git(
            "rev-parse", "--verify", "--end-of-options", (revision or "HEAD") + "^{commit}"
        )
        if revision is not None and result.returncode:
            raise EtudeError("source_revision_unavailable", 422)
        self.revision = result.stdout.decode().strip() if result.returncode == 0 else None

    def git(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", "-C", str(self.root), *args], capture_output=True, check=False, timeout=30
        )

    def select(self, key: str, *, deadline: float) -> tuple[ObjectRef, bytes]:
        if time.monotonic() >= deadline:
            raise EtudeError("timeout", 504)
        if Path(key).is_absolute() or ".." in Path(key).parts:
            raise EtudeError("source_path_denied", 403)
        identity = f"{self.revision}:{key}"
        size = self.git("cat-file", "-s", identity) if self.revision else None
        if size is not None and size.returncode == 0 and int(size.stdout) > MAX_BYTES:
            raise EtudeError("source_too_large", 413)
        committed = self.git("show", identity) if self.revision else None
        if self.pinned:
            if committed is None or committed.returncode:
                raise EtudeError("source_missing", 404)
            data = committed.stdout
        else:
            path = (self.root / key).resolve()
            if not path.is_relative_to(self.root):
                raise EtudeError("source_path_denied", 403)
            try:
                with path.open("rb") as stream:
                    data = stream.read(MAX_BYTES + 1)
            except FileNotFoundError:
                raise EtudeError("source_missing", 404) from None
        revision = (
            self.revision
            if committed is not None and committed.returncode == 0 and committed.stdout == data
            else None
        )
        return ObjectRef(key=key, version=revision or "working", sha256=sha256(data)), data


class S3Source:
    def __init__(self, destination: str, jobs: tuple[str, ...]) -> None:
        uri = urlsplit(destination)
        if (
            uri.scheme != "s3"
            or not uri.netloc
            or uri.query
            or uri.fragment
            or not uri.path.strip("/")
            or len(jobs) > 100
            or not jobs
            or len(set(jobs)) != len(jobs)
            or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,199}", j) for j in jobs)
            or any(p in {".", ".."} for p in uri.path.split("/"))
        ):
            raise EtudeError("invalid", 422)
        self.prefixes = tuple(destination.rstrip("/") + "/" + job for job in jobs)

    def validate_key(self, key: str, version: str | None) -> None:
        uri = urlsplit(key)
        if not any(
            key == p + "/spec.json" or key.startswith(p + "/runtime/") for p in self.prefixes
        ) or any(p in {".", ".."} for p in uri.path.split("/")):
            raise EtudeError("source_not_enumerated", 403)
        if uri.query or uri.fragment or version in {"null", ""}:
            raise EtudeError("invalid", 422)

    @contextmanager
    def download(
        self, key: str, version: str | None, *, deadline: float, large_run: bool = False
    ) -> Iterator[tuple[str, Path]]:
        """Expose one private, version-checked file only until its consumer finishes."""
        self.validate_key(key, version)
        uri = urlsplit(key)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EtudeError("timeout", 504)
        with tempfile.TemporaryDirectory(prefix="etude-observe-") as directory:
            path = Path(directory) / "object"
            command = [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                uri.netloc,
                "--key",
                uri.path.lstrip("/"),
                "--cli-connect-timeout",
                "5",
                "--cli-read-timeout",
                "10",
                "--no-cli-pager",
                "--output",
                "json",
            ]
            if not large_run:
                command += ["--range", f"bytes=0-{MAX_BYTES}"]
            if version is not None:
                command += ["--version-id", version]
            command.append(str(path))
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    timeout=min(1200 if large_run else 30, remaining),
                    check=False,
                )
                if result.returncode:
                    raise EtudeError("source_read_failed", 503)
                metadata = obj(json.loads(result.stdout))
                returned_version = string(metadata["VersionId"])
                if returned_version in {"", "null"} or (
                    version is not None and returned_version != version
                ):
                    raise EtudeError("invalid", 422)
                yield returned_version, path
            except subprocess.TimeoutExpired:
                raise EtudeError("timeout", 504) from None
            except (OSError, ValueError, KeyError, TypeError):
                raise EtudeError("source_read_failed", 503) from None

    def select(
        self, key: str, *, deadline: float, version: str | None = None
    ) -> tuple[ObjectRef, bytes]:
        with self.download(key, version, deadline=deadline) as (selected, path):
            with path.open("rb") as stream:
                raw = stream.read(MAX_BYTES + 1)
        return ObjectRef(key=key, version=selected, sha256=sha256(raw)), raw

    def read(self, key: str, version: str, *, deadline: float) -> bytes:
        return self.select(key, version=version, deadline=deadline)[1]
