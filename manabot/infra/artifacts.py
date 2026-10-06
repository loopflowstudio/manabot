"""Content-addressed S3 bytes and a verified local cache.

StoredArtifact is a location, digest and size, not policy admission. S3ArtifactStore
never overwrites a key. Consumers verify bytes before applying their usual world,
schema and model checks. AWS credentials come only from the standard SDK chain.
"""

import base64
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.username is not None
        or ":" in parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("expected an s3://bucket/key URI without credentials or query")
    return parsed.netloc, parsed.path.lstrip("/")


class StoredArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    version_id: str | None = None

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str) -> str:
        if not split_s3_uri(value)[1]:
            raise ValueError("artifact URI requires an object key")
        return value


def verify_file(path: Path, sha256: str, size: int) -> None:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if path.stat().st_size != size or digest != sha256:
        raise ValueError(f"artifact bytes differ from receipt: {path}")


class S3ArtifactStore:
    def __init__(self, *, profile: str | None = None) -> None:
        import boto3

        # boto3's dynamic service client is untyped; keep it inside this adapter.
        self._client: Any = boto3.Session(profile_name=profile).client("s3")

    def publish(
        self, path: Path, destination: str, *, sha256: str, size: int
    ) -> StoredArtifact:
        """Upload a stable snapshot, then read it back; retries verify existing bytes.

        Single PUT is deliberately bounded to 5 GiB. Failures leave local evidence
        intact and may leave unreferenced remote blobs, safe to reuse on retry.
        """
        from botocore.exceptions import ClientError

        if size > 5 * 1024**3:
            raise ValueError("artifact exceeds the supported 5 GiB single-PUT limit")
        bucket, prefix = split_s3_uri(destination)
        key = "/".join(filter(None, (prefix.rstrip("/"), "sha256", sha256)))
        with tempfile.TemporaryDirectory(prefix="manabot-upload-") as directory:
            snapshot = Path(directory) / "bytes"
            shutil.copyfile(path, snapshot)
            verify_file(snapshot, sha256, size)
            with snapshot.open("rb") as source:
                try:
                    result = self._client.put_object(
                        Bucket=bucket,
                        Key=key,
                        Body=source,
                        ContentLength=size,
                        ChecksumSHA256=base64.b64encode(bytes.fromhex(sha256)).decode(),
                        IfNoneMatch="*",
                    )
                    version = result.get("VersionId")
                except ClientError as error:
                    if error.response["ResponseMetadata"]["HTTPStatusCode"] != 412:
                        raise
                    version = None
            reference = StoredArtifact(
                uri=f"s3://{bucket}/{key}",
                sha256=sha256,
                bytes=size,
                version_id=version,
            )
            # Readback checks actual bytes, never multipart ETags or user metadata.
            downloaded_version = self._download(reference, Path(directory) / "readback")
            return reference.model_copy(
                update={"version_id": reference.version_id or downloaded_version}
            )

    def _download(self, reference: StoredArtifact, target: Path) -> str | None:
        bucket, key = split_s3_uri(reference.uri)
        request = {"Bucket": bucket, "Key": key}
        if reference.version_id is not None:
            request["VersionId"] = reference.version_id
        response = self._client.get_object(**request)
        body = response["Body"]
        try:
            if response["ContentLength"] != reference.bytes:
                raise ValueError("S3 artifact size differs from receipt")
            with target.open("wb") as output:
                shutil.copyfileobj(body, output)
        finally:
            body.close()
        verify_file(target, reference.sha256, reference.bytes)
        version = response.get("VersionId")
        return version if isinstance(version, str) else None

    def fetch(self, reference: StoredArtifact, cache: Path) -> Path:
        """Validate cache hits; atomically install verified downloads.

        A corrupt cache fails explicitly. It is never loaded or silently trusted.
        """
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / reference.sha256
        if target.exists():
            verify_file(target, reference.sha256, reference.bytes)
            return target
        with tempfile.TemporaryDirectory(dir=cache, prefix="download-") as directory:
            temporary = Path(directory) / "bytes"
            self._download(reference, temporary)
            temporary.replace(target)
        return target


def materialize_artifact(reference: StoredArtifact) -> Path:
    """Resolve a pinned artifact using the configured AWS profile and local cache."""
    cache = Path(os.environ.get("MANABOT_ARTIFACT_CACHE", ".runs/artifact-cache"))
    target = cache / reference.sha256
    if target.exists():
        verify_file(target, reference.sha256, reference.bytes)
        return target
    return S3ArtifactStore().fetch(reference, cache)
