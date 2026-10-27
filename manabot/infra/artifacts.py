<<<<<<< HEAD
"""Content-addressed S3 bytes and a verified local cache.

StoredArtifact is a location, digest and size, not policy admission. S3ArtifactStore
never overwrites a key. Consumers verify bytes before applying their usual world,
schema and model checks. AWS credentials come only from the standard SDK chain.
=======
"""Content-addressed bucket bytes and a verified local cache.

StoredArtifact is a location, digest and size, not policy admission. A location is
`s3://bucket/key` (AWS) or `r2://bucket/key` (Cloudflare R2, which charges nothing
for downloads). ArtifactStore never overwrites a key. Consumers verify bytes before
applying their usual world, schema and model checks. AWS credentials come from the
standard SDK chain; R2's from `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID` and
`R2_SECRET_ACCESS_KEY`.
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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

<<<<<<< HEAD

def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
=======
Scheme = Literal["s3", "r2"]


def split_bucket_uri(uri: str) -> tuple[Scheme, str, str]:
    """The scheme, bucket and key of an `s3://` or `r2://` location."""
    parsed = urlsplit(uri)
    if (
        parsed.scheme not in ("s3", "r2")
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        or not parsed.netloc
        or parsed.username is not None
        or ":" in parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
<<<<<<< HEAD
        raise ValueError("expected an s3://bucket/key URI without credentials or query")
    return parsed.netloc, parsed.path.lstrip("/")
=======
        raise ValueError(
            "expected an s3:// or r2://bucket/key URI without credentials or query"
        )
    scheme: Scheme = "s3" if parsed.scheme == "s3" else "r2"
    return scheme, parsed.netloc, parsed.path.lstrip("/")
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f


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
<<<<<<< HEAD
        if not split_s3_uri(value)[1]:
=======
        if not split_bucket_uri(value)[2]:
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
            raise ValueError("artifact URI requires an object key")
        return value


def verify_file(path: Path, sha256: str, size: int) -> None:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if path.stat().st_size != size or digest != sha256:
        raise ValueError(f"artifact bytes differ from receipt: {path}")


<<<<<<< HEAD
class S3ArtifactStore:
    def __init__(self, *, profile: str | None = None) -> None:
        import boto3

        # boto3's dynamic service client is untyped; keep it inside this adapter.
        self._client: Any = boto3.Session(profile_name=profile).client("s3")
=======
class ArtifactStore:
    """S3 and R2 through one S3-protocol client per scheme, made on first use.

    `profile` names the AWS profile for `s3://` locations; R2 has no profiles.
    """

    def __init__(self, *, profile: str | None = None) -> None:
        self._profile = profile
        # boto3's dynamic service clients are untyped; keep them inside this adapter.
        self._clients: dict[Scheme, Any] = {}

    def _client(self, scheme: Scheme) -> Any:
        if scheme not in self._clients:
            import boto3
            from botocore.config import Config

            if scheme == "s3":
                client = boto3.Session(profile_name=self._profile).client("s3")
            else:
                client = boto3.client(
                    "s3",
                    endpoint_url=f"https://{_environment('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
                    aws_access_key_id=_environment("R2_ACCESS_KEY_ID"),
                    aws_secret_access_key=_environment("R2_SECRET_ACCESS_KEY"),
                    region_name="auto",
                    # The explicit SHA-256 is the only checksum sent; R2 checks it.
                    config=Config(
                        request_checksum_calculation="when_required",
                        response_checksum_validation="when_required",
                    ),
                )
            self._clients[scheme] = client
        return self._clients[scheme]
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f

    def publish(
        self, path: Path, destination: str, *, sha256: str, size: int
    ) -> StoredArtifact:
<<<<<<< HEAD
        """Upload a stable snapshot, then read it back; retries verify existing bytes.

=======
        """Upload a stable snapshot that storage verifies; reads no object body.

        The PUT carries the SHA-256, so storage rejects bytes that differ in transit.
        An existing key is confirmed by a HEAD of its stored size and SHA-256.
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        Single PUT is deliberately bounded to 5 GiB. Failures leave local evidence
        intact and may leave unreferenced remote blobs, safe to reuse on retry.
        """
        from botocore.exceptions import ClientError

        if size > 5 * 1024**3:
            raise ValueError("artifact exceeds the supported 5 GiB single-PUT limit")
<<<<<<< HEAD
        bucket, prefix = split_s3_uri(destination)
        key = "/".join(filter(None, (prefix.rstrip("/"), "sha256", sha256)))
=======
        scheme, bucket, prefix = split_bucket_uri(destination)
        client = self._client(scheme)
        key = "/".join(filter(None, (prefix.rstrip("/"), "sha256", sha256)))
        checksum = base64.b64encode(bytes.fromhex(sha256)).decode()
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        with tempfile.TemporaryDirectory(prefix="manabot-upload-") as directory:
            snapshot = Path(directory) / "bytes"
            shutil.copyfile(path, snapshot)
            verify_file(snapshot, sha256, size)
            with snapshot.open("rb") as source:
                try:
<<<<<<< HEAD
                    result = self._client.put_object(
=======
                    result = client.put_object(
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
                        Bucket=bucket,
                        Key=key,
                        Body=source,
                        ContentLength=size,
<<<<<<< HEAD
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
=======
                        ChecksumSHA256=checksum,
                        IfNoneMatch="*",
                    )
                except ClientError as error:
                    if error.response["ResponseMetadata"]["HTTPStatusCode"] != 412:
                        raise
                    result = client.head_object(
                        Bucket=bucket, Key=key, ChecksumMode="ENABLED"
                    )
                    if (
                        result["ContentLength"] != size
                        or result.get("ChecksumSHA256") != checksum
                    ):
                        raise ValueError(
                            f"existing object differs from receipt: {key}"
                        ) from error
        # R2 answers a PUT with a version ID but cannot GET by one, so only S3
        # versions are kept. Keys are digests and never overwritten either way.
        version = result.get("VersionId") if scheme == "s3" else None
        return StoredArtifact(
            uri=f"{scheme}://{bucket}/{key}",
            sha256=sha256,
            bytes=size,
            version_id=version if isinstance(version, str) else None,
        )

    def download(self, reference: StoredArtifact, target: Path) -> str | None:
        """Write verified bytes to target and return the version read, if any."""
        scheme, bucket, key = split_bucket_uri(reference.uri)
        request = {"Bucket": bucket, "Key": key}
        if reference.version_id is not None and scheme == "s3":
            request["VersionId"] = reference.version_id
        response = self._client(scheme).get_object(**request)
        body = response["Body"]
        try:
            if response["ContentLength"] != reference.bytes:
                raise ValueError("stored artifact size differs from receipt")
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
<<<<<<< HEAD
            self._download(reference, temporary)
=======
            self.download(reference, temporary)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
            temporary.replace(target)
        return target


def materialize_artifact(reference: StoredArtifact) -> Path:
<<<<<<< HEAD
    """Resolve a pinned artifact using the configured AWS profile and local cache."""
=======
    """Resolve a pinned artifact through the environment's credentials and local cache."""
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    cache = Path(os.environ.get("MANABOT_ARTIFACT_CACHE", ".runs/artifact-cache"))
    target = cache / reference.sha256
    if target.exists():
        verify_file(target, reference.sha256, reference.bytes)
        return target
<<<<<<< HEAD
    return S3ArtifactStore().fetch(reference, cache)
=======
    return ArtifactStore().fetch(reference, cache)


def _environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"R2 access needs {name} in the environment")
    return value
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
