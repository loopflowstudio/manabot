"""Content-addressed bucket bytes and a verified local cache.

StoredArtifact is a location, digest and size, not policy admission. A location is
`s3://bucket/key` (AWS) or `r2://bucket/key` (Cloudflare R2, which charges nothing
for downloads). `publish_artifact` never overwrites a key; `download_artifact`
and `fetch_artifact` verify what they read. Consumers verify bytes before
applying their usual world, schema and model checks. AWS credentials come from the
standard SDK chain; R2's from `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID` and
`R2_SECRET_ACCESS_KEY`.
"""

import base64
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import unquote, urlsplit

from etude.run.records import ArtifactRef
from etude.run.storage import storage_from_uri
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    # Artifact verification must not import managed evaluation machinery.
    from etude.eval.records import Player

Scheme = Literal["s3", "r2"]


def split_bucket_uri(uri: str) -> tuple[Scheme, str, str]:
    """The scheme, bucket and key of an `s3://` or `r2://` location."""
    parsed = urlsplit(uri)
    if (
        parsed.scheme not in ("s3", "r2")
        or not parsed.netloc
        or parsed.username is not None
        or ":" in parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "expected an s3:// or r2://bucket/key URI without credentials or query"
        )
    scheme: Scheme = "s3" if parsed.scheme == "s3" else "r2"
    return scheme, parsed.netloc, parsed.path.lstrip("/")


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
        if not split_bucket_uri(value)[2]:
            raise ValueError("artifact URI requires an object key")
        return value


def verify_file(path: Path, sha256: str, size: int) -> None:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if path.stat().st_size != size or digest != sha256:
        raise ValueError(f"artifact bytes differ from receipt: {path}")


def _client(scheme: Scheme, profile: str | None) -> Any:
    """A fresh S3-protocol client; `profile` names the AWS profile for `s3://`.

    boto3's dynamic service clients are untyped, so they never leave this module.
    R2 has no profiles.
    """
    import boto3
    from botocore.config import Config

    if scheme == "s3":
        return boto3.Session(profile_name=profile).client("s3")
    return boto3.client(
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


def publish_artifact(
    path: Path,
    destination: str,
    *,
    sha256: str,
    size: int,
    profile: str | None = None,
) -> StoredArtifact:
    """Upload a stable snapshot that storage verifies; reads no object body.

    The PUT carries the SHA-256, so storage rejects bytes that differ in transit.
    An existing key is confirmed by a HEAD of its stored size and SHA-256.
    Single PUT is deliberately bounded to 5 GiB. Failures leave local evidence
    intact and may leave unreferenced remote blobs, safe to reuse on retry.
    """
    from botocore.exceptions import ClientError

    if size > 5 * 1024**3:
        raise ValueError("artifact exceeds the supported 5 GiB single-PUT limit")
    scheme, bucket, prefix = split_bucket_uri(destination)
    client = _client(scheme, profile)
    key = "/".join(filter(None, (prefix.rstrip("/"), "sha256", sha256)))
    checksum = base64.b64encode(bytes.fromhex(sha256)).decode()
    with tempfile.TemporaryDirectory(prefix="manabot-upload-") as directory:
        snapshot = Path(directory) / "bytes"
        shutil.copyfile(path, snapshot)
        verify_file(snapshot, sha256, size)
        with snapshot.open("rb") as source:
            try:
                result = client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=source,
                    ContentLength=size,
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


def download_artifact(
    reference: StoredArtifact, target: Path, *, profile: str | None = None
) -> str | None:
    """Write verified bytes to target and return the version read, if any."""
    scheme, bucket, key = split_bucket_uri(reference.uri)
    request = {"Bucket": bucket, "Key": key}
    if reference.version_id is not None and scheme == "s3":
        request["VersionId"] = reference.version_id
    response = _client(scheme, profile).get_object(**request)
    body = response["Body"]
    try:
        if response["ContentLength"] != reference.bytes:
            raise ValueError("stored artifact size differs from receipt")
        with target.open("wb") as output:
            shutil.copyfileobj(body, output)
    finally:
        body.close()
    verify_file(target, reference.sha256, reference.bytes)
    version = response.get("VersionId")
    return version if isinstance(version, str) else None


def fetch_artifact(
    reference: StoredArtifact, cache: Path, *, profile: str | None = None
) -> Path:
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
        download_artifact(reference, temporary, profile=profile)
        temporary.replace(target)
    return target


def materialize_artifact(reference: StoredArtifact) -> Path:
    """Resolve a pinned artifact through the environment's credentials and local cache."""
    cache = Path(os.environ.get("MANABOT_ARTIFACT_CACHE", ".runs/artifact-cache"))
    target = cache / reference.sha256
    if target.exists():
        verify_file(target, reference.sha256, reference.bytes)
        return target
    return fetch_artifact(reference, cache)


def local_checkpoint_file(seated: "Player") -> Path:
    """The verified file a model plays; raises for code or a remote checkpoint."""
    if seated.kind != "model":
        raise ValueError("expected a model with a digest and byte count")
    artifact = seated.checkpoint.artifact
    if artifact.sha256 is None or artifact.size_bytes is None:
        raise ValueError("expected a model with a digest and byte count")
    uri = urlsplit(artifact.uri)
    if uri.scheme != "file" or uri.netloc:
        raise ValueError("checkpoint must be a verified local file")
    path = Path(unquote(uri.path))
    verify_file(path, artifact.sha256, artifact.size_bytes)
    return path


def fetch_etude_checkpoint(reference: ArtifactRef, target: Path) -> Path:
    """The verified local bytes of an Etude checkpoint artifact; loads no weights.

    A local file is verified in place and returned. An S3 or R2 object is
    downloaded to `target` through the SDK's storage adapter and verified before
    it is renamed there, so a partial download never becomes a cached policy.
    Each call writes its own partial file, so concurrent fetches of one target
    are safe.
    """
    if reference.sha256 is None or reference.size_bytes is None:
        raise ValueError("Etude checkpoint requires a digest and byte count")
    uri = urlsplit(reference.uri)
    if uri.scheme == "file" and uri.netloc in {"", "localhost"}:
        path = Path(unquote(uri.path))
        verify_file(path, reference.sha256, reference.size_bytes)
        return path
    if uri.scheme not in {"s3", "r2"} or uri.query or uri.fragment or not uri.netloc:
        raise ValueError("Etude checkpoint must name a local file or S3/R2 object")
    storage = storage_from_uri(f"{uri.scheme}://{uri.netloc}")
    key = uri.path.lstrip("/")
    descriptor, name = tempfile.mkstemp(
        prefix=f"{target.name}.", suffix=".partial", dir=target.parent
    )
    partial = Path(name)
    with os.fdopen(descriptor, "wb") as file:
        offset = 0
        while offset < reference.size_bytes:
            length = min(8 * 1024 * 1024, reference.size_bytes - offset)
            block = storage.get_range(key, offset, length)
            if len(block) != length:
                raise ValueError("incomplete Etude checkpoint download")
            file.write(block)
            offset += length
    verify_file(partial, reference.sha256, reference.size_bytes)
    partial.replace(target)
    return target


def _environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"R2 access needs {name} in the environment")
    return value
