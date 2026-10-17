"""Reuse verified source objects without changing an experiment's source identity.

Etude's source publication seam receives the original git tar. This launcher-only
adapter checks those exact bytes before copying a prepared object into the new
cohort. A gzip receipt separately identifies the original tar and its transport
bytes; Etude's worker detects gzip even when the object name ends in ``.tar``.
The uploader must verify the complete object and decompression before issuing
the receipt. This adapter checks the copied object's size and digest metadata.
"""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PreparedSource(BaseModel):
    """A verified object, plus the uncompressed source identity when compressed."""

    status: Literal["verified"]
    bucket: str = Field(min_length=1)
    key: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    transport: Literal["identity", "gzip"] = "identity"
    source_tar_bytes: int | None = Field(default=None, gt=0)
    source_tar_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def check_source_identity(self) -> "PreparedSource":
        if self.transport == "gzip" and (
            self.source_tar_bytes is None or self.source_tar_sha256 is None
        ):
            raise ValueError("gzip receipts require the original source tar identity")
        if self.transport == "identity" and (
            self.source_tar_bytes not in (None, self.bytes)
            or self.source_tar_sha256 not in (None, self.sha256)
        ):
            raise ValueError("uncompressed receipt has conflicting source identities")
        return self

    def verify_source(self, body: bytes) -> None:
        size = self.source_tar_bytes or self.bytes
        digest = self.source_tar_sha256 or self.sha256
        if len(body) != size or hashlib.sha256(body).hexdigest() != digest:
            raise ValueError("source bytes differ from the verified prepared object")


def install_prepared_sources(receipts: list[Path], *, prefix: str) -> None:
    """Intercept only admitted revisions under one launcher's object prefix.

    Install once before queueing work. Unmatched writes retain the SDK behavior;
    a matching source with different bytes fails before any cloud mutation.
    This is a process-local adapter for Etude 0.1.1, not a storage migration.
    """
    import boto3
    from botocore.config import Config
    from etude.launch.storage.s3 import Bucket

    if not prefix or not prefix.endswith("/"):
        raise ValueError("a nonempty object prefix ending in / is required")
    sources = [
        PreparedSource.model_validate_json(path.read_text()) for path in receipts
    ]
    identities = {(source.bucket, source.commit) for source in sources}
    if len(identities) != len(sources):
        raise ValueError("duplicate prepared source revisions")
    client = boto3.client(
        "s3",
        config=Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 1}),
    )
    original = Bucket.put

    def put(bucket: Bucket, key: str, body: bytes, *, once: bool = False) -> None:
        for source in sources:
            if (
                not once
                and bucket.name == source.bucket
                and key.startswith(prefix)
                and key.endswith(f"/sources/{source.commit}.tar")
            ):
                source.verify_source(body)
                client.copy_object(
                    Bucket=bucket.name,
                    Key=key,
                    CopySource={"Bucket": source.bucket, "Key": source.key},
                )
                metadata = client.head_object(Bucket=bucket.name, Key=key)
                if (
                    metadata.get("ContentLength") != source.bytes
                    or metadata.get("Metadata", {}).get("sha256") != source.sha256
                ):
                    raise RuntimeError("copied source receipt differs")
                return
        original(bucket, key, body, once=once)

    Bucket.put = put
