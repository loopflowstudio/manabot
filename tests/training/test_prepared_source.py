"""Prepared transport preserves source identity and limits intercepted writes."""

import gzip
import hashlib
from pathlib import Path
from typing import TypedDict

import boto3
from etude.launch.storage.s3 import Bucket, Keys
import pytest

from experiments.runners.prepared_source import PreparedSource, install_prepared_sources


class _Metadata(TypedDict):
    ContentLength: int
    Metadata: dict[str, str]


class _Storage:
    def __init__(self, receipt: PreparedSource) -> None:
        self.copies: list[dict[str, object]] = []
        self.metadata: _Metadata = {
            "ContentLength": receipt.bytes,
            "Metadata": {"sha256": receipt.sha256},
        }

    def copy_object(self, **kwargs: object) -> None:
        self.copies.append(kwargs)

    def head_object(self, **kwargs: object) -> _Metadata:
        return self.metadata


def test_prepared_gzip_checks_source_before_copy_and_preserves_other_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_tar = b"exact git tar bytes"
    compressed = gzip.compress(original_tar, mtime=0)
    receipt = PreparedSource(
        status="verified",
        bucket="campaign",
        key="prepared/source.tar.gz",
        commit="a" * 40,
        bytes=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(),
        transport="gzip",
        source_tar_bytes=len(original_tar),
        source_tar_sha256=hashlib.sha256(original_tar).hexdigest(),
    )
    path = tmp_path / "receipt.json"
    path.write_text(receipt.model_dump_json())
    storage = _Storage(receipt)

    def client(*args: object, **kwargs: object) -> _Storage:
        return storage

    passed: list[tuple[str, bytes, bool]] = []

    def original_put(
        bucket: Bucket, key: str, body: bytes, *, once: bool = False
    ) -> None:
        passed.append((key, body, once))

    monkeypatch.setattr(boto3, "client", client)
    monkeypatch.setattr(Bucket, "put", original_put)
    install_prepared_sources([path], prefix="campaign/")
    bucket = Bucket(
        "campaign", "us-west-2", Keys(access_key="unused", secret_key="unused")
    )
    key = f"campaign/cohort/sources/{receipt.commit}.tar"
    with pytest.raises(ValueError, match="source bytes differ"):
        bucket.put(key, b"different source")
    assert storage.copies == []
    bucket.put(key, original_tar)
    assert storage.copies == [
        {
            "Bucket": "campaign",
            "Key": key,
            "CopySource": {"Bucket": "campaign", "Key": receipt.key},
        }
    ]
    bucket.put(key, original_tar, once=True)
    other = key.replace("campaign/", "another/", 1)
    bucket.put(other, original_tar)
    assert passed == [(key, original_tar, True), (other, original_tar, False)]
    storage.metadata["Metadata"]["sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="copied source receipt differs"):
        bucket.put(key, original_tar)


def test_gzip_receipt_requires_uncompressed_identity() -> None:
    with pytest.raises(ValueError, match="original source tar identity"):
        PreparedSource(
            status="verified",
            bucket="campaign",
            key="prepared/source.tar.gz",
            commit="a" * 40,
            bytes=10,
            sha256="b" * 64,
            transport="gzip",
        )
