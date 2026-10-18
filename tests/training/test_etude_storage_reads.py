"""Storage uncertainty must preserve the existing attempt and truthful absence."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from etude.launch.storage.s3 import Bucket, Denied, StorageUnavailable
import pytest

from experiments.runners.etude_storage_reads import protect_storage_reads


def _bucket() -> Bucket:
    # Only the SDK read transport is replaced; no credentials or network are used.
    return cast(Bucket, SimpleNamespace(name="test-bucket"))


@pytest.mark.parametrize("value", (b"complete receipt", None))
def test_same_read_recovers_without_inventing_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: bytes | None
) -> None:
    calls: list[tuple[Bucket, str]] = []
    pauses: list[float] = []
    bucket = _bucket()

    def get(target: Bucket, key: str) -> bytes | None:
        calls.append((target, key))
        if len(calls) < 3:
            raise StorageUnavailable(f"GET {key}: no reply")
        return value

    monkeypatch.setattr(Bucket, "get", get)
    write = Bucket.put
    receipt = tmp_path / "reads.jsonl"
    with protect_storage_reads(receipt, pause=pauses.append):
        assert Bucket.get(bucket, "session/out/receipt") == value
        assert Bucket.put is write
    assert Bucket.get is get
    assert calls == [(bucket, "session/out/receipt")] * 3
    assert pauses == [1.0, 2.0]
    events = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [event["attempt"] for event in events] == [1, 2]
    assert all(event["phase"] == "storage_read_retry" for event in events)


def test_exhaustion_raises_original_failure_and_restores_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = StorageUnavailable("GET session/out/receipt: no reply")
    calls = 0

    def get(bucket: Bucket, key: str) -> bytes | None:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(Bucket, "get", get)
    receipt = tmp_path / "reads.jsonl"
    with pytest.raises(StorageUnavailable) as caught:
        with protect_storage_reads(receipt, pause=lambda _: None):
            Bucket.get(_bucket(), "session/out/receipt")
    assert caught.value is failure and calls == 3
    assert Bucket.get is get
    events = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert events[-1]["phase"] == "storage_read_exhausted"
    assert events[-1]["attempt"] == 3


@pytest.mark.parametrize(
    "failure",
    (
        Denied("GET key: status 403"),
        StorageUnavailable("GET key: malformed response"),
        StorageUnavailable("PUT key: no reply"),
        StorageUnavailable("POST key: no reply"),
        OSError("provider creation may have happened"),
    ),
)
def test_other_failures_are_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    calls = 0
    pauses: list[float] = []

    def get(bucket: Bucket, key: str) -> bytes | None:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(Bucket, "get", get)
    receipt = tmp_path / "reads.jsonl"
    with pytest.raises(type(failure)) as caught:
        with protect_storage_reads(receipt, pause=pauses.append):
            Bucket.get(_bucket(), "key")
    assert caught.value is failure and calls == 1
    assert pauses == [] and not receipt.exists()
    assert Bucket.get is get
