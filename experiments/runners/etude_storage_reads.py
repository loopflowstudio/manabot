"""Keep a transient receipt-read failure from abandoning a running campaign.

The launcher installs protect_storage_reads around its existing scheduler. Only
Etude 0.1.2 GET requests that received no reply are retried, at most three times.
Writes, provider creation, denied access and unreadable responses still fail.
Unknown storage state never becomes evidence that a run or artifact is absent.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
import time


@contextmanager
def protect_storage_reads(
    receipt: Path, *, pause: Callable[[float], None] = time.sleep
) -> Iterator[None]:
    """Install a process-local, bounded retry for the pinned SDK's read boundary."""
    from etude.launch.storage.s3 import Bucket, StorageUnavailable

    original = Bucket.get

    def get(bucket: Bucket, key: str) -> bytes | None:
        for attempt in range(1, 4):
            try:
                return original(bucket, key)
            except StorageUnavailable as error:
                if not str(error).startswith("GET ") or not str(error).endswith(
                    ": no reply"
                ):
                    raise
                receipt.parent.mkdir(parents=True, exist_ok=True)
                with receipt.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "phase": "storage_read_exhausted"
                                if attempt == 3
                                else "storage_read_retry",
                                "at": datetime.now(UTC).isoformat(),
                                "bucket": bucket.name,
                                "key": key,
                                "attempt": attempt,
                            }
                        )
                        + "\n"
                    )
                if attempt == 3:
                    raise
                pause(float(attempt))
        raise AssertionError("Unreachable after the bounded storage read")

    Bucket.get = get
    try:
        yield
    finally:
        Bucket.get = original
