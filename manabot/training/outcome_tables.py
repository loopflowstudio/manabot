"""Immutable per-update self-play rows carried with their Etude metric run.

Deck identities are table values, never metric names. Each update logs one
content digest, so the metric index cannot multiply columns by deck pair.
Etude's ordinary attempt/update selection chooses the reference to read; an
unreferenced table supplies no outcomes. The SDK ships author artifacts before
their metric references, independently of model checkpoints.
"""

from collections.abc import Sequence
from hashlib import sha256
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Literal

from etude.run.storage import Storage
from pydantic import BaseModel, ConfigDict

from manabot.training.rollout.self_play import SelfPlayOutcome

OUTCOME_TABLE_METRIC = "self_play_outcomes/table"


class OutcomeTable(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    format: Literal["self-play-outcomes-v1"] = "self-play-outcomes-v1"
    rows: list[SelfPlayOutcome]


def _key(digest: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("invalid self-play outcome table digest")
    return f"artifacts/self-play-outcomes/{digest}.json"


def write_outcomes(root: Path, rows: Sequence[SelfPlayOutcome]) -> str:
    """Durably seal one update before logging its digest; retain no history."""
    body = OutcomeTable(rows=list(rows)).model_dump_json().encode()
    digest = sha256(body).hexdigest()
    target = root / _key(digest)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != body:
            raise ValueError("self-play outcome table digest mismatch")
        return digest
    # Both SDK carriers ignore .partial files. Unique staging names keep
    # concurrent writes independent; the context cleans up on failure.
    with NamedTemporaryFile(
        dir=target.parent, suffix=".partial", delete_on_close=False
    ) as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
        # Staging files are owner-only; publish with the run's ordinary
        # umask-derived mode so other readers and carriers see it like metrics.
        os.fchmod(f.fileno(), 0o666 & ~_umask())
        f.close()
        Path(f.name).replace(target)
    return digest


def _umask() -> int:
    # Reading the umask requires setting it; restore immediately.
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def read_outcomes(storage: Storage, prefix: str, digest: str) -> list[SelfPlayOutcome]:
    """Read through the metric block's source, so moved or remote runs still work.

    Missing or damaged referenced evidence fails rather than implying no games.
    """
    body = storage.get(prefix + _key(digest))
    if sha256(body).hexdigest() != digest:
        raise ValueError("self-play outcome table digest mismatch")
    return OutcomeTable.model_validate_json(body).rows
