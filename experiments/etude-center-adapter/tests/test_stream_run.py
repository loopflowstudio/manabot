from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from etude_client.collector import ObjectRef
from etude_client.errors import EtudeError
from etude_client.models import sha256
from etude_fantasia.stream_run import read_run


def test_run_stream_keeps_all_origins_and_validates_discarded_bytes(tmp_path: Path) -> None:
    origins = [
        {"run_id": "祖先", "iteration": 63000, "extra": {"digests": ["a", "b"]}},
        {"run_id": "later", "iteration": 100000},
    ]
    raw = json.dumps(
        {
            "id": "run",
            "schema_version": 1,
            "stages": [
                {"learning_state_origin": origin, "series": [{"step": i} for i in range(20000)]}
                for origin in origins
            ],
        },
        ensure_ascii=False,
    ).encode()
    path = tmp_path / "run.json"
    path.write_bytes(raw)
    ref = ObjectRef(key="s3://fixture/run", version="version", sha256=sha256(raw))
    result = read_run(path, ref, len(raw), time.monotonic() + 30)
    assert result.summary == {
        "id": "run",
        "schema_version": 1,
        "stages": [{"learning_state_origin": origin} for origin in origins],
    }
    assert result.reference == ref and result.size_bytes == len(raw)
    path.write_bytes(raw[:-2] + b"  ")
    with pytest.raises(EtudeError):
        read_run(path, ref, len(raw), time.monotonic() + 30)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"id":"x","id":"y"}',
        b'{"ignored":[1,]}',
        b'{"ignored":NaN}',
        b'{"ignored":01}',
        b'{"ignored":"\\uZZZZ"}',
        b'{"ignored":"\xff"}',
        b'{"ignored": [1 2]}',
        b"{} trailing",
        b'{"ignored": {"x":1,"x":2}}',
    ],
)
def test_invalid_discarded_json_is_rejected(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "run.json"
    path.write_bytes(raw)
    ref = ObjectRef(key="s3://fixture/run", version="version", sha256=sha256(raw))
    with pytest.raises(EtudeError):
        read_run(path, ref, len(raw), time.monotonic() + 30)
