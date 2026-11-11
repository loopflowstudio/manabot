"""The MTG-281 projection retains metric coordinates and rejects damaged arrays."""

from pathlib import Path
import sys

from etude.run.block_file import block_key, parse
from etude.run.storage import DirectoryStorage
from etude.train.block import Builder, encode
import pytest

from scripts.measure_collection_growth import main, read_block


def test_collection_projection_preserves_sparse_values(tmp_path: Path) -> None:
    writer = "012345abcdef"
    builder = Builder(("updates",))
    builder.add(
        1000, {"updates": 1}, {"time/collection": 4.0, "self_play_outcomes/a": 9.0}
    )
    builder.add(2000, {"updates": 2}, {"coordinates/rss_bytes": 123.0})
    builder.add(2500, {}, {"time/collection": 99.0})
    builder.add(3000, {"updates": 3}, {"time/collection": 6.0})
    key = block_key(2, 0, 0, writer)
    storage = DirectoryStorage(tmp_path)
    storage.put("run/" + key, encode(builder.table(writer, 0)))
    file = parse(key)
    assert file is not None
    projection = read_block(storage, "run", file, "fixture")
    assert [(row.update, row.wall_ms, row.metrics) for row in projection.updates] == [
        (1, 1000, {"time/collection": 4.0}),
        (2, 2000, {"coordinates/rss_bytes": 123.0}),
        (3, 3000, {"time/collection": 6.0}),
    ]
    block = projection.block
    assert (block.first_update, block.last_update, block.level) == (1, 3, 2)
    assert block.header_bytes > 0


def test_collection_projection_rejects_array_corruption(tmp_path: Path) -> None:
    writer = "012345abcdef"
    builder = Builder(("updates",))
    builder.add(1000, {"updates": 1}, {"time/collection": 4.0})
    key = block_key(2, 0, 0, writer)
    body = bytearray(encode(builder.table(writer, 0)))
    length = int.from_bytes(body[12:16], "little")
    start = 16 + length + (-(16 + length) % 64)
    body[start] ^= 1
    storage = DirectoryStorage(tmp_path)
    storage.put("run/" + key, bytes(body))
    file = parse(key)
    assert file is not None
    with pytest.raises(ValueError, match="CRC"):
        read_block(storage, "run", file, "fixture")


@pytest.mark.parametrize("coordinate", [1.5, float("inf"), float("-inf")])
def test_collection_projection_rejects_invalid_update(
    tmp_path: Path, coordinate: float
) -> None:
    writer = "012345abcdef"
    builder = Builder(("updates",))
    builder.add(1000, {"updates": coordinate}, {"time/collection": 4.0})
    key = block_key(2, 0, 0, writer)
    storage = DirectoryStorage(tmp_path)
    storage.put("run/" + key, encode(builder.table(writer, 0)))
    file = parse(key)
    assert file is not None
    with pytest.raises(ValueError, match="invalid update coordinate"):
        read_block(storage, "run", file, "fixture")


def test_capture_refuses_existing_snapshot_before_reading_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("retained evidence\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_collection_growth.py",
            "--source",
            "invalid://never-opened",
            "--run",
            "run",
            "--cache",
            str(tmp_path / "cache"),
            "--out",
            str(snapshot),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert snapshot.read_text() == "retained evidence\n"
