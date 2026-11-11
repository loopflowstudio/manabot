"""Outcome evidence survives shipping and resume without matchup metric columns."""

from pathlib import Path
import shutil

from etude.run.block_file import block_key
from etude.run.delivery import StorageDestination, ship
from etude.run.storage import DirectoryStorage
from etude.train.block import Builder
from etude.train.manifest import Manifest, write_manifest
from etude.train.read import read
from etude.train.records import Step
from etude.train.run import track, write_block
import pytest

from manabot.training.outcome_tables import (
    OUTCOME_TABLE_METRIC,
    read_outcomes,
    write_outcomes,
)
from manabot.training.rollout.self_play import SelfPlayCounter, SelfPlayOutcome
from manabot.training.update_metrics import _updates, metric_row


def _outcomes(pairs: int = 2) -> list[SelfPlayOutcome]:
    counter = SelfPlayCounter()
    for i in range(pairs):
        counter.record((f"{i:064x}", f"{i + 1:064x}"), i % 2)
    counter.record(("mirror", "mirror"), None)
    return counter.drain()


def test_outcome_tables_ship_and_read_from_relocated_run(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    run = track(
        experiment="outcome-tables",
        directory=source,
        clocks=[Step("updates", "updates")],
        strict=True,
    )
    outcomes = _outcomes()
    try:
        digest = write_outcomes(run.files.root, outcomes)
        assert write_outcomes(run.files.root, outcomes) == digest
        run.log({OUTCOME_TABLE_METRIC: digest, "loss": 0.25}, updates=1)
    finally:
        run.finish()
    keys = run.files.deliverable()
    table = f"artifacts/self-play-outcomes/{digest}.json"
    assert table in keys
    assert keys.index(table) < min(
        keys.index(key) for key in keys if key.startswith("metrics/")
    )
    assert all(not key.endswith(".partial") for key in keys)
    assert (
        ship(run.files, run.name, StorageDestination(DirectoryStorage(destination)))
        is None
    )
    shutil.rmtree(source)

    data = read(destination, "outcome-tables", run.name)
    assert data.metrics() == ["loss"]
    projected = _updates(data)[1]
    assert projected == {
        "loss": 0.25,
        **metric_row({"self_play_outcomes": outcomes}),
    }


def test_outcome_table_missing_corrupt_and_invalid_references_fail(
    tmp_path: Path,
) -> None:
    storage = DirectoryStorage(tmp_path)
    digest = write_outcomes(tmp_path, _outcomes())
    path = tmp_path / f"artifacts/self-play-outcomes/{digest}.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="digest mismatch"):
        read_outcomes(storage, "", digest)
    with pytest.raises(ValueError, match="digest mismatch"):
        write_outcomes(tmp_path, _outcomes())
    path.unlink()
    with pytest.raises(FileNotFoundError):
        read_outcomes(storage, "", digest)
    with pytest.raises(ValueError, match="invalid.*digest"):
        read_outcomes(storage, "", "../outside")


def test_failed_outcome_write_leaves_no_deliverable_or_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_sync(fd: int) -> None:
        raise OSError("fixture sync failure")

    with monkeypatch.context() as patch:
        patch.setattr("manabot.training.outcome_tables.os.fsync", fail_sync)
        with pytest.raises(OSError, match="fixture sync failure"):
            write_outcomes(tmp_path, _outcomes())
    assert list((tmp_path / "artifacts/self-play-outcomes").iterdir()) == []
    digest = write_outcomes(tmp_path, _outcomes())
    assert read_outcomes(DirectoryStorage(tmp_path), "", digest) == _outcomes()


@pytest.mark.parametrize("missing_original_attempt", [False, True])
def test_update_selection_matches_sdk_with_clock_only_and_rewound_attempts(
    tmp_path: Path, missing_original_attempt: bool
) -> None:
    clocks = [Step("updates", "updates")]
    with track(experiment="selection", directory=tmp_path, clocks=clocks) as original:
        for update in (1, 2, 4):
            original.log({"loss": update}, updates=update)
    with track(
        experiment="selection", run=original.name, directory=tmp_path, clocks=clocks
    ) as resumed:
        resumed.log(updates=2)
        resumed.log({"loss": 30}, updates=3)
        resumed.log({"loss": 31}, updates=3)

    def selected() -> dict[int, int | float | str | bool]:
        data = read(tmp_path, "selection", original.name)
        if missing_original_attempt:
            # Retained blocks can outlive their attempt metadata. Their wall
            # clock still orders them before later recorded attempts.
            data = data.model_copy(update={"attempts": data.attempts[1:]})
        rows = {update: row["loss"] for update, row in _updates(data).items()}
        points = data.points("loss", "updates")
        assert rows == dict(zip(points.x, points.y, strict=True))
        return rows

    assert selected() == {1: 1, 2: 2, 3: 31}
    with track(
        experiment="selection",
        run=original.name,
        directory=tmp_path,
        clocks=clocks,
        resumed_at={"updates": 2},
    ):
        pass
    assert selected() == {1: 1, 2: 2}


def test_resume_selects_tables_before_loading_and_preserves_historical_rows(
    tmp_path: Path,
) -> None:
    clocks = [Step("updates", "updates")]
    original = track(experiment="outcome-resume", directory=tmp_path, clocks=clocks)
    historical = metric_row({"self_play_outcomes": _outcomes(1)})
    try:
        original.log(historical, updates=1)
        # Superseded references must never be fetched, even if their bytes
        # were not included in the downloaded evidence.
        original.log({OUTCOME_TABLE_METRIC: "0" * 64}, updates=2)
    finally:
        original.finish()
    resumed = track(
        experiment="outcome-resume",
        run=original.name,
        directory=tmp_path,
        clocks=clocks,
        resumed_at={"updates": 1},
    )
    resumed.finish()
    assert _updates(read(tmp_path, "outcome-resume", original.name)) == {1: historical}

    resumed = track(
        experiment="outcome-resume",
        run=original.name,
        directory=tmp_path,
        clocks=clocks,
        resumed_at={"updates": 1},
    )
    outcomes = _outcomes(3)
    try:
        digest = write_outcomes(resumed.files.root, outcomes)
        resumed.log({OUTCOME_TABLE_METRIC: "0" * 64}, updates=2)
        resumed.log({OUTCOME_TABLE_METRIC: digest}, updates=2)
        empty = write_outcomes(resumed.files.root, [])
        resumed.log({OUTCOME_TABLE_METRIC: empty}, updates=3)
    finally:
        resumed.finish()
    rows = _updates(read(tmp_path, "outcome-resume", original.name))
    assert rows == {
        1: historical,
        2: metric_row({"self_play_outcomes": outcomes}),
        3: {},
    }


def test_metric_manifest_does_not_index_outcome_table_contents(tmp_path: Path) -> None:
    """Same bounded update history, increasing matchup diversity, fixed index work.

    Exercise the real SDK block/index code. Byte counts, not host speed, guard
    the regression; table data grows on disk without entering the metric index.
    """
    writer = "abcdef012345"
    sizes: list[int] = []
    for pairs in (1, 16, 64):
        root = tmp_path / str(pairs)
        for update in range(1, 9):
            outcomes = _outcomes(pairs)
            outcomes[0]["wins"] += update
            digest = write_outcomes(root, outcomes)
            builder = Builder(("updates",))
            builder.add(update, {"updates": update}, {OUTCOME_TABLE_METRIC: digest})
            table = builder.table(writer, update)
            write_block(root, block_key(2, update, update, writer), table)
        write_manifest(root, writer)
        write_manifest(root, writer, permanent=False)
        path = root / f"metrics/{writer}.permanent.json"
        manifest = Manifest.model_validate_json(path.read_bytes())
        assert len(manifest.blocks) == 8
        assert all(
            [column.name for column in block.header.series] == [OUTCOME_TABLE_METRIC]
            for block in manifest.blocks
        )
        sizes.append(path.stat().st_size)
    # CRC decimal widths may differ; no deck names or counts enter the header.
    assert max(sizes) - min(sizes) < 32
