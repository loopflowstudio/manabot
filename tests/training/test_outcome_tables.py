"""Self-play outcome tables ship, resume and project through Etude row tables."""

from pathlib import Path
import shutil

from etude.run import StorageDestination, ship
from etude.storage import DirectoryStorage
from etude.train.read import read
from etude.train.tracking import track
import pytest

from manabot.training.rollout.self_play import SelfPlayCounter, SelfPlayOutcome
from manabot.training.update_metrics import OUTCOME_TABLE, _updates, metric_row


def _outcomes(pairs: int = 2) -> list[SelfPlayOutcome]:
    counter = SelfPlayCounter()
    for i in range(pairs):
        counter.record((f"{i:064x}", f"{i + 1:064x}"), i % 2)
    counter.record(("mirror", "mirror"), None)
    return counter.drain()


def _projected(outcomes: list[SelfPlayOutcome]) -> dict[str, object]:
    return dict(metric_row({OUTCOME_TABLE: outcomes}))


def test_outcome_tables_ship_and_read_from_relocated_run(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    run = track(
        experiment="outcome-tables",
        directory=source,
        clocks=["updates"],
        strict=True,
    )
    outcomes = _outcomes()
    try:
        # Training logs the update's metrics, then its table at the same clocks.
        run.log({"loss": 0.25}, updates=1)
        run.log_table(OUTCOME_TABLE, outcomes, updates=1)
    finally:
        run.finish()
    assert (
        ship(run.files, run.name, StorageDestination(DirectoryStorage(destination)))
        is None
    )
    shutil.rmtree(source)

    data = read(destination, "outcome-tables", run.name)
    assert data.metrics() == ["loss"]
    assert _updates(data) == {1: {"loss": 0.25, **_projected(outcomes)}}


@pytest.mark.parametrize("missing_original_attempt", [False, True])
def test_update_selection_matches_sdk_with_clock_only_and_rewound_attempts(
    tmp_path: Path, missing_original_attempt: bool
) -> None:
    clocks = ["updates"]
    with track(experiment="selection", directory=tmp_path, clocks=clocks) as original:
        for update in (1, 2, 4):
            original.log({"loss": update}, updates=update)
    with track(
        experiment="selection",
        run=original.name,
        directory=tmp_path,
        clocks=clocks,
        resume=True,
    ) as resumed:
        resumed.log(updates=2)
        resumed.log({"loss": 30}, updates=3)
        resumed.log({"loss": 31}, updates=3)

    def selected() -> dict[int, int | float | str | bool]:
        data = read(tmp_path, "selection", original.name)
        if missing_original_attempt:
            # Retained blocks can outlive their attempt metadata. Their wall
            # clock still orders them before later recorded attempts.
            data = data.model_copy(update={"writers": data.writers[1:]})
        rows = {update: row["loss"] for update, row in _updates(data).items()}
        points = data.points("loss", "updates")
        assert rows == dict(zip(points.x, points.y, strict=True))
        return rows

    assert selected() == {1: 1, 3: 31}
    with track(
        experiment="selection",
        run=original.name,
        directory=tmp_path,
        clocks=clocks,
        resume=True,
    ) as rewound:
        rewound.log(updates=2)
    assert selected() == {1: 1}
    # Retained checkpoint-cut writers kept the boundary, even with no rows.
    data = read(tmp_path, "selection", original.name)
    legacy = data.model_copy(
        update={
            "writers": tuple(
                writer.model_copy(update={"first_row": False})
                for writer in data.writers
            )
        }
    )
    assert _updates(legacy) == {1: {"loss": 1}, 2: {"loss": 2}}


def test_resume_selects_tables_and_preserves_historical_rows(tmp_path: Path) -> None:
    clocks = ["updates"]
    historical = metric_row({OUTCOME_TABLE: _outcomes(1)})
    with track(experiment="outcome-resume", directory=tmp_path, clocks=clocks) as run:
        run.log(historical, updates=1)
        run.log({"loss": 1.0}, updates=2)
        run.log_table(OUTCOME_TABLE, _outcomes(2), updates=2)
    # The superseded update-2 table must never be fetched, even if its bytes
    # were not included in the downloaded evidence.
    for path in run.files.root.glob("artifacts/*.json"):
        path.unlink()
    with track(
        experiment="outcome-resume",
        run=run.name,
        directory=tmp_path,
        clocks=clocks,
        resume=True,
    ) as resumed:
        resumed.log({"loss": 1.5}, updates=2)
    assert _updates(read(tmp_path, "outcome-resume", run.name)) == {
        1: historical,
        2: {"loss": 1.5},
    }

    outcomes = _outcomes(3)
    with track(
        experiment="outcome-resume",
        run=run.name,
        directory=tmp_path,
        clocks=clocks,
        resume=True,
    ) as resumed:
        resumed.log_table(OUTCOME_TABLE, _outcomes(4), updates=2)
        resumed.log({"loss": 2.0}, updates=2)
        resumed.log_table(OUTCOME_TABLE, outcomes, updates=2)
        resumed.log_table(OUTCOME_TABLE, [], updates=3)
    assert _updates(read(tmp_path, "outcome-resume", run.name)) == {
        1: historical,
        2: {"loss": 2.0, **_projected(outcomes)},
        3: {},
    }


def test_retired_outcome_references_name_the_revision_that_reads_them(
    tmp_path: Path,
) -> None:
    with track(
        experiment="outcome-retired",
        directory=tmp_path,
        clocks=["updates"],
    ) as run:
        run.log({"self_play_outcomes/table": "0" * 64, "loss": 0.5}, updates=1)
    with pytest.raises(ValueError, match="bf4e280f7"):
        _updates(read(tmp_path, "outcome-retired", run.name))
