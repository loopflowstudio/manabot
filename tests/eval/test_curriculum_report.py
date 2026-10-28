"""Small grids verify equal cells, paired seats and visible missing outcomes."""

import hashlib
from pathlib import Path
import shutil

from duckdb import DuckDBPyConnection
from etude.eval.match import Tables
from etude.eval.query import statsdb
from etude.eval.records import (
    Benchmark,
    CheckpointKey,
    CheckpointRef,
    Code,
    Environment,
    Inference,
    Match,
    MatchRecord,
    Matchup,
    Model,
    Version,
    digest,
)
from etude.eval.tournament import hold
from etude.run.records import ArtifactRef
from etude.run.storage import DirectoryStorage, Storage
import pytest

from experiments import curriculum_report
from experiments.curriculum_report import (
    ARMS,
    Arm,
    Bounds,
    compare,
    load_games,
    stratify,
)
from experiments.elemental_grid import Game, GameRow
from experiments.elemental_long_eval import game_rows


def _schedule() -> list[Game]:
    return [
        Game(deck, "opponent", 2 * deal + seat, deal, seat)
        for deck, deals in (("a", (10, 11)), ("b", (20,)))
        for deal in deals
        for seat in (0, 1)
    ]


def _results(schedule: list[Game]) -> dict[Arm, list[GameRow]]:
    return {
        arm: [
            GameRow(
                model_deck=g.model_deck,
                heuristic_deck=g.heuristic_deck,
                model_seat=g.model_seat,
                deal=g.deal,
                score=float(
                    (g.model_deck == "a" and (g.model_seat == 0 or arm in {"M", "ME"}))
                    or (g.model_deck == "b" and arm in {"E", "ME"})
                ),
                valid=True,
                terminated=True,
                truncated=False,
                failure=None,
                decisions=1,
                model_decisions=1,
                seconds=0.0,
            )
            for g in schedule
        ]
        for arm in ARMS
    }


def test_equal_cell_and_seat_weights_and_paired_interaction() -> None:
    schedule = _schedule()
    result = compare(schedule, _results(schedule), seed=1, replicates=8)
    assert result.complete
    assert result.scores == {"U": 0.25, "M": 0.5, "E": 0.75, "ME": 1.0}
    expected = {"M-U": 0.25, "E-U": 0.5, "ME-U": 0.75, "ME-M-E+U": 0.0}
    for name, value in expected.items():
        estimate = result.contrasts[name]
        # The seats disagree, but complete-deal averages are constant per cell.
        assert (estimate.value, estimate.low, estimate.high) == (value, value, value)


def test_missing_and_invalid_games_do_not_become_a_changed_population() -> None:
    schedule = _schedule()
    rows = _results(schedule)
    rows["U"][0] = rows["U"][0].model_copy(update={"valid": False, "score": None})
    rows["ME"].pop()
    result = compare(schedule, rows, seed=1, replicates=8)
    assert not result.complete
    assert result.scores == result.contrasts == {}
    assert result.coverage["U"].invalid == 1
    assert result.coverage["U"].missing == 0
    assert result.coverage["ME"].missing == 1
    assert result.coverage["ME"].valid == len(schedule) - 1
    assert result.score_bounds["U"] == Bounds(0.125, 0.25)
    assert result.score_bounds["ME"] == Bounds(0.75, 1.0)
    assert result.contrast_bounds["ME-U"] == Bounds(0.5, 0.875)


def test_duplicates_and_off_schedule_rows_are_rejected() -> None:
    schedule = _schedule()
    rows = _results(schedule)
    rows["M"].append(rows["M"][0])
    with pytest.raises(ValueError, match="duplicate"):
        compare(schedule, rows, seed=1, replicates=8)
    rows["M"] = rows["M"][:-1]
    rows["M"][0] = rows["M"][0].model_copy(update={"deal": 999})
    with pytest.raises(ValueError, match="off-schedule"):
        compare(schedule, rows, seed=1, replicates=8)


def test_incomplete_seat_schedule_and_invalid_valid_scores_are_rejected() -> None:
    schedule = _schedule()
    with pytest.raises(ValueError, match="both seats"):
        compare(schedule[:-1], _results(schedule[:-1]), seed=1, replicates=8)
    rows = _results(schedule)
    rows["E"][0] = rows["E"][0].model_copy(update={"score": float("nan")})
    with pytest.raises(ValueError, match="finite score"):
        compare(schedule, rows, seed=1, replicates=8)


def test_shared_deal_noise_cancels_from_paired_contrasts() -> None:
    schedule = _schedule()
    rows = _results(schedule)
    shared = [
        row.model_copy(update={"score": float(row.deal == 10)}) for row in rows["U"]
    ]
    result = compare(schedule, {arm: shared for arm in ARMS}, seed=9, replicates=16)
    assert result.scores == dict.fromkeys(ARMS, 0.25)
    for estimate in result.contrasts.values():
        assert (estimate.value, estimate.low, estimate.high) == (0.0, 0.0, 0.0)


def _fixture_play(match: Match) -> Tables:
    # Controlled outcomes, never inference or training on the fixture bytes.
    rows = _results(_schedule())
    arm = next(arm for arm in ARMS if arm == match.players[0].name)
    selected = rows[arm]
    if match.conditions.get("invalid"):
        selected = [
            row.model_copy(
                update={"valid": False, "score": None, "failure": "fixture invalid"}
            )
            for row in selected
        ]
    return {"games": game_rows(selected, "subset")}


def test_load_real_etude_tables_and_reject_changed_frozen_identity(
    tmp_path: Path,
) -> None:
    source = DirectoryStorage(tmp_path / "tracking")
    world = Environment(name="fixture-world", files=digest("world"))
    anchor = Code.model_validate(
        {"name": "heuristic", "version": {"source": digest("anchor")}}
    )
    policies: dict[Arm, Model] = {}
    for arm in ARMS:
        path = tmp_path / f"{arm}.pt"
        body = f"identity fixture {arm}; not a policy".encode()
        path.write_bytes(body)
        policies[arm] = Model(
            name=arm,
            inference=Inference.model_validate(
                {"version": {"source": digest("inference")}}
            ),
            checkpoint=CheckpointRef(
                key=CheckpointKey(run=f"train-{arm}", name="u0.pt"),
                artifact=ArtifactRef(
                    uri=path.as_uri(),
                    sha256=hashlib.sha256(body).hexdigest(),
                    size_bytes=len(body),
                ),
            ),
        )
    tournament = Benchmark(
        name="held-out",
        environment=world,
        matchups=tuple(
            Matchup(
                name=arm, players=(policies[arm], anchor), conditions=({},), games=6
            )
            for arm in ARMS
        ),
    )
    hold(
        tournament,
        experiment="curriculum_adapters",
        play=_fixture_play,
        directory=tmp_path / "writer",
        storages=[source],
    )
    rows = load_games(
        source, policies, anchor=anchor, environment=world, tournament="held-out"
    )
    assert compare(_schedule(), rows, seed=1, replicates=8).scores == {
        "U": 0.25,
        "M": 0.5,
        "E": 0.75,
        "ME": 1.0,
    }
    assert all(
        not games
        for games in load_games(
            source, policies, anchor=anchor, environment=world, tournament="development"
        ).values()
    )
    changed = dict(policies)
    changed["M"] = policies["M"].model_copy(
        update={
            "inference": policies["M"].inference.model_copy(
                update={"version": Version(source=digest("changed"))}
            )
        }
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        load_games(
            source, changed, anchor=anchor, environment=world, tournament="held-out"
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        load_games(
            source,
            policies,
            anchor=anchor,
            environment=world.model_copy(update={"files": digest("changed world")}),
            tournament="held-out",
        )

    changed_artifact = dict(policies)
    changed_artifact["U"] = policies["U"].model_copy(
        update={
            "checkpoint": policies["U"].checkpoint.model_copy(
                update={
                    "artifact": policies["U"].checkpoint.artifact.model_copy(
                        update={"sha256": digest("other bytes")}
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        load_games(
            source,
            changed_artifact,
            anchor=anchor,
            environment=world,
            tournament="held-out",
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        load_games(
            source,
            policies,
            anchor=anchor.model_copy(
                update={"version": Version(source=digest("other heuristic"))}
            ),
            environment=world,
            tournament="held-out",
        )
    invalid_source = DirectoryStorage(tmp_path / "invalid-tracking")
    assert all(
        not games
        for games in load_games(
            invalid_source,
            policies,
            anchor=anchor,
            environment=world,
            tournament="invalid-only",
        ).values()
    )
    invalid = Benchmark(
        name="invalid-only",
        environment=world,
        matchups=(
            Matchup(
                name="invalid",
                players=(policies["U"], anchor),
                conditions=({"invalid": True},),
                games=6,
            ),
        ),
    )
    hold(
        invalid,
        experiment="curriculum_adapters",
        play=_fixture_play,
        directory=tmp_path / "invalid-writer",
        storages=[invalid_source],
    )
    invalid_rows = load_games(
        invalid_source,
        policies,
        anchor=anchor,
        environment=world,
        tournament="invalid-only",
    )
    assert len(invalid_rows["U"]) == 6
    assert all(
        row.score is None and not row.valid and row.failure == "fixture invalid"
        for row in invalid_rows["U"]
    )
    assert (
        compare(_schedule(), invalid_rows, seed=1, replicates=8).coverage["U"].invalid
        == 6
    )


def test_curriculum_reader_preserves_retained_025_players_and_native_seats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).parents[1] / "fixtures/etude_sdk_025/elemental"
    store = tmp_path / "store"
    shutil.copytree(fixture, store)
    retained = {path: path.read_bytes() for path in store.rglob("*") if path.is_file()}
    root = store / "mtg183-eval"
    match = next(
        record.match
        for path in (root / "matches").glob("*.json")
        if (record := MatchRecord.model_validate_json(path.read_bytes()))
        and record.match.players[0].name == "u000000"
        and isinstance(record.match.players[1], Code)
    )
    model, anchor = match.players
    assert isinstance(model, Model) and isinstance(anchor, Code)
    world = Environment.model_validate_json((root / "environment.json").read_bytes())

    def legacy_database(experiment: str, source: Storage) -> DuckDBPyConnection:
        assert experiment == "curriculum_adapters"
        db = statsdb("mtg183-elemental-eval", source)
        # This real pre-campaign fixture has no tournament. Give only its
        # disposable SQL projection the selection label; retained bytes stay put.
        db.execute("update runs set tournament = 'held-out'")
        return db

    monkeypatch.setattr(curriculum_report, "statsdb", legacy_database)
    rows = load_games(
        DirectoryStorage(store),
        {"U": model},
        anchor=anchor,
        environment=world,
        tournament="held-out",
    )
    assert len(rows["U"]) == match.games
    assert {row.model_seat for row in rows["U"]} == {0, 1}
    assert all(row.score == 1.0 and row.valid for row in rows["U"])
    assert all(not rows[arm] for arm in ("M", "E", "ME"))
    assert all(path.read_bytes() == content for path, content in retained.items())


def test_deck_and_seat_strata_keep_equal_cell_weights_and_missing_games() -> None:
    schedule = _schedule()
    rows = _results(schedule)
    rows["U"][0] = rows["U"][0].model_copy(update={"valid": False, "score": None})
    rows["ME"].pop()
    strata = {(row.by, row.value): row.summary for row in stratify(schedule, rows)}
    deck_a = strata[("model_deck", "a")]
    assert deck_a.score_bounds["M"] == Bounds(1, 1)
    assert deck_a.score_bounds["U"] == Bounds(0.25, 0.5)
    assert deck_a.coverage["U"].invalid == 1
    assert deck_a.coverage["U"].missing == 0
    deck_b = strata[("model_deck", "b")]
    assert deck_b.score_bounds["ME"] == Bounds(0.5, 1)
    assert deck_b.coverage["ME"].valid == deck_b.coverage["ME"].missing == 1
    seat_zero = strata[("model_seat", 0)]
    # Cell a has twice as many deals as b, but each keeps half the weight.
    assert seat_zero.score_bounds["M"] == Bounds(0.5, 0.5)
    assert seat_zero.score_bounds["U"] == Bounds(0.25, 0.5)
    assert seat_zero.score_bounds["E"] == Bounds(1, 1)
    assert strata[("model_seat", 1)].score_bounds["E"] == Bounds(0.5, 0.5)
    assert strata[("heuristic_deck", "opponent")].score_bounds["E"] == Bounds(
        0.75, 0.75
    )
    assert len(strata) == 5
