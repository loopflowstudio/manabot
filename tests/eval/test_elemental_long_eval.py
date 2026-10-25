"""MTG-183's Etude evaluation: admission, the frozen design as matches, and following."""

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random

from etude.eval.match import Tables
from etude.eval.records import Match
from etude.launch.records import Status
from etude.run.records import ArtifactRef
from etude.train.records import CheckpointRecord
import pytest

from experiments.elemental_grid import DECKS, GameRow, schedule
from experiments.elemental_long import h2h_schedule
from experiments.elemental_long_eval import (
    EVIDENCE,
    Checkpoint,
    Conditions,
    EvalState,
    SegmentSource,
    admit,
    checkpoint_agent,
    done_keys,
    follow,
    game_rows,
    play,
    units,
    window_checkpoints,
)


def _artifact(path: Path, data: bytes) -> ArtifactRef:
    path.write_bytes(data)
    return ArtifactRef(
        uri=path.as_uri(),
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _record(path: Path, updates: int) -> CheckpointRecord:
    return CheckpointRecord(
        ordinal=1,
        artifact=_artifact(path, str(updates).encode()),
        at={"updates": updates, "learner_transitions": updates * 10},
        wall_ms=0,
    )


def _status(phase: str, outcome: str | None) -> Status:
    return Status.model_validate(
        {
            "run": "w3",
            "phase": phase,
            "reason": None,
            "outcome": outcome,
            "cause": None,
            "until": "updates=2700",
            "clocks": {},
            "usd": Decimal(1),
            "attempts": 1,
            "rental_ended": None,
        }
    )


def _ledger(tmp_path: Path, updates: tuple[int, ...]) -> Path:
    """The retired cohort follower's ledger shape, as it holds w1/w2."""
    checkpoints = {}
    for u in updates:
        path = tmp_path / f"w2-{u}.pt"
        path.write_bytes(f"w2-{u}".encode())
        checkpoints[str(u)] = {
            "updates": u,
            "job": "mtg183-elemental-long-w2",
            "training_run": "w2-run",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "coordinates": {"updates": u},
        }
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"checkpoints": checkpoints, "units": {}}))
    return ledger


def test_segment_joins_windows_and_settles_when_terminal(tmp_path: Path) -> None:
    windows = window_checkpoints(_ledger(tmp_path, (2500, 2550)))
    exports = [_record(tmp_path / f"{u}.pt", u) for u in (2550, 2600, 2650)]
    phase = ["running", None]
    segment = SegmentSource(
        run="w3",
        checkpoints=lambda: exports[:2] if phase[0] == "running" else exports,
        status=lambda: _status(phase[0], phase[1]),
    )
    state = EvalState()
    admit(state, windows, segment)
    # The segment's initial export repeats w2's endpoint; the window keeps it.
    assert state.checkpoints[2550].run == "mtg183-elemental-long-w2"
    assert state.checkpoints[2600].run == "w3"
    assert state.segment_outcome is None and state.endpoint is None
    # A segment that stops short of its target still ends the seed.
    phase[:] = ["done", "failed"]
    admit(state, windows, segment)
    assert state.segment_outcome == "failed" and state.endpoint == 2650
    assert state.error is None


def _state(tmp_path: Path, updates: tuple[int, ...], endpoint: int | None) -> EvalState:
    return EvalState(
        checkpoints={
            u: Checkpoint(
                updates=u,
                run="w1",
                artifact=_artifact(tmp_path / f"{u}.pt", f"weights {u}".encode()),
                coordinates={},
            )
            for u in updates
        },
        endpoint=endpoint,
    )


def test_units_play_the_frozen_schedules_one_match_per_deck(tmp_path: Path) -> None:
    design = units(_state(tmp_path, (0, 50, 100), 100))
    assert [u.key for u in design] == [
        "heuristic-000000-full",
        "heuristic-000050-subset",
        "pair-000050-v-000000",
        "heuristic-000100-full",
        "pair-000100-v-000050",
        "pair-000100-v-000000",
    ]
    full, subset, pair = design[0], design[1], design[2]
    for unit in design:
        assert [Conditions.model_validate(m.conditions).deck for m in unit.matches] == (
            list(DECKS)
        )
    deals = lambda unit: sorted(  # noqa: E731
        deal
        for m in unit.matches
        for g in Conditions.model_validate(m.conditions).groups
        for deal in g.deals
    )
    assert deals(full) == sorted(g.deal for g in schedule())
    assert deals(subset) == sorted(g.deal for g in schedule() if g.index < 10)
    assert deals(pair) == sorted(g.deal for g in h2h_schedule())
    # The heuristic is player 1 of every grid; the older checkpoint of every pair.
    assert {m.agents[1].name for m in full.matches} == {"heuristic"}
    assert {(m.agents[0].name, m.agents[1].name) for m in pair.matches} == {
        ("u000050", "u000000")
    }


def test_agent_identity_ignores_where_the_bytes_live(tmp_path: Path) -> None:
    state = _state(tmp_path, (0,), None)
    local = checkpoint_agent(state.checkpoints[0])
    moved = state.checkpoints[0].model_copy(
        update={
            "artifact": state.checkpoints[0].artifact.model_copy(
                update={"uri": "r2://manabot/jobs/w1/0.pt"}
            )
        }
    )
    assert checkpoint_agent(moved).key == local.key


def test_failed_group_is_stored_as_invalid_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that cannot load the checkpoint fails its group, not the run."""
    monkeypatch.setenv(EVIDENCE, str(tmp_path / "arena"))
    state = _state(tmp_path, (0,), None)
    unit = units(state)[0]
    first = Conditions.model_validate(unit.matches[0].conditions)
    one = first.model_copy(
        update={"groups": (first.groups[0].model_copy(update={"deals": (1,)}),)}
    )
    match = Match(agents=unit.matches[0].agents, conditions=one.model_dump(mode="json"))
    games = play(match)["games"]
    assert [(row["player"], row["valid"], row["score"]) for row in games] == [
        (0, False, None),
        (1, False, None),
    ]
    assert str(games[0]["failure"]).startswith("group failed")
    assert list((tmp_path / "arena" / match.key).glob("attempt-*/failed.txt"))


def test_game_rows_mirror_each_game_for_both_players() -> None:
    row = GameRow(
        model_deck="elemental_ur",
        heuristic_deck="elemental_bg",
        model_seat=1,
        deal=7,
        score=1.0,
        valid=True,
        terminated=True,
        truncated=False,
        failure=None,
        decisions=40,
        model_decisions=20,
        seconds=1.5,
    )
    first, second = game_rows([row], "full")
    assert (first["deck"], first["seat"], first["score"]) == ("elemental_ur", 1, 1.0)
    assert (second["deck"], second["seat"], second["score"]) == ("elemental_bg", 0, 0.0)
    assert first["unit"] == second["unit"] == "full"


def _fake_play(match: Match) -> Tables:
    rng = random.Random(match.seed)
    conditions = Conditions.model_validate(match.conditions)
    rows = [
        GameRow(
            model_deck=conditions.deck,
            heuristic_deck=group.heuristic_deck,
            model_seat=group.model_seat,
            deal=deal,
            score=float(rng.random() < 0.5),
            valid=True,
            terminated=True,
            truncated=False,
            failure=None,
            decisions=1,
            model_decisions=1,
            seconds=0.0,
        )
        for group in conditions.groups
        for deal in group.deals
    ]
    return {"games": game_rows(rows, conditions.kind)}


def test_follow_plays_every_unit_through_the_endpoint(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, (0, 50))
    segment = SegmentSource(
        run="w3",
        checkpoints=lambda: [_record(tmp_path / "100.pt", 100)],
        status=lambda: _status("done", "succeeded"),
    )
    played: list[str] = []

    def counted(match: Match) -> Tables:
        played.append(match.key)
        return _fake_play(match)

    state = follow(
        tmp_path / "eval",
        until=float("inf"),
        windows=lambda: window_checkpoints(ledger),
        segment=segment,
        storages=[],
        play_match=counted,
        workers=1,
        poll_seconds=0,
        yielding=lambda: False,
    )
    assert state.done and state.endpoint == 100 and state.segment_outcome == "succeeded"
    design = units(state)
    assert len(played) == len(set(played)) == 10 * len(design)
    assert {m.key for u in design for m in u.matches} <= done_keys(
        tmp_path / "eval/track"
    )
    assert EvalState.model_validate_json(
        (tmp_path / "eval/state.json").read_bytes()
    ).done
