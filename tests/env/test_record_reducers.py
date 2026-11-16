"""The same questions produce the same answers live and from saved games."""

import json
from pathlib import Path

import pytest

import managym
from managym.card_capture import CapturedCards
from managym.decision import Command, DecisionFrame
from managym.recording import PlayerSetup, ScalarCapture, read_record, read_records
from managym.replay import RecordedDecision, ReplayStop, walk
from managym.statistics import Behaviour, CardPerformance, Reducer, reduce


class Choices(Reducer[int]):
    def start(self, setup: list[PlayerSetup]) -> None:
        self.count = 0

    def decision(self, decision: RecordedDecision) -> None:
        self.count += 1

    def end(self, result: ReplayStop, capture: CapturedCards) -> list[int]:
        return [self.count]


def test_live_reducers_equal_replayed_tables(tmp_path: Path) -> None:
    path = tmp_path / "game.jsonl"
    env = managym.Env(seed=7)
    capture = ScalarCapture(
        seed=7, games_path=path, reducers=[CardPerformance(), Behaviour(), Choices()]
    )
    players = [
        managym.PlayerConfig(name=str(seat), decklist={"Mountain": 12})
        for seat in (0, 1)
    ]
    capture.reset(env, players)
    decisions = 0
    while not env.is_game_over():
        capture.step(env, 0)
        decisions += 1
        assert decisions < 1000
    capture.close()
    replayed = reduce(read_records(path), [CardPerformance(), Behaviour(), Choices()])
    assert replayed == capture.rows()
    replay = walk(read_record(path), capture_cards=True)
    with pytest.raises(ValueError, match="verified traversal"):
        replay.cards()
    list(replay)
    assert replay.cards().capture.exposure.choices
    uncaptured = walk(read_record(path))
    list(uncaptured)
    with pytest.raises(ValueError, match="capture_cards=True"):
        uncaptured.cards()
    cards, behaviour, custom = replayed
    assert cards and all(row.games == 1 for row in cards)
    assert any(row.land_opportunities for row in behaviour)
    assert custom == [decisions]


def test_original_command_and_preconditions_survive_live_and_replayed_reducers(
    tmp_path: Path,
) -> None:

    path = tmp_path / "commands.jsonl"
    env = managym.Env(seed=23)
    capture = ScalarCapture(
        seed=23,
        games_path=path,
        journal_path=tmp_path / "journal.jsonl",
        reducers=[Behaviour(), Choices()],
    )
    deck = {"Mountain": 10, "Gray Ogre": 20, "Lightning Bolt": 4}
    capture.reset(env, [managym.PlayerConfig(str(seat), deck) for seat in (0, 1)])
    for _ in range(15):
        actions = env.observation_for_player(
            env.current_agent_index() or 0
        ).action_space.actions
        chosen = next(
            (
                i
                for i, action in enumerate(actions)
                if action.action_type == managym.ActionEnum.PRIORITY_PLAY_LAND
            ),
            next(
                (
                    i
                    for i, action in enumerate(actions)
                    if action.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
                ),
                0,
            ),
        )
        capture.step(env, chosen)
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    assert frame.object_candidates
    command = Command(
        "exact-native-command",
        frame.revision,
        frame.offers[0]["id"],
        object_preconditions=(frame.object_candidates[0],),
    )
    capture.command(env, command.to_json(), annotation={"value": 0.25})
    digest = env.state_digest()
    capture.close()
    record = read_record(path)
    journal = read_record(tmp_path / "journal.jsonl")
    assert journal.games[0].command_frame_hashes == record.games[0].command_frame_hashes
    assert len(record.games[0].command_frame_hashes[-1]) == 64
    assert len(list(walk(journal))) == 16
    replay = walk(record)
    decisions = list(replay)
    assert replay.stop is not None and replay.stop.status == "interrupted"
    assert replay.env.state_digest() == digest
    assert decisions[-1].commitment.to_json() == command.to_json()
    assert decisions[-1].frame == frame
    assert decisions[-1].annotation.value == 0.25
    assert reduce([record], [Behaviour(), Choices()]) == capture.rows()

    edited = record.model_copy(deep=True)
    edited.games[0].command_frame_hashes[-1] = "0" * 64
    mismatched = walk(edited)
    assert len(list(mismatched)) == 15
    assert mismatched.stop is not None and mismatched.stop.status == "diverged"
    assert "legal semantic frame differs" in mismatched.stop.reason

    # A post-transition mismatch must leave native card facts at the verified
    # prefix too, rather than counting the rejected transition's choices/moves.
    altered = record.model_copy(deep=True)
    altered.games[0].digests[1] = "0" * 64
    prefix = walk(altered, capture_cards=True)
    assert list(prefix) == []
    assert prefix.stop.status == "diverged"
    assert not prefix.cards().capture.exposure.choices
    assert reduce([altered], [Choices()]) == [[0]]

    stale = json.loads(record.games[0].actions[-1])
    stale["expected_revision"] -= 1
    record.games[0].actions[-1] = json.dumps(stale)
    replay = walk(record)
    assert len(list(replay)) == 15
    assert replay.stop is not None and replay.stop.status == "diverged"
    assert "stale" in replay.stop.reason
    record.games[0].actions[-1] = json.dumps({"offer_id": 0})
    malformed = walk(record)
    assert len(list(malformed)) == 15
    assert malformed.stop is not None and malformed.stop.status == "diverged"


@pytest.mark.parametrize("stale", [False, True])
def test_rejected_answers_and_stale_command_do_not_reach_live_reducers(
    tmp_path: Path,
    stale: bool,
) -> None:

    env = managym.Env(seed=7)
    capture = ScalarCapture(seed=7, reducers=[Choices()], failure_directory=tmp_path)
    capture.reset(
        env, [managym.PlayerConfig(str(seat), {"Mountain": 12}) for seat in (0, 1)]
    )
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    assert all(not offer["choices"] for offer in frame.offers)
    command = Command(
        "rejected",
        frame.revision + int(stale),
        frame.offers[0]["id"],
        answers=()
        if stale
        else ({"kind": "candidates", "role": 1, "candidates": [0]},),
    )
    digest = env.state_digest()
    with pytest.raises(managym.AgentError, match="illegal|unexpected|stale"):
        capture.command(env, command.to_json())
    assert env.state_digest() == digest
    # A rejected input is retained as pending failure evidence, never a commitment.
    assert capture.inputs.committed == 0
    capture.close()
    assert capture.rows() == [[0]]
