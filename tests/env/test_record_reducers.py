"""The same questions produce the same answers live and from saved games."""

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Never

import pytest

from manabot.env.observation import ObservationSpace, step_buffers
import managym
from managym.card_capture import CapturedCards
from managym.decision import Command, DecisionFrame
from managym.recording import (
    PlayerSetup,
    RecordedVectorEnv,
    ScalarCapture,
    read_record,
    read_records,
)
from managym.replay import (
    RecordedDecision,
    ReplayStop,
    main,
    replay_failure_inputs,
    walk,
)
from managym.statistics import Behaviour, CardPerformance, Reducer, Totals, reduce
from tests.fixtures.recorded_commands import record_until_object_candidate


class Choices(Reducer[int]):
    def start(self, setup: list[PlayerSetup]) -> None:
        self.count = 0

    def decision(self, decision: RecordedDecision) -> None:
        self.count += 1

    def end(self, result: ReplayStop, capture: CapturedCards) -> list[int]:
        return [self.count]


def mountain_duel() -> managym.GameSetup:
    return managym.GameSetup(
        [
            managym.PlayerConfig(name=str(seat), decklist={"Mountain": 12})
            for seat in (0, 1)
        ]
    )


@pytest.mark.parametrize("load_slots", [False, True])
def test_isolated_opening_failure_is_recorded_at_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_slots: bool,
) -> None:
    games_path = tmp_path / "games.jsonl"
    journal = tmp_path / "journal.jsonl"
    vector = RecordedVectorEnv(
        num_envs=2,
        seed=7,
        games_path=games_path,
        journal_path=journal,
        reducers=[Choices()],
    )
    vector.set_observation_layout(step_buffers(2), layout=ObservationSpace().layout)
    native = vector.native
    setup = mountain_duel()
    policies = ("heuristic:64", "agent")

    class FailedOpeningReceipt:
        # The native test injects a real construction error. Exercise receipt
        # consumer behavior without exposing failure injection in the binding.
        def __getattr__(self, name: str) -> Any:
            return getattr(native, name)

        def get_last_info(self) -> list[dict[str, object]]:
            return [
                {},
                {
                    "game_error": "stalled opening",
                    "game_error_class": "AgentError",
                    "game_error_trace": "opening stack",
                },
            ]

    try:
        if load_slots:
            vector.reset_each_into_buffers([setup, setup])
        monkeypatch.setattr(vector, "native", FailedOpeningReceipt())
        if load_slots:
            vector.load_slots_into_buffers(
                [managym.SlotLoad.fresh(1, setup, seed=30, player_policies=policies)],
                isolated=True,
            )
        else:
            vector.reset_each_into_buffers(
                [setup, setup],
                seeds=[31, 30],
                player_policies=[("agent", "agent"), policies],
                isolated=True,
            )
        assert vector._analysis_active == [True, False]
        snapshot = vector.snapshot()
        assert snapshot.needs_reproduction
        assert not snapshot.failed_opening(snapshot.games[0])
        assert snapshot.failed_opening(snapshot.games[1])
        failure = snapshot.games[1].failure
        assert failure is not None and failure.operation == "reset"
        assert failure.traceback == "opening stack"
        [written] = [
            record
            for record in read_records(games_path)
            if record.failed_opening(record.games[0])
        ]
        sibling, projected = read_records(journal)
        assert not sibling.failed_opening(sibling.games[0])
        assert not sibling.needs_reproduction
        detached = vector.take_slot_record(1)
        for record in (written, projected, detached):
            assert record.operation == "reset" and record.execution_outcome == "failed"
            assert record.error_class == "AgentError"
            assert record.traceback == "opening stack"
            assert record.player_policies == policies
            [failed] = record.games
            assert failed.seed == 30 and failed.actions == [] and failed.committed == 0
            assert failed.failure is None
            replay = walk(record, capture_cards=True)
            assert list(replay) == []
            assert replay.stop == ReplayStop("interrupted", 0, "stalled opening")
            assert replay.cards().final_cards == ()
    finally:
        vector.close()

    class ReproducedOpening:
        def __init__(
            self, *, player_policies: tuple[str, str], **kwargs: object
        ) -> None:
            assert player_policies == policies

        def reset_all(self, setup: managym.GameSetup) -> Never:
            raise managym.AgentError("stalled opening")

    def open_vector(
        *, seed: int, **kwargs: Any
    ) -> managym.VectorEnv | ReproducedOpening:
        if seed == 30:
            return ReproducedOpening(**kwargs)
        return native_type(seed=seed, **kwargs)

    native_type = managym.VectorEnv
    monkeypatch.setattr(managym, "VectorEnv", open_vector)
    [reproduced] = replay_failure_inputs(projected)
    assert reproduced["status"] == "failed" and reproduced["phase"] == "reset"
    assert reproduced["at_retained_failure"] and reproduced["matches_error"]

    monkeypatch.setattr(sys, "argv", ["replay", str(journal), "--worker"])
    main()
    healthy, failed = json.loads(capsys.readouterr().out)["results"]
    assert healthy["stream"] == 0 and healthy["status"] == "advanced"
    assert failed["stream"] == 1 and failed["phase"] == "reset"
    assert failed["at_retained_failure"] and failed["matches_error"]


@pytest.mark.parametrize("record_fraction", [0.0, 1.0])
def test_isolated_step_failure_retains_slot_evidence(
    tmp_path: Path, record_fraction: float
) -> None:
    games_path = tmp_path / "games.jsonl"
    journal = tmp_path / "journal.jsonl"
    vector = RecordedVectorEnv(
        num_envs=2,
        seed=7,
        games_path=games_path,
        journal_path=journal,
        record_fraction=record_fraction,
    )
    vector.set_observation_layout(step_buffers(2), layout=ObservationSpace().layout)
    setup = mountain_duel()
    try:
        vector.reset_each_into_buffers([setup, setup])
        vector.step_isolated_into_buffers([0, 0])
        vector.step_isolated_into_buffers([0, 999_999])
        info = vector.get_last_info()[1]
        assert info["game_error_trace"]
        vector.step_isolated_into_buffers([0, 0], active=[True, False])
        for evidence in (vector.snapshot(), read_record(journal)):
            assert evidence.needs_reproduction
            assert evidence.games[0].failure is None
            game = evidence.games[1]
            assert game.end is not None and game.end.status == "failed"
            assert game.failure is not None and game.failure.operation == "step"
            assert game.failure.error_class == info["game_error_class"]
            assert game.failure.traceback == info["game_error_trace"]
            assert game.frame == info["game_error_frame"]
            assert game.actions == [0, 999_999] and game.committed == 1
        sibling, projected = read_records(journal)
        assert not sibling.needs_reproduction
        detached = vector.take_slot_record(1)
        records = [projected, detached]
        if record_fraction:
            records.append(read_record(games_path))
        else:
            assert not games_path.exists()
        for record in records:
            assert record.operation == "step" and record.execution_outcome == "failed"
            assert record.needs_reproduction
            assert (
                record.error_class == "AgentError"
                and record.error == info["game_error"]
            )
            assert record.traceback == info["game_error_trace"]
            [game] = record.games
            assert game.failure is None
            assert game.frame == info["game_error_frame"]
            replay = walk(record)
            assert [decision.chosen for decision in replay] == [0]
            assert replay.stop == ReplayStop("interrupted", 1, record.error)
            [reproduced] = replay_failure_inputs(record)
            assert reproduced["at_retained_failure"] and reproduced["matches_error"]
            assert reproduced["matches_binding"]
            assert reproduced["original_outcome"] == "failed"
    finally:
        vector.close()

    completed = subprocess.run(
        [sys.executable, "-m", "managym.replay", str(journal), "--seconds", "10"],
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    healthy, failed = json.loads(completed.stdout.splitlines()[-1])["results"]
    assert healthy["stream"] == 0 and healthy["actions"] == 3
    assert not healthy["at_retained_failure"]
    assert failed["stream"] == 1 and failed["status"] == "failed"
    assert failed["action_index"] == 1 and failed["at_retained_failure"]
    assert failed["matches_binding"] and failed["matches_error"]


def test_boundaryless_snapshot_replays_isolated_failure_beside_siblings() -> None:
    vector = RecordedVectorEnv(num_envs=3, seed=7, record_in_memory=True)
    vector.set_observation_layout(step_buffers(3), layout=ObservationSpace().layout)
    setup = mountain_duel()
    try:
        vector.reset_each_into_buffers([setup, setup, setup])
        vector.step_isolated_into_buffers([0, 0, 0])
        vector.step_isolated_into_buffers([0, 999_999, 0])
        vector.take_slot_record(2)
        evidence = vector.snapshot()
        assert evidence.boundary is None
        healthy, failed, detached = replay_failure_inputs(evidence)
    finally:
        vector.close()
    assert healthy["status"] == "advanced" and not healthy["at_retained_failure"]
    assert failed["status"] == "failed" and failed["at_retained_failure"]
    assert failed["matches_error"] and failed["original_outcome"] == "failed"
    assert detached == {"stream": 2, "status": "setup_unavailable"}


def test_failed_scalar_reset_walks_an_empty_prefix(tmp_path: Path) -> None:
    path = tmp_path / "games.jsonl"
    capture = ScalarCapture(seed=7, games_path=path, failure_directory=tmp_path)
    setup = managym.GameSetup(
        [
            managym.PlayerConfig(name="bad", decklist={"Unknown card": 12}),
            managym.PlayerConfig(name="good", decklist={"Mountain": 12}),
        ]
    )
    with pytest.raises(managym.AgentError, match="Unknown card"):
        capture.reset(managym.Env(seed=7), setup)
    snapshot = capture.snapshot()
    capture.close()
    for record in (snapshot, read_record(path)):
        assert record.failed_opening(record.games[0])
        replay = walk(record)
        assert list(replay) == []
        assert replay.stop == ReplayStop("interrupted", 0, record.error)
        [reproduced] = replay_failure_inputs(record)
        assert reproduced["status"] == "failed" and reproduced["phase"] == "reset"
        assert reproduced["at_retained_failure"] and reproduced["matches_error"]


def test_live_reducers_equal_replayed_tables(tmp_path: Path) -> None:
    path = tmp_path / "game.jsonl"
    env = managym.Env(seed=7)
    capture = ScalarCapture(
        seed=7, games_path=path, reducers=[CardPerformance(), Behaviour(), Choices()]
    )
    capture.reset(env, mountain_duel())
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


def test_integer_replay_restores_verified_cards_after_divergent_step(
    tmp_path: Path,
) -> None:
    env = managym.Env(seed=7)
    env.enable_exposure_capture()
    path = tmp_path / "game.jsonl"
    capture = ScalarCapture(seed=7, games_path=path)
    capture.reset(env, mountain_duel())
    capture.step(env, 0)
    expected_digest = env.state_digest()
    expected_cards = json.loads(env.card_capture_json())
    capture.step(env, 0)
    capture.step(env, 0)
    capture.close()
    record = read_record(path)
    record.games[0].digests[2] = "0" * 64

    replay = walk(record, capture_cards=True)
    assert [decision.index for decision in replay] == [0]
    assert replay.stop is not None and replay.stop.status == "diverged"
    assert replay.stop.index == 1
    assert replay.env.state_digest() == expected_digest
    assert json.loads(replay.env.card_capture_json()) == expected_cards
    assert len(replay.cards().capture.exposure.choices) == 1


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
    frame, ordinal = record_until_object_candidate(env, capture)
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
    assert len(list(walk(journal))) == ordinal + 1
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
    assert len(list(mismatched)) == ordinal
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
    assert len(list(replay)) == ordinal
    assert replay.stop is not None and replay.stop.status == "diverged"
    assert "stale" in replay.stop.reason
    record.games[0].actions[-1] = json.dumps({"offer_id": 0})
    malformed = walk(record)
    assert len(list(malformed)) == ordinal
    assert malformed.stop is not None and malformed.stop.status == "diverged"


@pytest.mark.parametrize("stale", [False, True])
def test_rejected_answers_and_stale_command_do_not_reach_live_reducers(
    tmp_path: Path,
    stale: bool,
) -> None:

    env = managym.Env(seed=7)
    capture = ScalarCapture(seed=7, reducers=[Choices()], failure_directory=tmp_path)
    capture.reset(env, mountain_duel())
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


def test_totals_count_spells_draws_and_damage_live_and_replayed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bolts.jsonl"
    env = managym.Env(seed=5)
    capture = ScalarCapture(seed=5, games_path=path, reducers=[Totals()])
    players = [
        managym.PlayerConfig(
            name=str(seat), decklist={"Mountain": 10, "Lightning Bolt": 10}
        )
        for seat in (0, 1)
    ]
    capture.reset(env, players)
    casts = [0, 0]
    while not env.is_game_over():
        observation = env.observation_for_player(env.current_agent_index() or 0)
        actions = observation.action_space.actions
        # Cast whenever able. Every life point lost is Lightning Bolt damage,
        # so damage taken is the life lost.
        choice = next(
            (
                i
                for i, action in enumerate(actions)
                if action.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
            ),
            0,
        )
        casts[int(observation.agent.player_index)] += (
            actions[choice].action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
        )
        capture.step(env, choice)
    capture.close()
    (live,) = capture.rows()
    assert reduce(read_records(path), [Totals()]) == [live]
    final = env.observation_for_player(0)
    life = {
        int(final.agent.player_index): final.agent.life,
        int(final.opponent.player_index): final.opponent.life,
    }
    for row in live:
        assert row.games == 1 and row.turns == final.turn.turn_number
        assert row.damage_taken == 20 - life[row.seat]
        assert row.spells_cast == casts[row.seat] > 0
        assert row.cards_drawn > 0
    assert sum(row.damage_dealt or 0 for row in live) <= sum(
        row.damage_taken or 0 for row in live
    )
