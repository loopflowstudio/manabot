"""A private scalar position can outlive its producer and reopen as a live game."""

import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Literal
from unittest.mock import Mock

from blake3 import blake3
import pytest

from manabot.env.observation import ObservationSpace, column, encode_observation
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.sim import sim
from manabot.verify.text_observation import reset_to_first_main
import managym
from managym.decision import Command, DecisionFrame
from managym.recording import (
    ScalarCapture,
    read_record,
    record_from_row,
    records_from_source,
)
from managym.replay import replay_failure_inputs, walk
from tests.fixtures.recorded_commands import (
    preferred_action,
    record_until_object_candidate,
)


def setup(decklists: Literal["open", "closed"] = "closed") -> managym.GameSetup:
    return managym.GameSetup(
        [
            managym.PlayerConfig(name, {"Mountain": 8, "Gray Ogre": 4})
            for name in ("a", "b")
        ],
        decklists=decklists,
    )


def cold_restore(env: managym.Env) -> managym.Env:
    replay = env.snapshot().to_replay_json()
    return managym.GameSnapshot.from_replay_json(replay).restore()


def assert_replayed_command(
    env: managym.Env, restored: managym.Env, command: str
) -> None:
    transition = env.execute_semantic_command_json(command)
    assert restored.execute_semantic_command_json(command) == transition


def decision_frame(env: managym.Env) -> DecisionFrame:
    return DecisionFrame.from_json(env.semantic_decision_frame_json())


def replay_priority_until_stack_empty(env: managym.Env, restored: managym.Env) -> None:
    for _ in range(2):
        if not env.observation_for_player(0).stack_objects:
            return
        frame = decision_frame(env)
        passed = next(o for o in frame.offers if o["verb"] == "pass_priority")
        command = Command(
            f"pass-{frame.revision}", frame.revision, passed["id"]
        ).to_json()
        assert_replayed_command(env, restored, command)
    assert not env.observation_for_player(0).stack_objects


def test_reseeded_replay_preserves_the_reset_seed_and_both_reseeds() -> None:
    source = managym.Env(seed=42, decision_memory_include_own=True)
    source.reset(setup())
    source.step(0)
    source.set_seed(99)
    position = source.snapshot()
    del source
    branch = position.restore(managym.RestoreOptions.exact(reseed=71))
    assert branch.hand_snapshot(0).seed == 42
    branch.step(0)
    branch = branch.snapshot().restore(managym.RestoreOptions.exact(reseed=72))
    raw = branch.snapshot().to_replay_json()
    saved = json.loads(raw)
    assert saved["seed"] == 42
    assert saved["actions"] == [0, 0]
    assert [i["at_action"] for i in saved["interventions"]] == [1, 2]
    assert saved["player_configs"][1]["disclosure"] == "closed"

    reopened = managym.GameSnapshot.from_replay_json(raw).restore()
    reopened.step(0)
    branch.step(0)
    assert reopened.search_witness_json() == branch.search_witness_json()
    assert logs(reopened) == logs(branch)
    assert position.restore().state_digest() != branch.state_digest()


def test_snapshot_replays_commands_with_issued_object_bindings() -> None:
    source = managym.Env(seed=23)
    capture = ScalarCapture(seed=23)
    frame, ordinal = record_until_object_candidate(source, capture)
    command = Command(
        "snapshot",
        frame.revision,
        frame.offers[0]["id"],
        object_preconditions=(frame.object_candidates[0],),
    ).to_json()
    position = source.snapshot()
    restored = position.restore()
    issued = managym.GameSnapshot.from_replay_json(position.to_replay_json()).restore()
    issued.execute_semantic_command_json(command)
    # An identical position reached by index alone never issued the address.
    unissued = managym.Env(seed=23)
    unissued.reset(capture.inputs.setup())
    for action in capture.inputs.actions:
        assert isinstance(action, int)
        unissued.step(action)
    assert unissued.state_digest() == source.state_digest()
    unissued_cold = cold_restore(unissued)
    with pytest.raises(managym.AgentError, match="unknown_object_candidate"):
        unissued_cold.execute_semantic_command_json(command)
    # Execute the already-issued address without asking the branch for a frame.
    source.execute_semantic_command_json(command)
    restored.execute_semantic_command_json(command)
    assert restored.state_digest() == source.state_digest()
    raw = restored.snapshot().to_replay_json()
    assert json.loads(json.loads(raw)["actions"][ordinal]) == json.loads(command)
    cold = managym.GameSnapshot.from_replay_json(raw).restore()
    assert cold.search_witness_json() == restored.search_witness_json()
    capture.close()


@pytest.mark.parametrize("policies", [("random", "agent"), ("heuristic", "random")])
def test_snapshot_records_only_external_calls(
    policies: tuple[
        Literal["agent", "random", "heuristic"], Literal["agent", "random", "heuristic"]
    ],
) -> None:
    source = managym.Env(seed=42, player_policies=policies)
    source.reset(setup())
    for _ in range(3):
        source.step(0 if "agent" in policies else None)
    raw = source.snapshot().to_replay_json()
    assert json.loads(raw)["actions"] == ([0] if "agent" in policies else [-1]) * 3
    cold = managym.GameSnapshot.from_replay_json(raw).restore()
    source.step(0 if "agent" in policies else None)
    cold.step(0 if "agent" in policies else None)
    assert cold.search_witness_json() == source.search_witness_json()


def test_snapshot_metadata_and_terminal_boundary() -> None:
    env = managym.Env(seed=42)
    with pytest.raises(managym.AgentError, match="before reset"):
        env.snapshot()
    seat = reset_to_first_main(env, setup())
    obs = env.observation_for_player(seat)
    opening = env.snapshot()
    meta = opening.metadata()
    assert meta.turn == 1
    assert meta.phase == managym.PhaseEnum.PRECOMBAT_MAIN
    assert meta.life == [20, 20]
    assert meta.board_counts == [0, 0]
    assert not meta.active_card_defs and meta.stack_len == 0
    assert meta.action_count == len(obs.action_space.actions)
    assert opening.acting_player == meta.acting_player == obs.agent.player_index
    assert meta.action_kind == managym.ActionSpaceEnum.PRIORITY
    land = next(
        i
        for i, a in enumerate(obs.action_space.actions)
        if a.action_type == managym.ActionEnum.PRIORITY_PLAY_LAND
    )
    env.step(land)
    board = env.snapshot().metadata()
    assert board.board_counts == [1, 0] and len(board.active_card_defs) == 1
    for _ in range(64):
        last = env.snapshot()
        env.step(0)
        if env.is_game_over():
            break
    else:
        pytest.fail("fixture did not reach its terminal action")
    terminal = env.snapshot()
    meta = terminal.metadata()
    assert meta.terminal and meta.action_count == 0
    assert meta.action_kind is None and meta.acting_player is None
    assert terminal.acting_player is None
    ended = last.restore(managym.RestoreOptions.exact(forced_first_action=0))
    assert ended.is_game_over() and ended.state_digest() == env.state_digest()
    with pytest.raises(managym.AgentError, match="non-terminal"):
        terminal.restore(managym.RestoreOptions.exact(forced_first_action=0))
    assert terminal.restore().is_game_over()
    assert (
        managym.GameSnapshot.from_replay_json(terminal.to_replay_json())
        .metadata()
        .terminal
    )


def test_snapshot_rejects_corrupt_replay_and_retains_unjournaled_hot_state() -> None:
    source = managym.Env(seed=42)
    source.reset(setup())
    snapshot = source.snapshot()
    raw = snapshot.to_replay_json()
    saved = json.loads(raw)
    for patch, reason in [
        ({"schema": "unknown"}, "invalid snapshot"),
        (
            {
                "player_configs": [
                    seat | {"unknown": True} for seat in saved["player_configs"]
                ]
            },
            "unknown field",
        ),
        ({"card_digest": "bad"}, "card_digest"),
        ({"state_digest": "bad"}, "replay_digest"),
        ({"player_policies": ["random", "random"]}, "replay_digest"),
        ({"decision_memory": True}, "replay_digest"),
        ({"actions": [999]}, "replay_digest"),
        (
            {"interventions": [{"op": "reseed", "seed": 1, "at_action": 2}]},
            "replay_digest",
        ),
    ]:
        with pytest.raises(managym.AgentError, match=reason):
            managym.GameSnapshot.from_replay_json(json.dumps(saved | patch))
    source.random_action_index()
    hot = source.snapshot()
    assert hot.restore().state_digest() == source.state_digest()
    with pytest.raises(managym.AgentError, match="unjournaled"):
        hot.to_replay_json()
    assert snapshot.to_replay_json() == raw


def test_native_exposure_replays_external_calls_with_recorded_choosers() -> None:
    vector = managym.VectorEnv(
        1,
        seed=42,
        player_policies=("random", "agent"),
        decision_memory_include_own=True,
    )
    vector.enable_exposure_capture()
    vector.reset_all(setup())
    vector.step([0])
    record = record_from_row(
        {
            "schema": "training-exposure-v1",
            "skip_trivial": True,
            "stream": 0,
            "identities": {},
            "context": {},
            "seconds": 0.0,
            "game": json.loads(vector.current_exposure()[0]),
        }
    )
    assert record.games[0].actions == [0]
    assert record.resolved_player_policies == ("random", "agent")
    replay = walk(record)
    assert len(list(replay)) == 1
    assert replay.stop is not None and replay.stop.status == "interrupted"
    assert replay.env.state_digest() == vector.state_digests()[0]
    assert replay.env.decision_memory_counts() == vector.decision_memory_counts()[0]


def logs(env: managym.Env) -> list[str]:
    return [env.observation_for_player(i).decision_memory_json() for i in (0, 1)]


@pytest.mark.parametrize("decklists", ["open", "closed"])
def test_shuffled_siblings_share_a_deal_across_opening_choices(
    decklists: Literal["open", "closed"],
) -> None:
    source = managym.Env(seed=42, decision_memory_include_own=True)
    source.reset(setup(decklists))
    position = source.snapshot()
    options = managym.RestoreOptions.shuffle_libraries
    shuffled = position.restore(options(101))
    assert shuffled.state_digest() != source.state_digest()
    assert shuffled.state_digest() != position.restore(options(102)).state_digest()
    for seat in (0, 1):
        assert shuffled.hand_snapshot(seat).cards == source.hand_snapshot(seat).cards
        assert (
            shuffled.observation_for_player(seat).toJSON()
            == source.observation_for_player(seat).toJSON()
        )
    for choice in (0, 1):
        branch = position.restore(options(101, forced_first_action=choice))
        control = shuffled.snapshot().restore()
        control.step(choice)
        assert control.search_witness_json() == branch.search_witness_json()
        assert logs(control) == logs(branch)
        raw = branch.snapshot().to_replay_json()
        assert json.loads(raw)["actions"] == [choice]
        cold = managym.GameSnapshot.from_replay_json(raw).restore()
        for env in (cold, branch):
            env.step(0)
        assert cold.search_witness_json() == branch.search_witness_json()
        assert logs(cold) == logs(branch)


def test_promoted_fork_replays_frozen_then_live_logs_with_determinization() -> None:
    source = managym.Env(seed=42, decision_memory_include_own=True)
    source.reset(setup("open"))
    source.step(0)
    prefix = logs(source)
    fork = source.clone_env()
    fork.step(0)
    assert logs(fork) == prefix
    viewer = fork.snapshot().metadata().acting_player
    assert viewer is not None
    fork.determinize(seed=5, perspective=viewer)
    branch = fork.snapshot().restore(
        managym.RestoreOptions.shuffle_libraries(
            13, enable_decision_memory=True, forced_first_action=0
        )
    )
    assert logs(source) == prefix and logs(branch) != prefix
    raw = branch.snapshot().to_replay_json()
    assert json.loads(raw)["interventions"] == [
        {"at_action": 1, "op": "set_decision_memory", "enabled": False},
        {"at_action": 2, "op": "determinize", "viewer": viewer, "seed": 5},
        {"at_action": 2, "op": "shuffle_libraries", "seed": 13},
        {"at_action": 2, "op": "set_decision_memory", "enabled": True},
    ]
    cold = managym.GameSnapshot.from_replay_json(raw).restore()
    assert cold.search_witness_json() == branch.search_witness_json()
    assert logs(cold) == logs(branch)


def test_forced_builtin_override_replays_at_the_trailing_boundary() -> None:
    source = managym.Env(seed=42, player_policies=("random", "heuristic"))
    source.reset(setup())
    options = managym.RestoreOptions.exact(
        enable_decision_memory=True, forced_first_action=1
    )
    branch = source.snapshot().restore(options)
    control = managym.Env(seed=42, decision_memory=True)
    control.reset(setup())
    control.step(1)
    assert branch.state_digest() == control.state_digest()
    assert logs(branch) == logs(control)
    raw = branch.snapshot().to_replay_json()
    saved = json.loads(raw)
    assert saved["actions"] == []
    assert saved["interventions"] == [
        {"op": "set_decision_memory", "at_action": 0, "enabled": True},
        {"op": "force_action", "at_action": 0, "action": 1},
    ]
    cold = managym.GameSnapshot.from_replay_json(raw).restore()
    assert cold.search_witness_json() == branch.search_witness_json()
    replay = walk(record_from_row(saved))
    assert list(replay) == []
    assert replay.stop is not None and replay.stop.status == "interrupted"
    assert replay.env.search_witness_json() == branch.search_witness_json()
    assert logs(replay.env) == logs(branch)
    assert (
        replay.observation.toJSON()
        == branch.observation_for_player(branch.current_agent_index() or 0).toJSON()
    )
    with pytest.raises(managym.AgentError, match="built-in chooser"):
        cold.step(1)
    for env in (branch, cold):
        env.step(None)
    assert cold.search_witness_json() == branch.search_witness_json()
    assert logs(cold) == logs(branch)


def test_restore_rejects_illegal_or_rebound_actions_without_changing_the_source() -> (
    None
):
    source = managym.Env(seed=42)
    source.reset(setup())
    position = source.snapshot()
    with pytest.raises(managym.AgentError, match="decklist closed"):
        position.restore(managym.RestoreOptions.determinize(0, 42))
    for viewer in (-1, 2):
        with pytest.raises(ValueError, match="viewer must be 0 or 1"):
            managym.RestoreOptions.determinize(viewer, 42)
    with pytest.raises(managym.AgentError, match="out of bounds"):
        position.restore(
            managym.RestoreOptions.exact(
                forced_first_action=position.metadata().action_count
            )
        )
    assert position.restore().search_witness_json() == source.search_witness_json()
    seat = reset_to_first_main(source, setup("open"))
    obs = source.observation_for_player(seat)
    land = next(
        i
        for i, a in enumerate(obs.action_space.actions)
        if a.action_type == managym.ActionEnum.PRIORITY_PLAY_LAND
    )
    position = source.snapshot()
    with pytest.raises(managym.AgentError, match="changed after resampling"):
        position.restore(
            managym.RestoreOptions.determinize(1, 35, forced_first_action=land)
        )
    assert position.restore().search_witness_json() == source.search_witness_json()


def test_snapshot_walk_reissues_historical_command_precondition() -> None:
    env = managym.Env(seed=23)
    capture = ScalarCapture(seed=23)
    previous, _ = record_until_object_candidate(env, capture)
    capture.close()
    env.step(0)
    frame = decision_frame(env)
    env.execute_semantic_command_json(
        Command(
            "later",
            frame.revision,
            frame.offers[0]["id"],
            object_preconditions=(previous.object_candidates[0],),
        ).to_json()
    )
    saved = json.loads(env.snapshot().to_replay_json())
    record = record_from_row(saved)
    assert record.games[0].object_candidate_frames == saved["object_candidate_frames"]
    replay = walk(record)
    assert len(list(replay)) == len(record.games[0].actions)
    assert replay.stop is not None and replay.stop.status == "interrupted"
    assert replay.env.search_witness_json() == env.search_witness_json()
    (result,) = replay_failure_inputs(record)
    assert result["status"] == "advanced"
    assert result["actions"] == len(record.games[0].actions)


def position_evidence(env: managym.Env) -> dict[str, str | list[str]]:
    return {
        "digest": env.state_digest(),
        "memory": logs(env),
        "frame": env.semantic_decision_frame_json(),
    }


def test_banked_branch_round_trips_through_replay_and_academy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from managym.academy import trace
    from managym.academy.watch import import_game, inspect_watch

    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "academy")
    shuffle_seed = 77665544332211
    source = managym.Env(seed=42, decision_memory_include_own=True)
    reset_to_first_main(source, setup())
    source.step(0)
    bank = source.snapshot()
    assert bank.metadata().action_count >= 2
    assert all(json.loads(log) for log in logs(source))
    shared = bank.restore(managym.RestoreOptions.shuffle_libraries(shuffle_seed))
    branches = [
        bank.restore(
            managym.RestoreOptions.shuffle_libraries(
                shuffle_seed, forced_first_action=i
            )
        )
        for i in (0, 1)
    ]
    for index, branch in enumerate(branches):
        control = shared.snapshot().restore()
        control.step(index)
        assert branch.search_witness_json() == control.search_witness_json()
        assert logs(branch) == logs(control)

    # After the next draw, the unshuffled game offers a different legal frame.
    branch = branches[0]
    branch.step(0)
    source.step(0)
    source.step(0)
    frame = branch.semantic_decision_frame_json()
    assert source.semantic_decision_frame_json() != frame
    branch = branch.snapshot().restore(
        managym.RestoreOptions.shuffle_libraries(987654321123)
    )
    raw = branch.snapshot().to_replay_json()
    path = tmp_path / "branch.json"
    path.write_text(raw)
    # Both seats kept their opening hands before the three play decisions.
    expected_actions = [0, 0, 0, 0, 0]
    committed = len(expected_actions)
    assert json.loads(raw)["interventions"][-1]["at_action"] == committed
    before = position_evidence(branch)
    assert before["frame"] == frame
    with pytest.raises(managym.AgentError) as rejected:
        branch.step(999)
    branch.step(0)
    after = position_evidence(branch)
    del bank, source, shared, control, branch, branches

    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, sys
from pathlib import Path
import managym
from tests.env.test_game_snapshot import position_evidence
env = managym.GameSnapshot.from_replay_json(Path(sys.argv[1]).read_text()).restore()
before = position_evidence(env)
env.step(0)
print(json.dumps([before, position_evidence(env)]))
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(child.stdout) == [before, after]

    records = records_from_source(path)
    assert records is not None
    (record,) = records
    assert record.context == {}
    replay = walk(record)
    assert [decision.chosen for decision in replay] == expected_actions
    assert replay.stop is not None and replay.stop.status == "interrupted"
    assert position_evidence(replay.env) == before

    imported = import_game(json.loads(raw))
    assert (imported["stop"], imported["count"]) == ("interrupted", committed)
    for ordinal in range(committed):
        shown = inspect_watch(imported["id"], ordinal, imported["token"])
        assert shown["observation"]["opponent"]["hand"] == []
        revealed = inspect_watch(
            imported["id"], ordinal, imported["token"], reveal=True
        )
        assert revealed["observation"]["opponent"]["hand"]
        assert revealed["decision"] == shown["decision"]
        for view in (shown, revealed):
            assert view["source_kind"] is None
            assert str(shuffle_seed) not in json.dumps(view)
            assert "987654321123" not in json.dumps(view)

    record.games[0].committed = committed
    record.games[0].actions.append(999)
    record.games[0].frame = frame
    record.execution_outcome = "failed"
    record.error_class = type(rejected.value).__name__
    record.error = str(rejected.value)
    (result,) = replay_failure_inputs(record)
    assert result["action_index"] == committed
    assert result["status"] == "failed"
    assert (
        result["at_retained_failure"]
        and result["matches_binding"]
        and result["matches_error"]
    )
    record.games[0].interventions.clear()
    assert replay_failure_inputs(record)[0]["status"] == "binding_mismatch"


def checksummed_replay(env: managym.Env, **fields: object) -> str:
    """Simulate a writer with different rules while retaining a valid replay checksum."""
    saved = json.loads(env.snapshot().to_replay_json()) | fields
    del saved["replay_digest"]
    canonical = json.dumps(
        saved, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return json.dumps(saved | {"replay_digest": blake3(canonical).hexdigest()})


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"state_digest": "old rules digest"}, "state_digest"),
        ({"actions": [0, 999, 0]}, "action 1"),
    ],
)
def test_valid_replay_from_changed_rules_keeps_verified_decisions(
    tmp_path: Path, fields: dict[str, object], reason: str
) -> None:
    env = managym.Env(seed=42)
    env.reset(setup())
    env.step(0)
    env.step(0)
    raw = checksummed_replay(env, **fields)
    with pytest.raises(managym.AgentError, match=reason):
        managym.GameSnapshot.from_replay_json(raw)
    path = tmp_path / "branch.json"
    path.write_text(raw)
    record = read_record(path)
    replay = walk(record)
    assert [decision.index for decision in replay] == [0]
    assert replay.stop is not None
    assert (replay.stop.status, replay.stop.index) == ("diverged", 1)


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"decision_memory_include_own": True}, "invalid initial decision memory"),
        ({"interventions": [{"at_action": 2, "op": "reseed", "seed": 1}]}, "offset"),
        (
            {
                "interventions": [
                    {"at_action": 1, "op": "reseed", "seed": 1},
                    {"at_action": 0, "op": "reseed", "seed": 2},
                ]
            },
            "offset",
        ),
    ],
)
def test_record_reader_checks_signed_replay_structure(
    fields: dict[str, object], reason: str
) -> None:
    env = managym.Env(seed=42)
    env.reset(setup())
    env.step(0)
    with pytest.raises(ValueError, match=reason):
        record_from_row(json.loads(checksummed_replay(env, **fields)))


def test_complete_walk_requires_every_issued_frame() -> None:
    env = managym.Env(seed=42)
    env.reset(setup())
    raw = checksummed_replay(env, object_candidate_frames=["missing frame"])
    replay = walk(record_from_row(json.loads(raw)))
    assert list(replay) == []
    assert replay.stop is not None and replay.stop.status == "diverged"
    assert replay.stop.reason and "candidate frame missing" in replay.stop.reason


@pytest.mark.parametrize("field", ["card_digest", "state_digest"])
def test_record_reader_checks_snapshot_replay_before_projection(field: str) -> None:
    env = managym.Env(seed=42)
    env.reset(setup())
    saved = json.loads(env.snapshot().to_replay_json())
    saved[field] = "edited"
    with pytest.raises(ValueError, match="digest mismatch"):
        records_from_source(saved)


def test_watch_publishes_the_determinized_offers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from managym.academy import trace
    from managym.academy.recording import Recording
    from managym.academy.watch import import_game, inspect_watch

    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    env = managym.Env(seed=42)
    seat = reset_to_first_main(env, setup("open"))
    original = env.observation_for_player(seat)
    branch = env.snapshot().restore(managym.RestoreOptions.determinize(1, 35))
    offers = branch.observation_for_player(0).action_space.actions
    assert len(offers) != len(original.action_space.actions)
    branch.step(len(offers) - 1)
    imported = import_game(json.loads(branch.snapshot().to_replay_json()))
    shown = inspect_watch(imported["id"], imported["count"] - 1, imported["token"])
    assert len(shown["decision"]["frame"]["offers"]) == len(offers)
    assert shown["decision"]["offer"]["label"] == "Pass priority"
    # Review marks the re-dealt decision, and only it, without the seed.
    recording = Recording(imported["id"], trace.load_trace(imported["id"]))
    choices = [
        recording.metadata(position)
        for position in recording.positions
        if position.kind == "choice"
    ]
    assert [choice["branched"] for choice in choices] == [None] * (len(choices) - 1) + [
        ["determinize"]
    ]
    assert "35" not in json.dumps(choices[-1]["branched"])


def test_standalone_snapshot_replay_and_unterminated_journal_row(
    tmp_path: Path,
) -> None:
    env = managym.Env(seed=42)
    env.reset(setup())
    saved = json.loads(env.snapshot().to_replay_json())
    for seat in saved["player_configs"]:
        del seat["sideboard"]
    path = tmp_path / "position.json"
    path.write_text(json.dumps(saved, indent=2))
    records = records_from_source(path)
    assert records is not None and len(records) == 1
    assert records[0].games[0].setup()[0].sideboard == {}
    journal = path.with_suffix(".jsonl")
    journal.write_text(json.dumps(saved))
    with pytest.warns(RuntimeWarning, match="Incomplete final game record"):
        assert records_from_source(journal) == []


def test_pending_cast_payment_command_survives_save_and_reload() -> None:
    env = managym.Env(seed=42)
    deck = {"Mountain": 16, "Forest": 16, "Gray Ogre": 16}
    env.reset(
        managym.GameSetup([managym.PlayerConfig(name, deck) for name in ("a", "b")])
    )
    for _ in range(32):
        actor = env.current_agent_index()
        assert actor is not None
        actions = env.observation_for_player(actor).action_space.actions
        taps = [
            i
            for i, action in enumerate(actions)
            if action.action_type == managym.ActionEnum.TAP_FOR_COST
        ]
        if taps:
            break
        index = next(
            (
                i
                for kind in (
                    managym.ActionEnum.PRIORITY_PLAY_LAND,
                    managym.ActionEnum.PRIORITY_CAST_SPELL,
                )
                for i, action in enumerate(actions)
                if action.action_type == kind
            ),
            0,
        )
        env.step(index)
    else:
        pytest.fail("Gray Ogre never reached manual mana payment")
    frame = decision_frame(env)
    command = Command(
        "tap-source", frame.revision, frame.offers[taps[0]]["id"]
    ).to_json()
    restored = cold_restore(env)
    for branch in (env, restored):
        branch.execute_semantic_command_json(command)
    assert restored.search_witness_json() == env.search_witness_json()


def test_pending_adventure_target_command_survives_cold_restore() -> None:
    env = managym.Env(seed=42)
    deck = {
        "Mountain": 24,
        "Gray Ogre": 8,
        "Lightning Bolt": 4,
        "Smaug, the Great Calamity": 8,
    }
    env.reset(
        managym.GameSetup([managym.PlayerConfig(name, deck) for name in ("a", "b")])
    )
    for _ in range(320):
        frame = decision_frame(env)
        adventure = next(
            (offer for offer in frame.offers if "Spew Flame" in offer["label"]), None
        )
        if adventure is not None:
            obs = env.observation_for_player(frame.actor)
            ogres = [
                p
                for p in [*obs.agent_permanents, *obs.opponent_permanents]
                if p.characteristics.name == "Gray Ogre"
            ]
            if len(ogres) >= 2:
                env.execute_semantic_command_json(
                    Command("adventure", frame.revision, adventure["id"]).to_json()
                )
                target_frame = decision_frame(env)
                assert all(
                    offer["verb"] == "choose_target" for offer in target_frame.offers
                )
                break

        def priority(index: int) -> int:
            offer = frame.offers[index]
            if offer["verb"] == "play_land":
                return 0
            if (offer.get("public_commitment") or {}).get("card") == "Gray Ogre":
                return 1
            if offer["verb"] == "pass_priority":
                return 2
            return 3

        chosen = min(range(len(frame.offers)), key=priority)
        env.step(chosen)
    else:
        pytest.fail("ordinary play never reached Spew Flame's target decision")

    command = Command(
        "target", target_frame.revision, target_frame.offers[0]["id"]
    ).to_json()
    restored = cold_restore(env)
    assert restored.semantic_decision_frame_json() == env.semantic_decision_frame_json()
    before_payment = {
        p.id
        for p in env.observation_for_player(frame.actor).agent_permanents
        if p.tapped
    }
    assert_replayed_command(env, restored, command)
    for branch in (env, restored):
        observation = branch.observation_for_player(frame.actor)
        paid = [
            p
            for p in observation.agent_permanents
            if p.tapped and p.id not in before_payment
        ]
        assert len(paid) == 5 and all(
            p.characteristics.name == "Mountain" for p in paid
        )
        [spell] = observation.stack_objects
        assert spell.characteristics is not None
        assert spell.characteristics.name == "Spew Flame"
        assert spell.targets[0].permanent_id in {ogre.id for ogre in ogres}
    assert restored.search_witness_json() == env.search_witness_json()


def test_pending_exude_toxin_x_replays_payment_and_resolution_shuffle() -> None:
    env = managym.Env(seed=42)
    env.reset(
        managym.GameSetup(
            [
                managym.PlayerConfig("a", {"Swamp": 24, "Scavenger Regent": 12}),
                managym.PlayerConfig("b", {"Mountain": 24, "Lightning Bolt": 12}),
            ]
        )
    )
    for _ in range(160):
        frame = decision_frame(env)
        obs = env.observation_for_player(0)
        toxin = next((o for o in frame.offers if "Exude Toxin" in o["label"]), None)
        if toxin and sum(not p.tapped for p in obs.agent_permanents) >= 3:
            env.execute_semantic_command_json(
                Command("omen", frame.revision, toxin["id"]).to_json()
            )
            break
        index = next(
            (
                i
                for verb in ("play_land", "pass_priority")
                for i, offer in enumerate(frame.offers)
                if offer["verb"] == verb
            ),
            0,
        )
        env.step(index)
    else:
        pytest.fail("ordinary play never offered the payable X spell")

    pending_frame = env.semantic_decision_frame_json()
    frame = DecisionFrame.from_json(pending_frame)
    zero_branch = env.snapshot().restore()
    zero = next(o for o in frame.offers if "X = 0" in o["label"])
    zero_branch.execute_semantic_command_json(
        Command("announce-zero", frame.revision, zero["id"]).to_json()
    )
    payment = decision_frame(zero_branch)
    pay = next(offer for offer in payment.offers if offer["verb"] == "pay_cost")
    zero_branch.execute_semantic_command_json(
        Command("pay-two", payment.revision, pay["id"]).to_json()
    )
    [zero_spell] = zero_branch.observation_for_player(0).stack_objects
    assert zero_spell.effective_mana_value == 2
    assert (
        sum(p.tapped for p in zero_branch.observation_for_player(0).agent_permanents)
        == 2
    )
    assert env.semantic_decision_frame_json() == pending_frame

    [one] = [offer for offer in frame.offers if "X = 1" in offer["label"]]
    command = Command("announce-one", frame.revision, one["id"]).to_json()
    restored = cold_restore(env)
    assert_replayed_command(env, restored, command)
    before = json.loads(env.search_witness_json())["rng_continuation"]
    space = ObservationSpace()
    for branch in (env, restored):
        assert "cost {X}{B}{B}; mana value 3" in capture_text_decision(branch).text
        for viewer in (0, 1):
            obs = branch.observation_for_player(viewer)
            row = encode_observation(space, obs)["stack"][0]
            assert row[column("stack", "mana_value")] == pytest.approx(0.2)
            assert row[column("stack", "effective_mana_value")] == pytest.approx(0.3)
            [item] = obs.stack_objects
            assert item.characteristics is not None
            assert item.characteristics.name == "Exude Toxin"
            assert item.characteristics.mana_cost.mana_value == 2
            assert item.effective_mana_value == 3
        assert (
            sum(p.tapped for p in branch.observation_for_player(0).agent_permanents)
            == 3
        )
    replay_priority_until_stack_empty(env, restored)
    assert json.loads(env.search_witness_json())["rng_continuation"] != before
    assert env.search_witness_json() == restored.search_witness_json()
    assert env.state_digest() == restored.state_digest()


def test_etali_resumes_once_from_a_cold_charring_bite_target() -> None:
    etali = "Etali, Primal Storm"
    env = managym.Env(seed=2)
    env.enable_statistics()
    env.enable_exposure_capture()
    env.reset(
        managym.GameSetup(
            [
                managym.PlayerConfig("a", {"Mountain": 28, etali: 8, "Gray Ogre": 12}),
                managym.PlayerConfig(
                    "b", {"Twinmaw Stormbrood": 24, "Lightning Bolt": 4, "Mountain": 10}
                ),
            ]
        )
    )

    def completed() -> int:
        stats = env.statistics()
        assert stats is not None
        count = next(row.resolutions for row in stats.cards if row.name == etali)
        assert count is not None
        return count

    for _ in range(120):
        frame = decision_frame(env)
        obs = env.observation_for_player(frame.actor)
        bite = next(
            (o for o in frame.offers if o["label"] == "Cast Charring Bite"), None
        )
        if bite and len(frame.offers) == 4:
            before = completed()
            env.execute_semantic_command_json(
                Command("bite", frame.revision, bite["id"]).to_json()
            )
            break
        etalis = {p.id for p in obs.agent_permanents if p.characteristics.name == etali}
        if frame.offers[0]["verb"] == "declare_attacker":
            index = next(
                i
                for i, a in enumerate(obs.action_space.actions)
                if a.declared == bool(etalis.intersection(a.focus))
            )
        else:

            def preference(index: int) -> int:
                offer = frame.offers[index]
                card = (offer.get("public_commitment") or {}).get("card")
                if offer["verb"] == "play_land":
                    return 0
                if offer["verb"] == "cast" and card == etali and not etalis:
                    return 1
                if offer["verb"] == "cast" and card == "Gray Ogre":
                    return 2
                return 3 if offer["verb"] == "pass_priority" else 4

            index = min(range(len(frame.offers)), key=preference)
        env.step(index)
    else:
        pytest.fail("Etali never offered both exiled cards")

    frame = decision_frame(env)
    assert len(frame.offers) > 1 and all(
        o["verb"] == "choose_target" for o in frame.offers
    )
    restored = cold_restore(env)
    fork = env.snapshot().restore()
    assert decision_frame(restored) == frame
    target = frame.offers[0]["id"]
    with pytest.raises(managym.AgentError, match="revision|stale"):
        restored.execute_semantic_command_json(
            Command("stale", frame.revision - 1, target).to_json()
        )
    assert restored.search_witness_json() == fork.search_witness_json()
    command = Command("target", frame.revision, target).to_json()
    assert_replayed_command(env, restored, command)
    assert decision_frame(fork) == frame
    remaining = decision_frame(env)
    assert decision_frame(restored) == remaining
    assert [o["verb"] for o in remaining.offers] == ["select_card", "decline"]
    assert completed() == before
    command = Command(
        "cast-ogre", remaining.revision, remaining.offers[0]["id"]
    ).to_json()
    assert_replayed_command(env, restored, command)
    assert completed() == before + 1
    assert [
        s.characteristics.name for s in env.observation_for_player(0).stack_objects
    ] == ["Gray Ogre", "Charring Bite"]
    assert restored.search_witness_json() == env.search_witness_json()


def test_transferred_seed_suture_replays_payment_and_surviving_link() -> None:
    angel = "Blossom-Blessed Angel"
    medic = "Emergency Phytomedic"
    thief = "Agent of Treachery"
    env = managym.Env(seed=42)
    env.reset(
        managym.GameSetup(
            [
                managym.PlayerConfig(
                    "owner",
                    {
                        "Plains": 20,
                        "Island": 8,
                        angel: 8,
                        medic: 8,
                        "Spectral Sailor": 8,
                    },
                ),
                managym.PlayerConfig("caster", {"Island": 20, "Plains": 12, thief: 12}),
            ]
        )
    )
    for _ in range(400):
        frame = decision_frame(env)
        obs = env.observation_for_player(0)
        names = {p.characteristics.name for p in obs.agent_permanents}
        suture = next(
            (
                o
                for o in frame.offers
                if o["verb"] == "cast" and "Seed Suture" in o["label"]
            ),
            None,
        )
        if frame.actor == 1 and suture:
            break
        acting = env.observation_for_player(frame.actor)
        angels = {p.id for p in obs.agent_permanents if p.characteristics.name == angel}

        def preference(index: int) -> int:
            offer = frame.offers[index]
            card = (offer.get("public_commitment") or {}).get("card")
            if offer["verb"] == "play_land":
                return 0
            if offer["verb"] == "cast" and (
                (card in (angel, medic) and card not in names)
                or (card == thief and {angel, medic} <= names)
            ):
                return 1
            if offer["verb"] == "choose_target" and angels.intersection(
                acting.action_space.actions[index].focus
            ):
                return 2
            return 3 if offer["verb"] == "pass_priority" else 4

        env.step(min(range(len(frame.offers)), key=preference))
    else:
        pytest.fail("ordinary play never offered the transferred Angel's Seed Suture")

    restored = cold_restore(env)
    assert restored.semantic_decision_frame_json() == env.semantic_decision_frame_json()
    before = env.observation_for_player(1)
    untapped = {p.id for p in before.agent_permanents if not p.tapped}
    command = Command("cast-suture", frame.revision, suture["id"]).to_json()
    assert_replayed_command(env, restored, command)
    target = decision_frame(env)
    command = Command("target", target.revision, target.offers[0]["id"]).to_json()
    assert_replayed_command(env, restored, command)
    payment = decision_frame(env)
    pay = next(o for o in payment.offers if o["verb"] == "pay_cost")
    command = Command("pay", payment.revision, pay["id"]).to_json()
    assert_replayed_command(env, restored, command)
    for branch in (env, restored):
        obs = branch.observation_for_player(1)
        [paid] = [p for p in obs.agent_permanents if p.tapped and p.id in untapped]
        assert paid.characteristics.name == "Plains"
        [spell] = obs.stack_objects
        assert spell.characteristics is not None
        assert spell.characteristics.name == "Seed Suture"
        assert spell.controller_id == obs.agent.id
        [source] = [
            p for p in obs.opponent_permanents if p.characteristics.name == medic
        ]
        [copy] = [
            c
            for c in obs.opponent_cards
            if c.cast_permissions and c.characteristics.name == "Seed Suture"
        ]
        assert copy.cast_permissions[0].follows_source_controller
        assert copy.cast_permissions[0].source_permanent_id == source.id
        assert copy.cast_permissions[0].player_id == obs.opponent.id
    assert restored.search_witness_json() == env.search_witness_json()


def test_pending_pact_upkeep_restores_and_replays_its_loss_command() -> None:
    env = managym.Env(seed=23)
    decks = [
        managym.PlayerConfig("Pact", {"Swamp": 20, "Demonic Pact": 8}),
        managym.PlayerConfig("Opponent", {"Plains": 28}),
    ]
    obs, _ = env.reset(managym.GameSetup(decks, mana_payment=("auto", "auto")))
    for _ in range(100):
        actions = obs.action_space.actions
        if len(actions) == 4 and all(
            action.action_type == managym.ActionEnum.CHOOSE_MODE for action in actions
        ):
            break
        choice = preferred_action(
            actions,
            managym.ActionEnum.PRIORITY_PLAY_LAND,
            managym.ActionEnum.PRIORITY_CAST_SPELL,
            managym.ActionEnum.PRIORITY_PASS_PRIORITY,
        )
        obs = env.step(choice)[0]
    else:
        pytest.fail("Pact never offered its four upkeep modes")

    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    command = Command("pact-loss", frame.revision, frame.offers[3]["id"]).to_json()
    before = env.search_witness_json()
    position = env.snapshot()
    reopened = managym.GameSnapshot.from_replay_json(position.to_replay_json())
    successors = []
    for snapshot in (position, reopened):
        for semantic in (False, True):
            branch = snapshot.restore()
            assert branch.search_witness_json() == before
            if semantic:
                _, _, reward, terminated, truncated, info = (
                    branch.step_semantic_command(command)
                )
            else:
                _, reward, terminated, truncated, info = branch.step(3)
            assert (reward, terminated, truncated) == (-1.0, True, False)
            assert (
                sim.determine_outcome(terminated, info) == sim.GameOutcome.VILLAIN_WIN
            )
            assert info["winner_index"] == 1
            assert info["winner_name"] == "Opponent"
            assert info["p0_loss_reason"] == "instructed_loss"
            successors.append(branch.search_witness_json())
    assert all(successor == successors[0] for successor in successors)
    assert env.search_witness_json() == before


def test_simulation_counts_native_outcomes_on_last_allowed_step() -> None:
    records = []
    deck = {"Plains": 20, "Herald of Eternal Dawn": 4}
    player = sim.DefaultPlayer("first action")
    for losses, protected, expected in (
        ((1,), False, sim.GameOutcome.HERO_WIN),
        ((0,), False, sim.GameOutcome.VILLAIN_WIN),
        ((0, 1), False, sim.GameOutcome.DRAW),
        ((0,), True, sim.GameOutcome.TIMEOUT),
    ):
        native = managym.Env(seed=23)
        reset_to_first_main(
            native,
            managym.GameSetup(
                [managym.PlayerConfig(str(seat), deck) for seat in (0, 1)]
            ),
        )
        if protected:
            native.scenario_force_battlefield(0, "Herald of Eternal Dawn")
        for seat in losses:
            native.scenario_set_life(seat, 0)
        obs = native.scenario_refresh()
        env = Mock(spec=sim.Env)
        env.last_raw_obs = obs
        env.reset.return_value = (obs, {})
        env.step.side_effect = native.step
        record = sim._simulate_game(env, player, player, 1)
        assert record.steps == 1
        assert record.outcome == expected
        assert native.is_game_over() == (not protected)
        records.append(record)

    summary = sim.simulation_summary(records)
    assert summary["total_games"] == 4
    assert summary["draws"] == summary["timeouts"] == 1
    assert summary["hero_wins"] == summary["villain_wins"] == 1
    assert summary["hero_win_rate"] == summary["villain_win_rate"] == 0.25


@pytest.mark.parametrize("info", [{}, {"winner_index": 2, "winner_name": "draw"}])
def test_simulation_rejects_unknown_terminal_result(info: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="native winner or draw"):
        sim.determine_outcome(True, info)


# Natural games that reach a private or owner-only choice from their seed. The
# shared script prefers these offers, in order, for whichever seat acts.
CHOICE_SCRIPT = (
    "Keep hand",
    "Play land",
    r"Cast .*(Manifest Dread|Lost Days|Sun-Blessed Healer|Llanowar Elves)",
    r"Target .*Llanowar",
    "Pay the offered cost",
    "Do not attack",
    "Pass priority",
)
CHOICES: dict[str, tuple[dict[str, int], dict[str, int], str]] = {
    "manifest selection": (
        {"Forest": 32, "Manifest Dread": 4, "Llanowar Elves": 4},
        {"Plains": 40},
        "Native description: Manifest",
    ),
    "kicker payment": (
        {"Plains": 28, "Sun-Blessed Healer": 12},
        {"Forest": 40},
        r"Decline / finish the offered choice .*Sun-Blessed Healer",
    ),
    "owner's library position": (
        {"Island": 28, "Lost Days": 12},
        {"Forest": 28, "Llanowar Elves": 12},
        "Native description: On the bottom",
    ),
}


def play_to_choice(choice: str) -> managym.Env:
    hero, villain, offered = CHOICES[choice]
    env = managym.Env(seed=4)
    env.reset(
        managym.GameSetup(
            [
                managym.PlayerConfig("hero", hero),
                managym.PlayerConfig("villain", villain),
            ]
        )
    )
    for _ in range(300):
        decision = capture_text_decision(env)
        if any(re.search(offered, a.description) for a in decision.actions):
            return env
        action = next(
            (
                a
                for pattern in CHOICE_SCRIPT
                for a in decision.actions
                if re.search(pattern, a.description)
            ),
            decision.actions[0],
        )
        submit_response(env, decision, action.reference)
    pytest.fail(f"{choice} was never offered")


@pytest.mark.parametrize("choice", CHOICES)
def test_private_choices_fork_commit_once_and_replay_from_the_seed(choice: str) -> None:
    env = play_to_choice(choice)
    actor = env.current_agent_index()
    assert actor is not None
    # Only the chooser is offered the choice.
    assert len(env.observation_for_player(1 - actor).action_space.actions) == 0
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    assert len(frame.offers) >= 2
    root = env.snapshot()
    pending = managym.GameSnapshot.from_replay_json(root.to_replay_json()).restore()
    assert pending.state_digest() == env.state_digest()
    assert pending.semantic_decision_frame_json() == env.semantic_decision_frame_json()

    consequences = set()
    for offer in frame.offers:
        branch = root.restore()
        command = Command(f"{choice}.{offer['id']}", frame.revision, offer["id"])
        branch.execute_semantic_command_json(command.to_json())
        committed = branch.state_digest()
        with pytest.raises(managym.AgentError):
            branch.execute_semantic_command_json(command.to_json())
        assert branch.state_digest() == committed
        replay = managym.GameSnapshot.from_replay_json(
            branch.snapshot().to_replay_json()
        ).restore()
        assert replay.state_digest() == committed
        assert (
            replay.semantic_decision_frame_json()
            == branch.semantic_decision_frame_json()
        )
        consequences.add(committed)
    assert len(consequences) == len(frame.offers)
    assert env.state_digest() == root.state_digest


def test_waypoint_target_replays_payment_and_new_prepared_copy() -> None:
    waypoint = "Skycoach Waypoint"
    creature = "Stingerquill Voxmancer"
    env = managym.Env(seed=42)
    env.reset(
        managym.GameSetup(
            [
                managym.PlayerConfig(
                    "caster", {waypoint: 12, "Swamp": 20, creature: 10}
                ),
                managym.PlayerConfig("opponent", {"Mountain": 20, "Wall of Stone": 10}),
            ],
            mana_payment=("auto", "auto"),
        )
    )
    for _ in range(350):
        frame = decision_frame(env)
        obs = env.observation_for_player(frame.actor)
        lands = {
            p.id for p in obs.agent_permanents if p.characteristics.name == waypoint
        }
        linked = {
            grant.source_permanent_id
            for card in obs.agent_cards
            for grant in card.cast_permissions
        }
        unprepared = {
            p.id
            for p in obs.agent_permanents
            if p.characteristics.name == creature and p.id not in linked
        }
        activation = next(
            (
                i
                for i, offer in enumerate(frame.offers)
                if offer["verb"] == "activate_ability"
                and lands.intersection(obs.action_space.actions[i].focus)
            ),
            None,
        )
        if frame.actor == 0 and activation is not None and unprepared:
            env.step(activation)
            target = decision_frame(env)
            choices = env.observation_for_player(0).action_space.actions
            index, source = next(
                (i, candidate)
                for i, action in enumerate(choices)
                for candidate in action.focus
                if candidate in unprepared
            )
            ready = {p.id for p in obs.agent_permanents if not p.tapped}
            break

        def preference(index: int) -> int:
            offer = frame.offers[index]
            name = (offer.get("public_commitment") or {}).get("card")
            if offer["verb"] == "play_land":
                return 0 if (name == waypoint) == (not lands) else 1
            if offer["verb"] == "cast" and name in {creature, "Wall of Stone"}:
                return 2
            if offer["verb"] == "cast" and name == "Vicious Verse":
                return 3
            return 4 if offer["verb"] == "pass_priority" else 5

        env.step(min(range(len(frame.offers)), key=preference))
    else:
        pytest.fail("no Waypoint activation with an unprepared Voxmancer")

    restored = cold_restore(env)
    assert decision_frame(restored) == target
    command = Command("prepare", target.revision, target.offers[index]["id"]).to_json()
    assert_replayed_command(env, restored, command)
    paid = [
        p
        for p in env.observation_for_player(0).agent_permanents
        if p.id in ready and p.tapped
    ]
    assert len(paid) == 4
    assert any(p.characteristics.name == waypoint for p in paid)

    def linked_verse(branch: managym.Env, viewer: int) -> int:
        obs = branch.observation_for_player(viewer)
        [copy] = [
            card
            for card in [*obs.agent_cards, *obs.opponent_cards]
            if any(g.source_permanent_id == source for g in card.cast_permissions)
            and card.characteristics.name == "Vicious Verse"
        ]
        return copy.id

    replay_priority_until_stack_empty(env, restored)
    for viewer in (0, 1):
        assert linked_verse(env, viewer) == linked_verse(restored, viewer)
    assert env.search_witness_json() == restored.search_witness_json()
