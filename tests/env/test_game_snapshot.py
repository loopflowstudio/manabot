"""A private scalar position can outlive its producer and reopen as a live game."""

import json
from pathlib import Path
import subprocess
import sys
from typing import Literal

from blake3 import blake3
import pytest

import managym
from managym.decision import Command, DecisionFrame
from managym.recording import (
    ScalarCapture,
    read_record,
    record_from_row,
    records_from_source,
)
from managym.replay import replay_failure_inputs, walk
from tests.fixtures.recorded_commands import record_until_object_candidate


def setup(decklists: Literal["open", "closed"] = "closed") -> managym.GameSetup:
    return managym.GameSetup(
        [
            managym.PlayerConfig(name, {"Mountain": 8, "Gray Ogre": 4})
            for name in ("a", "b")
        ],
        decklists=decklists,
    )


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
    unissued_cold = managym.GameSnapshot.from_replay_json(
        unissued.snapshot().to_replay_json()
    ).restore()
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
    obs, _ = env.reset(setup())
    meta = env.snapshot().metadata()
    assert meta.turn == 1
    assert meta.phase == managym.PhaseEnum.PRECOMBAT_MAIN
    assert meta.life == [20, 20]
    assert meta.board_counts == [0, 0]
    assert not meta.active_card_defs and meta.stack_len == 0
    assert meta.action_count == len(obs.action_space.actions)
    assert meta.acting_player == obs.agent.player_index
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
    obs, _ = source.reset(setup("open"))
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
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
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
    source.reset(setup())
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
    assert json.loads(raw)["interventions"][-1]["at_action"] == 3
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
    assert [decision.chosen for decision in replay] == [0, 0, 0]
    assert replay.stop is not None and replay.stop.status == "interrupted"
    assert position_evidence(replay.env) == before

    imported = import_game(json.loads(raw))
    assert (imported["stop"], imported["count"]) == ("interrupted", 3)
    for ordinal in range(3):
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

    record.games[0].committed = 3
    record.games[0].actions.append(999)
    record.games[0].frame = frame
    record.execution_outcome = "failed"
    record.error_class = type(rejected.value).__name__
    record.error = str(rejected.value)
    (result,) = replay_failure_inputs(record)
    assert result["action_index"] == 3
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
    from managym.academy.watch import import_game, inspect_watch

    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    env = managym.Env(seed=42)
    original = env.reset(setup("open"))[0]
    branch = env.snapshot().restore(managym.RestoreOptions.determinize(1, 35))
    offers = branch.observation_for_player(0).action_space.actions
    assert len(offers) != len(original.action_space.actions)
    branch.step(len(offers) - 1)
    imported = import_game(json.loads(branch.snapshot().to_replay_json()))
    shown = inspect_watch(imported["id"], 0, imported["token"])
    assert len(shown["decision"]["frame"]["offers"]) == len(offers)
    assert shown["decision"]["offer"]["label"] == "Pass priority"


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
        choice = next(
            (
                i
                for kind in (
                    managym.ActionEnum.PRIORITY_PLAY_LAND,
                    managym.ActionEnum.PRIORITY_CAST_SPELL,
                    managym.ActionEnum.PRIORITY_PASS_PRIORITY,
                )
                for i, action in enumerate(actions)
                if action.action_type == kind
            ),
            0,
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
        branch = snapshot.restore()
        assert branch.search_witness_json() == before
        branch.execute_semantic_command_json(command)
        for _ in range(3):
            if branch.is_game_over():
                break
            chooser = branch.current_agent_index()
            actions = branch.observation_for_player(chooser).action_space.actions
            branch.step(
                next(
                    i
                    for i, action in enumerate(actions)
                    if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
                )
            )
        assert branch.is_game_over()
        successors.append(branch.search_witness_json())
    assert successors[0] == successors[1]
    assert env.search_witness_json() == before
