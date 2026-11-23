"""A private scalar position can outlive its producer and reopen as a live game."""

import json
from pathlib import Path
import subprocess
import sys
from typing import Literal

import pytest

import managym
from managym.decision import Command
from managym.recording import ScalarCapture, record_from_row
from managym.replay import walk
from tests.fixtures.recorded_commands import record_until_object_candidate


def setup() -> managym.GameSetup:
    return managym.GameSetup(
        [
            managym.PlayerConfig(name, {"Mountain": 8, "Gray Ogre": 4})
            for name in ("a", "b")
        ],
        decklists="closed",
    )


def test_exact_recipe_reopens_in_another_process(tmp_path: Path) -> None:
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
    raw = branch.snapshot().to_recipe_json()
    recipe = json.loads(raw)
    assert recipe["seed"] == 42
    assert recipe["actions"] == [0, 0]
    assert [i["at_action"] for i in recipe["interventions"]] == [1, 2]
    assert recipe["player_configs"][1]["disclosure"] == "closed"

    path = tmp_path / "position.json"
    path.write_text(raw)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, sys
from pathlib import Path
import managym
env = managym.GameSnapshot.from_recipe_json(Path(sys.argv[1]).read_text()).restore()
env.step(0)
print(json.dumps([env.state_digest(), env.decision_memory_counts()]))
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    branch.step(0)
    assert json.loads(child.stdout) == [
        branch.state_digest(),
        branch.decision_memory_counts(),
    ]
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
    issued = managym.GameSnapshot.from_recipe_json(position.to_recipe_json()).restore()
    issued.execute_semantic_command_json(command)
    # An identical position reached by index alone never issued the address.
    unissued = managym.Env(seed=23)
    unissued.reset(capture.inputs.setup())
    for action in capture.inputs.actions:
        assert isinstance(action, int)
        unissued.step(action)
    assert unissued.state_digest() == source.state_digest()
    unissued_cold = managym.GameSnapshot.from_recipe_json(
        unissued.snapshot().to_recipe_json()
    ).restore()
    with pytest.raises(managym.AgentError, match="unknown_object_candidate"):
        unissued_cold.execute_semantic_command_json(command)
    # Execute the already-issued address without asking the branch for a frame.
    source.execute_semantic_command_json(command)
    restored.execute_semantic_command_json(command)
    assert restored.state_digest() == source.state_digest()
    recipe = restored.snapshot().to_recipe_json()
    assert json.loads(json.loads(recipe)["actions"][ordinal]) == json.loads(command)
    cold = managym.GameSnapshot.from_recipe_json(recipe).restore()
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
    recipe = source.snapshot().to_recipe_json()
    assert json.loads(recipe)["actions"] == ([0] if "agent" in policies else [-1]) * 3
    cold = managym.GameSnapshot.from_recipe_json(recipe).restore()
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
        if env.is_game_over():
            break
        env.step(0)
    terminal = env.snapshot()
    meta = terminal.metadata()
    assert meta.terminal and meta.action_count == 0
    assert meta.action_kind is None and meta.acting_player is None
    assert terminal.restore().is_game_over()
    assert (
        managym.GameSnapshot.from_recipe_json(terminal.to_recipe_json())
        .metadata()
        .terminal
    )


def test_snapshot_rejects_corrupt_recipe_and_retains_unjournaled_hot_state() -> None:
    source = managym.Env(seed=42)
    source.reset(setup())
    snapshot = source.snapshot()
    raw = snapshot.to_recipe_json()
    for patch, reason in [
        ({"schema": "unknown"}, "invalid snapshot"),
        ({"card_digest": "bad"}, "card_digest"),
        ({"state_digest": "bad"}, "recipe_digest"),
        ({"player_policies": ["random", "random"]}, "recipe_digest"),
        ({"decision_memory": True}, "recipe_digest"),
        ({"actions": [999]}, "recipe_digest"),
        (
            {"interventions": [{"op": "reseed", "seed": 1, "at_action": 2}]},
            "recipe_digest",
        ),
    ]:
        with pytest.raises(managym.AgentError, match=reason):
            managym.GameSnapshot.from_recipe_json(json.dumps(json.loads(raw) | patch))
    source.random_action_index()
    hot = source.snapshot()
    assert hot.restore().state_digest() == source.state_digest()
    with pytest.raises(managym.AgentError, match="unjournaled"):
        hot.to_recipe_json()
    assert snapshot.to_recipe_json() == raw


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
