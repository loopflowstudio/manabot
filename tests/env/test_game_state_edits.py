"""Replayable edits through scalar and recorded vector Python APIs."""

from contextlib import closing
import json
from pathlib import Path

import numpy as np
from pydantic import ValidationError
import pytest

import managym
from managym.decision import Command, DecisionFrame
from managym.possible_worlds import PossibleWorldSpace
from managym.recording import GameStateEdits, RecordedVectorEnv, read_record
from managym.replay import walk


def chooser(env: managym.Env) -> managym.Observation:
    seat = env.current_agent_index()
    assert seat is not None
    return env.observation_for_player(seat)


def play_default(obs: managym.Observation) -> int:
    for kind in (
        managym.ActionEnum.PRIORITY_PLAY_LAND,
        managym.ActionEnum.PRIORITY_CAST_SPELL,
        managym.ActionEnum.PRIORITY_PASS_PRIORITY,
    ):
        for index, action in enumerate(obs.action_space.actions):
            if action.action_type == kind:
                return index
    return 0


def setup() -> managym.GameSetup:
    deck = {"Mountain": 28, "Gray Ogre": 8, "Hill Giant": 4}
    return managym.GameSetup([managym.PlayerConfig(str(seat), deck) for seat in (0, 1)])


@pytest.fixture
def blocking_game() -> managym.Env:
    env = managym.Env(seed=42, decision_memory=True, decision_memory_include_own=True)
    env.reset(setup())
    for _ in range(300):
        obs = chooser(env)
        kind = obs.action_space.action_space_type
        if kind == managym.ActionSpaceEnum.DECLARE_BLOCKER:
            assert sum(p.attacking for p in obs.opponent_permanents) >= 2
            return env
        if kind == managym.ActionSpaceEnum.DECLARE_ATTACKER:
            # Wait for a full history and a combat with alternate block targets.
            history = json.loads(obs.toJSON()).get("policy_history", [])
            attackers = sum(
                p.attacking or (not p.tapped and not p.is_summoning_sick)
                for p in obs.agent_permanents
                if p.characteristics.card_types.is_creature
            )
            blockers = sum(
                not p.tapped
                for p in obs.opponent_permanents
                if p.characteristics.card_types.is_creature
            )
            attack = len(history) == 32 and attackers >= 2 and blockers >= 2
            env.step(
                next(
                    i
                    for i, action in enumerate(obs.action_space.actions)
                    if action.declared == attack
                )
            )
        else:
            env.step(play_default(obs))
    pytest.fail("seeded game did not reach a choice between blockers")


def test_edits_round_trip_at_declare_blockers(
    blocking_game: managym.Env, tmp_path: Path
) -> None:
    env = blocking_game
    obs = chooser(env)
    root = env.snapshot()
    before = root.to_replay_json()
    orig = json.loads(obs.toJSON())
    card_links = [
        ref["Card"]
        for row in orig["policy_history"]
        for role in ("source", "target")
        if isinstance(ref := row[role], dict) and "Card" in ref
    ]
    links = {ref["context_id"] for ref in card_links} - {None}
    moved = next(
        p
        for p in obs.opponent_permanents
        if p.attacking and p.characteristics.name == "Hill Giant"
    )
    assert moved.id in links or moved.object_id in links
    focal_id = obs.action_space.actions[0].focus[0]
    land = next(p for p in obs.agent_permanents if p.characteristics.card_types.is_land)
    edit = managym.GameStateEdit
    edits = (
        edit.move_permanent(moved.id, to="library_bottom"),
        edit.set_tapped(land.id, not land.tapped),
        edit.set_damage(focal_id, 1),
        edit.set_counters(focal_id, kind="plus_one_plus_one", count=2),
        edit.set_life(obs.opponent.player_index, 5),
    )
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    stale = Command("before-edit", frame.revision, frame.offers[0]["id"])

    probe = env.clone_env()
    updated = probe.edit_game_state(edits)
    edited = json.loads(updated.toJSON())
    remaining = {p.id: p for p in updated.agent_permanents}
    assert remaining[focal_id].damage == 1
    assert remaining[focal_id].plus1_counters == 2
    assert remaining[land.id].tapped != land.tapped
    assert updated.opponent.life == 5
    assert moved.id not in {p.id for p in updated.opponent_permanents}
    before_focus = {tuple(action.focus) for action in obs.action_space.actions}
    assert (focal_id, moved.id) in before_focus
    assert {tuple(action.focus) for action in updated.action_space.actions} == (
        before_focus - {(focal_id, moved.id)}
    )
    assert updated.decision_memory_json() == obs.decision_memory_json()
    for key in ("turn", "stack_objects", "recent_events"):
        assert edited.get(key) == orig.get(key)
    for before_player, after_player in (
        (orig["agent"], edited["agent"]),
        (orig["opponent"], edited["opponent"]),
    ):
        for key in ("mana", "turn_history"):
            assert before_player.get(key) == after_player.get(key)

    for key in ("agent_permanents", "opponent_permanents"):
        touched = {moved.id, focal_id, land.id}
        assert [p for p in edited[key] if p["id"] not in touched] == [
            p for p in orig[key] if p["id"] not in touched
        ]
    for ref in card_links:
        if ref["context_id"] in (moved.id, moved.object_id):
            ref["context_id"] = None
    assert len(edited["policy_history"]) == 32
    assert edited["policy_history"] == orig["policy_history"]
    frozen = probe.decision_memory_counts()
    probe.step(0)
    assert probe.decision_memory_counts() == frozen
    assert (
        managym.GameSnapshot.from_replay_json(
            probe.snapshot().to_replay_json()
        ).state_digest
        == probe.state_digest()
    )

    for options in (
        managym.RestoreOptions.exact(game_state_edits=edits),
        managym.RestoreOptions.shuffle_libraries(19, game_state_edits=edits),
        managym.RestoreOptions.determinize(
            obs.agent.player_index, 19, game_state_edits=edits
        ),
    ):
        branch = root.restore(options)
        branch_obs = chooser(branch)
        owner = branch_obs.opponent
        bottom = owner.zone_counts[managym.ZoneEnum.LIBRARY] - 1
        assert owner.known_library[bottom] == moved.registry_key
        world_space = PossibleWorldSpace.from_engine(
            branch, branch_obs.agent.player_index
        )
        assert dict(world_space.known_library)[moved.characteristics.name] == 1
        with pytest.raises(managym.AgentError, match="stale"):
            branch.execute_semantic_command_json(stale.to_json())
        frame = DecisionFrame.from_json(branch.semantic_decision_frame_json())
        command = Command("after-edit", frame.revision, frame.offers[0]["id"])
        raw = branch.snapshot().to_replay_json()
        cold = managym.GameSnapshot.from_replay_json(raw).restore()
        assert cold.state_digest() == branch.state_digest()
        assert (
            cold.semantic_decision_frame_json() == branch.semantic_decision_frame_json()
        )
        assert chooser(cold).toJSON() == branch_obs.toJSON()
        for live in (branch, cold):
            live.execute_semantic_command_json(command.to_json())
        assert cold.state_digest() == branch.state_digest()
        assert branch.decision_memory_counts()[frame.actor][0] > frozen[frame.actor][0]
        path = tmp_path / "edited.json"
        path.write_text(branch.snapshot().to_replay_json())
        record = read_record(path)
        intervention = record.games[0].interventions[0]
        assert isinstance(intervention, GameStateEdits)
        assert (
            intervention.model_dump()["edits"]
            == json.loads(raw)["interventions"][0]["edits"]
        )
        replay = walk(record)
        decisions = list(replay)
        assert replay.stop is not None and replay.stop.status == "interrupted", (
            replay.stop
        )
        assert decisions[-1].interventions[0] == intervention
        assert replay.env.state_digest() == branch.state_digest()

    assert env.snapshot().to_replay_json() == before


def test_edit_errors_leave_the_boundary_and_journal_intact(
    blocking_game: managym.Env,
) -> None:
    env = blocking_game
    obs = chooser(env)
    root = env.snapshot()
    before = root.to_replay_json()
    focal = obs.action_space.actions[0].focus[0]
    edit = managym.GameStateEdit
    life = edit.set_life(obs.agent.player_index, 7)
    assert env.edit_game_state([]).toJSON() == obs.toJSON()
    for bad, message in (
        (edit.set_damage(focal, -1), "negative"),
        (edit.set_tapped(focal, True), "current declarer ineligible"),
    ):
        with pytest.raises(managym.AgentError, match=message):
            env.edit_game_state([life, bad])
        assert env.snapshot().to_replay_json() == before
    with pytest.raises(managym.AgentError, match="forced_first_action"):
        root.restore(
            managym.RestoreOptions.exact(forced_first_action=0, game_state_edits=[life])
        )
    with pytest.raises(managym.AgentError, match="unknown counter"):
        edit.set_counters(focal, "missing", 2)
    with pytest.raises(managym.AgentError, match="unknown variant"):
        edit.move_permanent(focal, to="missing")  # type: ignore[arg-type]
    with pytest.raises(managym.AgentError, match="player"):
        edit.set_life(-1, 7)
    with pytest.raises(managym.AgentError, match="before reset"):
        managym.Env().edit_game_state([life])


def test_edited_priority_branch_loads_in_a_recorded_vector(
    blocking_game: managym.Env,
) -> None:
    env = blocking_game
    for _ in range(16):
        obs = chooser(env)
        if obs.action_space.action_space_type == managym.ActionSpaceEnum.PRIORITY:
            break
        env.step(
            next(i for i, a in enumerate(obs.action_space.actions) if not a.declared)
        )
    else:
        pytest.fail("combat did not reach a priority decision")
    root = env.snapshot()
    land = next(p for p in obs.agent_permanents if p.characteristics.card_types.is_land)
    edits = [managym.GameStateEdit.move_permanent(land.id)]
    options = managym.RestoreOptions.exact(game_state_edits=edits)
    scalar = root.restore(options)
    probe = env.clone_env()
    assert probe.edit_game_state(edits).toJSON() == chooser(scalar).toJSON()
    assert scalar.state_digest() == probe.state_digest()
    assert chooser(scalar).agent.zone_counts[managym.ZoneEnum.GRAVEYARD] == (
        obs.agent.zone_counts[managym.ZoneEnum.GRAVEYARD] + 1
    )

    with closing(
        RecordedVectorEnv(num_envs=1, decision_memory_include_own=True)
    ) as vector:
        layout = managym.ObservationLayout()
        vector.set_observation_layout(
            {
                "rewards": np.zeros(1),
                "terminated": np.zeros(1, dtype=np.uint8),
                "truncated": np.zeros(1, dtype=np.uint8),
            },
            layout=layout,
        )
        vector.reset_all_into_buffers(setup())
        [loaded] = vector.load_slots_into_buffers(
            [managym.SlotLoad.snapshot(0, root, options)]
        )
        assert loaded.source_digest == root.state_digest
        assert vector.native.state_digests() == [scalar.state_digest()]
        [loaded_obs] = vector.native.decision_observations([True])
        assert (
            loaded_obs is not None and loaded_obs.toJSON() == chooser(scalar).toJSON()
        )
        vector.step_into_buffers([0])
        scalar.step(0)
        assert vector.native.state_digests() == [scalar.state_digest()]
        inputs = vector.game_inputs(0)
        [intervention] = inputs.interventions
        assert isinstance(intervention, GameStateEdits)
        assert intervention.at_action == len(inputs.actions) - 1


def test_game_state_edits_record_rejects_untyped_payloads() -> None:
    with pytest.raises(ValidationError, match="valid boolean"):
        GameStateEdits.model_validate(
            {
                "op": "game_state_edits",
                "at_action": 0,
                "edits": [{"kind": "set_tapped", "permanent": 42, "tapped": 1}],
            }
        )


def test_edits_read_as_their_constructor_calls() -> None:
    edits = [
        managym.GameStateEdit.move_permanent(42, to="library_bottom"),
        managym.GameStateEdit.set_tapped(7, True),
        managym.GameStateEdit.set_damage(7, 3),
        managym.GameStateEdit.set_counters(7, "plus_one_plus_one", 2),
        managym.GameStateEdit.set_life(1, 5),
    ]
    namespace = {"GameStateEdit": managym.GameStateEdit}
    for edit in edits:
        assert eval(repr(edit), namespace) == edit
    assert managym.GameStateEdit.set_life(1, 5) != managym.GameStateEdit.set_life(0, 5)
    assert len(set(edits + edits)) == len(edits)
