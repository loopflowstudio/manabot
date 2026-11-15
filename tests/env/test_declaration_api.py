"""Public declaration transactions retain ordinary Command and capture identity."""

import json

from pydantic import ValidationError
import pytest

import managym
from managym.experience import (
    DeclarationAnswer,
    DeclarationPlan,
    DeclarationProjection,
    ObjectSubject,
)


def _attack() -> tuple[managym.Env, DeclarationProjection]:
    env = managym.Env(seed=364, skip_trivial=True)
    env.enable_card_capture()
    env.reset(
        [
            managym.PlayerConfig("hero", {"Mountain": 24, "Boggart Brute": 16}),
            managym.PlayerConfig("villain", {"Mountain": 24, "Gray Ogre": 16}),
        ]
    )
    for _ in range(2):
        env.scenario_force_battlefield(0, "Boggart Brute", ready=True)
        env.scenario_force_battlefield(1, "Gray Ogre", ready=True)
    env.scenario_refresh()
    for index in range(20):
        projection = json.loads(env.declaration_projection_json(0))
        if projection is not None:
            parsed = DeclarationProjection.model_validate(projection)
            assert parsed.stage == "attackers"
            assert len(parsed.subjects) == 2
            assert all(
                row.object.label == "Boggart Brute" and row.power == 3
                for row in parsed.subjects
            )
            return env, parsed
        frame = json.loads(env.semantic_decision_frame_json())
        offer = next(row for row in frame["offers"] if row["verb"] == "pass_priority")
        env.step_semantic_command(
            json.dumps(
                {
                    "command_id": f"setup.{index}",
                    "expected_revision": frame["revision"],
                    "offer_id": offer["id"],
                    "answers": [],
                }
            )
        )
    pytest.fail("constructed position did not reach attack declaration")


def _plan(projection: DeclarationProjection) -> DeclarationPlan:
    answers: list[DeclarationAnswer] = []
    for row in projection.subjects:
        assert isinstance(row.object.subject, ObjectSubject)
        if not row.fixed:
            answers.append(
                DeclarationAnswer(
                    subject=row.object.subject.id,
                    destination=row.destinations[0],
                )
            )
    return DeclarationPlan(binding=projection.binding, assignments=answers)


def test_declaration_candidate_retains_child_roots_and_publishes_once() -> None:
    env, projection = _attack()
    assert json.loads(env.declaration_projection_json(1)) is None
    plan = _plan(projection)
    before = env.search_witness_json()
    capture_before = env.card_capture_json()
    env.validate_declaration_plan_json(plan.model_dump_json())
    prepared = env.prepare_declaration_plan_json(plan.model_dump_json())
    evidence = json.loads(prepared.to_json())
    assert env.search_witness_json() == before
    assert env.card_capture_json() == capture_before
    replay = env.clone_env()
    for index, command in enumerate(evidence["commands"]):
        assert (
            prepared.root(index).search_witness_json() == replay.search_witness_json()
        )
        actual, _, _, _, _, _ = replay.step_semantic_command(json.dumps(command))
        child = prepared.result(index)
        assert (
            json.loads(actual)
            == json.loads(child.transition_json)
            == evidence["transitions"][index]
        )
        assert child.env.search_witness_json() == replay.search_witness_json()
        assert child.observation.game_over is child.done
    candidate_capture = prepared.result(
        len(evidence["commands"]) - 1
    ).env.card_capture_json()
    prepared.publish()
    assert env.search_witness_json() == replay.search_witness_json()
    with pytest.raises(managym.AgentError, match="consumed"):
        prepared.publish()
    assert env.card_capture_json() == candidate_capture
    assert (
        json.loads(candidate_capture)["capture"]["initial"]
        == json.loads(capture_before)["capture"]["initial"]
    )


def test_declaration_rejections_preserve_the_current_root() -> None:
    env, projection = _attack()
    plan = _plan(projection)
    before = env.search_witness_json()
    missing = plan.model_dump(mode="json")
    del missing["assignments"][0]["destination"]
    with pytest.raises(ValidationError, match="destination"):
        DeclarationPlan.model_validate(missing)
    with pytest.raises(managym.AgentError, match="missing field.*destination"):
        env.validate_declaration_plan_json(json.dumps(missing))
    assert env.search_witness_json() == before
    duplicate = plan.model_copy(deep=True)
    duplicate.assignments[1] = duplicate.assignments[0]
    with pytest.raises(managym.AgentError, match="duplicate"):
        env.validate_declaration_plan_json(duplicate.model_dump_json())
    assert env.search_witness_json() == before
    capture = env.capture_decision(0)
    prepared = env.prepare_declaration_plan_json(plan.model_dump_json())
    stale = env.prepare_declaration_plan_json(plan.model_dump_json())
    command = json.loads(prepared.to_json())["commands"][0]
    prepared.publish()
    after = env.search_witness_json()
    with pytest.raises(managym.AgentError, match="stale declaration"):
        stale.publish()
    assert env.search_witness_json() == after
    # Publication retains this Env's origin; old captures fail as stale inputs,
    # rather than being spuriously attributed to another environment.
    with pytest.raises(managym.AgentError, match="stale_capture"):
        env.execute_captured_command_json(capture, json.dumps(command))
