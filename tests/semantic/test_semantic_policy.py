"""Real-engine proofs for the semantic structured-decision boundary."""

from __future__ import annotations

import pytest

from manabot.semantic import (
    SemanticDecision,
    SemanticDecisionAdapter,
    SemanticDecisionError,
    UnadmittedDefinitionError,
)
from manabot.sim.structured_policy import (
    DecodedSubmission,
    PolicyScores,
    RaggedPolicyDecoder,
)
import managym


def _target_root() -> tuple[managym.Env, managym.Observation]:
    env = managym.Env(seed=20_001, skip_trivial=False)
    env.reset(
        [
            managym.PlayerConfig("caster", {"Igneous Inspiration": 4, "Mountain": 36}),
            managym.PlayerConfig("target", {"Fire Nation Cadets": 36, "Mountain": 4}),
        ]
    )
    env.scenario_clear_hand(0)
    env.scenario_force_card_in_hand(0, "Igneous Inspiration")
    for _ in range(3):
        env.scenario_force_battlefield(0, "Mountain")
    for _ in range(33):
        env.scenario_force_battlefield(1, "Fire Nation Cadets")
    observation = env.scenario_refresh()
    for _ in range(100):
        if int(observation.agent.player_index) == 0 and any(
            action.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
            for action in observation.action_space.actions
        ):
            return env, observation
        pass_index = next(
            (
                index
                for index, action in enumerate(observation.action_space.actions)
                if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
            ),
            0,
        )
        observation, _, done, _, _ = env.step(pass_index)
        if done:
            break
    raise RuntimeError("Igneous Inspiration did not reach a castable priority window")


def _decision() -> tuple[managym.Env, SemanticDecisionAdapter, SemanticDecision]:
    env, observation = _target_root()
    adapter = SemanticDecisionAdapter.from_env(env)
    decision = adapter.bind(
        env,
        observation,
        match_id="semantic-policy-targets",
        revision=7,
        content_hash=adapter.pack.content_pack_hash,
        asset_manifest_hash="semantic-policy-test-assets",
    )
    return env, adapter, decision


def _cast_submission(decision: SemanticDecision) -> DecodedSubmission:
    return RaggedPolicyDecoder().decode(
        decision.batch,
        PolicyScores(
            offer_scores=tuple(
                1.0 if offer["verb"] == "cast" else -1.0
                for offer in decision.batch.offers
            ),
            candidate_scores=tuple(
                float(index) for index in range(len(decision.batch.candidates))
            ),
        ),
    )


def test_semantic_decision_binds_and_applies_real_35_target_command() -> None:
    env, adapter, decision = _decision()
    frame = decision.frame.model_dump(mode="json", exclude_unset=True)

    assert frame["projection"]["opponent"]["hand"] == []
    assert frame["projection"]["opponent"]["hand_hidden_count"] > 0
    assert frame["revision"] == 7
    assert frame["prompt"]["kind"] == "priority"
    assert frame["offers"] == list(decision.batch.offers)
    assert decision.batch.max_candidate_count == 35

    cast_index = next(
        index
        for index, offer in enumerate(decision.batch.offers)
        if offer["verb"] == "cast"
    )
    source_binding = decision.offer_sources[cast_index]
    assert source_binding is not None
    assert source_binding.object_row is not None
    source_row = decision.objects[source_binding.object_row]
    assert source_row.object_kind == "card"
    assert source_row.zone == "hand"
    assert source_row.program_rows
    assert (
        adapter.pack.ir.definitions[source_row.definition_row]["semantic_key"]
        == "stx.igneous_inspiration"
    )

    player_targets = [
        binding
        for binding in decision.candidate_subjects
        if binding.player_index is not None
    ]
    object_targets = [
        decision.objects[binding.object_row]
        for binding in decision.candidate_subjects
        if binding.object_row is not None
    ]
    assert {binding.player_index for binding in player_targets} == {0, 1}
    assert len(object_targets) == 33
    assert all(row.object_kind == "permanent" for row in object_targets)
    assert all(row.zone == "battlefield" for row in object_targets)
    assert {
        adapter.pack.ir.definitions[row.definition_row]["semantic_key"]
        for row in object_targets
    } == {"tla.fire_nation_cadets"}

    submission = _cast_submission(decision)
    command = decision.command(submission)
    assert command.expected_revision == frame["revision"]
    assert command.prompt_id == frame["prompt"]["id"]
    assert command.offer_id == frame["offers"][cast_index]["id"]
    assert command.answers[0].candidates == [34]

    before = env.state_digest()
    result = decision.step(env, command)

    assert result[5] == 1
    assert env.state_digest() != before


def test_semantic_decision_rejects_unbound_ids_and_revisions_before_step() -> None:
    env, _, decision = _decision()
    before = env.state_digest()

    fabricated = DecodedSubmission(
        offer_id=_cast_submission(decision).offer_id,
        answers=(
            {
                "kind": "candidates",
                "role": decision.batch.choices[0].role,
                "candidates": [999],
            },
        ),
    )
    with pytest.raises(SemanticDecisionError, match="unknown candidates"):
        decision.command(fabricated)

    command = decision.command(_cast_submission(decision))
    stale = command.model_copy(
        update={"expected_revision": command.expected_revision + 1}
    )
    with pytest.raises(SemanticDecisionError, match="revision"):
        decision.submission_json(stale)
    assert env.state_digest() == before


def test_semantic_decision_binds_one_of_64_complete_attacker_commands() -> None:
    env = managym.Env(seed=20_003, skip_trivial=False)
    observation, _ = env.reset(
        [
            managym.PlayerConfig("attacker", {"Fire Nation Cadets": 36, "Mountain": 4}),
            managym.PlayerConfig("defender", {"Fire Nation Cadets": 36, "Mountain": 4}),
        ]
    )
    env.scenario_clear_hand(0)
    for _ in range(6):
        env.scenario_force_battlefield(0, "Fire Nation Cadets", ready=True)
    observation = env.scenario_refresh()
    for _ in range(100):
        if (
            observation.action_space.action_space_type
            == managym.ActionSpaceEnum.DECLARE_ATTACKER
        ):
            break
        pass_index = next(
            (
                index
                for index, action in enumerate(observation.action_space.actions)
                if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
            ),
            0,
        )
        observation, _, done, _, _ = env.step(pass_index)
        assert not done
    else:
        pytest.fail("fixture did not reach declare attackers")

    adapter = SemanticDecisionAdapter.from_env(env)
    decision = adapter.bind(
        env,
        observation,
        match_id="semantic-policy-attackers",
        revision=11,
        content_hash=adapter.pack.content_pack_hash,
        asset_manifest_hash="semantic-policy-test-assets",
    )

    assert decision.frame.prompt is not None
    assert decision.frame.prompt.kind == "declare_attackers"
    assert decision.batch.max_legal_branches == 64
    assert decision.frame.offers[0].focus == [
        binding.subject.entity for binding in decision.candidate_subjects
    ]
    assert all(
        decision.objects[binding.object_row].object_kind == "permanent"
        for binding in decision.candidate_subjects
        if binding.object_row is not None
    )

    submission = RaggedPolicyDecoder().decode(
        decision.batch,
        PolicyScores(
            offer_scores=(1.0,),
            candidate_scores=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        ),
    )
    command = decision.command(submission)
    result = decision.step(env, command)

    assert command.answers[0].candidates == [1, 3, 5]
    assert result[5] == 1


def test_semantic_decision_fails_before_scoring_an_unadmitted_definition() -> None:
    env = managym.Env(seed=20_002, skip_trivial=False)
    env.reset(
        [
            managym.PlayerConfig("caster", {"Lightning Bolt": 4, "Mountain": 36}),
            managym.PlayerConfig("target", {"Fire Nation Cadets": 36, "Mountain": 4}),
        ]
    )
    env.scenario_clear_hand(0)
    env.scenario_force_card_in_hand(0, "Lightning Bolt")
    env.scenario_force_battlefield(0, "Mountain")
    observation = env.scenario_refresh()
    adapter = SemanticDecisionAdapter.from_env(env)

    with pytest.raises(UnadmittedDefinitionError, match="absent from semantic IR"):
        adapter.bind(
            env,
            observation,
            match_id="semantic-policy-unadmitted",
            revision=0,
            content_hash=adapter.pack.content_pack_hash,
            asset_manifest_hash="semantic-policy-test-assets",
        )
