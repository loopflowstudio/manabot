from __future__ import annotations

import json
import math

import pytest

from manabot.sim.structured_policy import (
    PolicyScores,
    RaggedPolicyDecoder,
    SeededSemanticScorer,
    StructuredPolicyError,
    flatten_projection,
)
import managym


def _candidate(candidate_id: int) -> dict[str, object]:
    return {
        "id": candidate_id,
        "value": {"kind": "subject", "subject": {"kind": "player", "id": 0}},
        "label": f"candidate-{candidate_id}",
        "help": None,
        "preview": None,
    }


def _projection(count: int, minimum: int = 1, maximum: int = 1) -> dict[str, object]:
    return {
        "actor": 0,
        "kind": "priority",
        "offers": [
            {
                "id": 7,
                "actor": 0,
                "verb": "cast",
                "source": None,
                "label": "fixture",
                "help": None,
                "choices": [
                    {
                        "kind": "select",
                        "role": 3,
                        "label": "target",
                        "candidates": {
                            "id": 1,
                            "depends_on": [],
                            "initial": [_candidate(i) for i in range(count)],
                        },
                        "min": minimum,
                        "max": maximum,
                        "ordered": False,
                        "distinct": True,
                    }
                ],
                "confirm_label": "choose",
            }
        ],
    }


def test_decoder_consumes_35_candidates_without_a_fixed_width() -> None:
    batch = flatten_projection(_projection(35))
    scores = SeededSemanticScorer(189).score(batch, 4)
    submission = RaggedPolicyDecoder().decode(batch, scores)

    assert batch.max_candidate_count == 35
    assert batch.max_legal_branches == 35
    assert len(submission.answers[0]["candidates"]) == 1
    assert submission.answers[0]["candidates"][0] in range(35)


def test_six_attacker_rows_represent_all_64_subsets_without_enumeration() -> None:
    batch = flatten_projection(_projection(6, minimum=0, maximum=6))
    scores = PolicyScores(offer_scores=(1.0,), candidate_scores=(-1, 1, -1, 1, -1, 1))
    submission = RaggedPolicyDecoder().decode(batch, scores)

    assert len(batch.candidates) == 6
    assert batch.max_legal_branches == 64
    assert submission.answers == (
        {"kind": "candidates", "role": 3, "candidates": [1, 3, 5]},
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda projection: projection.update(offers=[]), "at least one offer"),
        (
            lambda projection: projection["offers"].append(projection["offers"][0]),
            "duplicate offer",
        ),
        (
            lambda projection: projection["offers"][0]["choices"][0][
                "candidates"
            ].update(depends_on=[3]),
            "dependencies",
        ),
    ],
)
def test_projection_validation_fails_closed(mutate, message: str) -> None:
    projection = _projection(2)
    mutate(projection)
    with pytest.raises(StructuredPolicyError, match=message):
        flatten_projection(projection)


def test_decoder_rejects_misaligned_and_nonfinite_scores() -> None:
    batch = flatten_projection(_projection(2))
    decoder = RaggedPolicyDecoder()

    with pytest.raises(StructuredPolicyError, match="candidate score count"):
        decoder.decode(batch, PolicyScores((0.0,), (0.0,)))
    with pytest.raises(StructuredPolicyError, match="finite"):
        decoder.decode(batch, PolicyScores((0.0,), (0.0, math.nan)))


def _engine() -> tuple[managym.Env, managym.Observation]:
    env = managym.Env(seed=189, skip_trivial=False)
    observation, _ = env.reset(
        [
            managym.PlayerConfig("structured", {"Lightning Bolt": 4, "Mountain": 36}),
            managym.PlayerConfig("legacy", {"Gray Ogre": 36, "Mountain": 4}),
        ]
    )
    return env, observation


def _decode_verb(
    offers: managym.StructuredOfferSet, verb: str, candidate_scores: tuple[float, ...]
) -> tuple[object, str]:
    projection = json.loads(offers.projection_json())
    batch = flatten_projection(projection)
    offer_scores = tuple(
        1.0 if offer["verb"] == verb else -1.0 for offer in batch.offers
    )
    submission = RaggedPolicyDecoder().decode(
        batch,
        PolicyScores(
            offer_scores=offer_scores,
            candidate_scores=candidate_scores,
        ),
    )
    return batch, submission.to_json()


def test_bound_bridge_applies_35_target_cast_through_both_abis() -> None:
    structured, _ = _engine()
    structured.scenario_clear_hand(0)
    structured.scenario_clear_hand(1)
    structured.scenario_force_card_in_hand(0, "Lightning Bolt")
    structured.scenario_force_battlefield(0, "Mountain")
    for _ in range(33):
        structured.scenario_force_battlefield(1, "Gray Ogre")
    structured.scenario_refresh()

    legacy = structured.clone_env()
    offers = structured.structured_offers()
    batch, submission = _decode_verb(
        offers, "cast", tuple(float(index) for index in range(35))
    )
    assert batch.max_candidate_count == 35

    structured_result = structured.step_structured(offers, submission)
    legacy_result = legacy.step_legacy_submission(offers, submission)

    assert structured_result[5] == 1
    assert legacy_result[5] == 2
    assert structured.state_digest() == legacy.state_digest()
    assert structured_result[0].toJSON() == legacy_result[0].toJSON()


def test_bound_bridge_rejects_fabricated_and_stale_ids_without_mutation() -> None:
    env, _ = _engine()
    offers = env.structured_offers()
    before = env.state_digest()
    with pytest.raises(managym.AgentError, match="unknown offer"):
        env.step_structured(offers, '{"offer_id":999,"answers":[]}')
    assert env.state_digest() == before

    _, pass_submission = _decode_verb(offers, "pass_priority", ())
    env.step_structured(offers, pass_submission)
    before_stale = env.state_digest()
    with pytest.raises(managym.AgentError, match="no longer names"):
        env.step_structured(offers, pass_submission)
    assert env.state_digest() == before_stale


def test_bound_bridge_applies_one_of_64_attacker_declarations() -> None:
    structured = managym.Env(seed=190, skip_trivial=False)
    observation, _ = structured.reset(
        [
            managym.PlayerConfig("structured", {"Gray Ogre": 36, "Mountain": 4}),
            managym.PlayerConfig("legacy", {"Gray Ogre": 36, "Mountain": 4}),
        ]
    )
    structured.scenario_clear_hand(0)
    for _ in range(6):
        structured.scenario_force_battlefield(0, "Gray Ogre", ready=True)
    observation = structured.scenario_refresh()

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
        observation, _, done, _, _ = structured.step(pass_index)
        assert not done
    else:
        pytest.fail("fixture did not reach declare attackers")

    legacy = structured.clone_env()
    offers = structured.structured_offers()
    batch, submission = _decode_verb(
        offers, "declare_attackers", (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    )
    assert batch.max_legal_branches == 64

    structured.step_structured(offers, submission)
    legacy_result = legacy.step_legacy_submission(offers, submission)

    assert legacy_result[5] == 6
    assert structured.state_digest() == legacy.state_digest()
