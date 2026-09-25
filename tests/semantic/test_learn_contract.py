"""Learn through the ordinary Python Command, history and world readers."""

import json
import math
import random

import pytest

from manabot.belief.encoding import belief_schema_from_engine
from manabot.belief.learning import BeliefTrainingExample, pack_belief_examples
from manabot.belief.likelihood import _matching_offer_indexes
from manabot.belief.range import BeliefState
from manabot.belief.state import ViewerHistory
from manabot.belief.tracker import BeliefTracker
from manabot.semantic.decision_contract import SemanticDecisionContract
import managym
from managym.decision import Observation, SemanticContractError
from managym.possible_worlds import PossibleWorldError, PossibleWorldSpace, WorldQuery


@pytest.fixture(scope="module")
def learn_root():
    """Reach the recorded seed-0 Learn with legal Commands, without injection."""
    env = managym.Env(seed=0, skip_trivial=False)
    env.reset(
        [
            managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
            for key in ("ur_lessons", "gw_allies")
        ]
    )
    rng = random.Random(0)
    for _ in range(2000):
        assert not env.is_game_over(), "match ended before Learn"
        contract = SemanticDecisionContract.from_env(env)
        if any(offer["verb"] == "learn_take_lesson" for offer in contract.frame.offers):
            return env
        contract.apply(env, rng.choice(contract.frame.offers)["id"])
    pytest.fail("no Learn in bounded prefix")


@pytest.mark.parametrize(
    "verb,kind",
    [
        ("learn_take_lesson", "learn_take_lesson"),
        ("learn_discard", "discard"),
        ("decline", "decline_learn"),
    ],
)
def test_learn_commands_history_and_isolation(learn_root, verb, kind):
    source = learn_root.search_witness_json()
    env = learn_root.clone_env()
    contract = SemanticDecisionContract.from_env(env)
    actor = contract.frame.actor
    viewer = 1 - actor
    before = Observation.from_json(env.semantic_observation_json(viewer))
    assert before.decision is None
    history = ViewerHistory.from_observation(before)
    offer = contract.frame.find_verb(verb)
    indexes, count = _matching_offer_indexes(contract.frame, offer["public_commitment"])
    assert contract.frame.offers.index(offer) in indexes
    assert count == len(contract.frame.offers)

    transition = contract.apply(env, offer["id"])
    after = Observation.from_json(env.semantic_observation_json(viewer))
    history = history.advance(transition.receipt, after, acting=actor)
    assert history.semantic_events[-1].commitment.kind == kind
    assert after.opponent_hand_is_hidden()
    if verb == "learn_take_lesson":
        assert sum(after.viewer_state["opponent"]["known_hand"].values()) == 1
    committed = env.search_witness_json()
    with pytest.raises(SemanticContractError, match="stale"):
        contract.apply(env, offer["id"])
    assert env.search_witness_json() == committed
    assert learn_root.search_witness_json() == source


def test_retrieval_transports_belief_into_reveal_constrained_native_space(learn_root):
    env = learn_root.clone_env()
    contract = SemanticDecisionContract.from_env(env)
    viewer = 1 - contract.frame.actor
    before = PossibleWorldSpace.from_engine(env, viewer)
    history = ViewerHistory.from_observation(
        Observation.from_json(env.semantic_observation_json(viewer))
    )
    prior = BeliefState.compatible_prior(before)
    offer = contract.frame.find_verb("learn_take_lesson")
    lesson = offer["public_commitment"]["card"]
    transition = contract.apply(env, offer["id"])
    after = PossibleWorldSpace.from_engine(env, viewer)
    history = history.advance(
        transition.receipt,
        Observation.from_json(env.semantic_observation_json(viewer)),
        acting=contract.frame.actor,
    )
    assert after.known_hand == ((lesson, 1),)
    assert after.world_schema_identity == "managym.possible-world-space/v2"
    assert after.total_weight == math.comb(
        sum(dict(after.pool).values()) - 1, after.hand_size - 1
    )
    assert after.support(WorldQuery.lacks(lesson)).support_size == 0
    assert all(world.count(lesson) >= 1 for world in after.worlds)
    exits, returns, draws = BeliefTracker._public_transport_facts(
        before, after, offer["public_commitment"]
    )
    assert (exits, returns, draws) == ([], [lesson], 0)
    transported = prior.transport(
        after, known_exits=exits, known_returns=returns, hidden_draws=draws
    )
    assert transported.normalization_error < 1e-12
    assert transported.probabilities == pytest.approx(
        BeliefState.compatible_prior(after).probabilities
    )
    witness = env.search_witness_json()
    view = env.semantic_observation_json(viewer)
    for index in (0, after.support_size - 1):
        branch = after.materialize(index, seed=42)
        assert branch.semantic_observation_json(viewer) == view
    assert env.search_witness_json() == witness

    # Exercise the existing belief-learning input consumer without training.
    schema = belief_schema_from_engine(env, after, count_buckets=16)
    packed = pack_belief_examples(
        (
            BeliefTrainingExample(
                world_space=after,
                viewer_history=history,
                target_world=0,
                supervision_receipt="input-shape-only",
            ),
        ),
        schema,
    )
    assert packed.history_kind_ids.numel() == 1
    assert packed.history_card_indexes.item() > 0
    assert packed.world_counts.shape[0] == after.support_size


def test_known_hand_fixture_identity_and_constraints():
    args = dict(
        viewer=0,
        source_revision=1,
        source_viewer_state_hash="fixture",
        pool={"Lesson": 2, "Island": 2},
        hands=[({"Lesson": 1, "Island": 1}, 2), ({"Lesson": 2}, 1)],
    )
    space = PossibleWorldSpace.from_fixture(**args, known_hand={"Lesson": 1})
    assert space.identity != PossibleWorldSpace.from_fixture(**args).identity
    assert space.support(WorldQuery.lacks("Lesson")).support_size == 0
    assert space.support(WorldQuery.has("Lesson")).canonical_query == {"kind": "true"}
    assert space.support(WorldQuery.exactly("Lesson", 0)).canonical_query == {
        "kind": "empty"
    }
    assert space.support(WorldQuery.has("Island", 2)).support_size == 0
    with pytest.raises(PossibleWorldError, match="omits known"):
        PossibleWorldSpace.from_fixture(**args, known_hand={"Lesson": 2})


@pytest.mark.parametrize("version", [1, 3, "2", 2.0, True])
def test_world_reader_rejects_incompatible_version(version):
    class IncompatibleEngine:
        def possible_world_space_json(self, viewer):
            return json.dumps({"schema_version": version})

    with pytest.raises(PossibleWorldError, match="unsupported PossibleWorldSpace"):
        PossibleWorldSpace.from_engine(IncompatibleEngine(), 0)


@pytest.mark.parametrize(
    "known_hand", [None, [], {"Lesson": 0}, {"Lesson": 1.5}, {"Lesson": True}]
)
def test_world_reader_requires_explicit_known_counts(known_hand):
    class MalformedEngine:
        def possible_world_space_json(self, viewer):
            return json.dumps(
                {
                    "schema_version": 2,
                    "source_observation": {"schema_version": 5, "viewer": viewer},
                    "viewer": viewer,
                    "known_hand": known_hand,
                }
            )

    with pytest.raises(PossibleWorldError, match="known_hand"):
        PossibleWorldSpace.from_engine(MalformedEngine(), 0)


@pytest.mark.parametrize("version", [1, 4, 6, "5", 5.0, True])
def test_world_reader_rejects_incompatible_source_observation(version):
    class IncompatibleEngine:
        def possible_world_space_json(self, viewer):
            return json.dumps(
                {
                    "schema_version": 2,
                    "source_observation": {"schema_version": version},
                }
            )

    with pytest.raises(PossibleWorldError, match="unsupported source observation"):
        PossibleWorldSpace.from_engine(IncompatibleEngine(), 0)


def test_learn_outside_descriptors_programs_and_native_json_parity(learn_root):
    from manabot.semantic.learning import BoundSemanticPack

    env = learn_root.clone_env()
    retained = env.search_witness_json()
    pack = BoundSemanticPack.from_env(env)
    contract = SemanticDecisionContract.from_env(env)
    actor = contract.frame.actor
    before = env.observation_for_player(actor)
    other = env.observation_for_player(1 - actor)
    assert {card.name for card in before.agent_sideboard} == {
        "Accumulate Wisdom",
        "Firebending Lesson",
        "It'll Quench Ya!",
    }
    assert {card.name for card in other.agent_sideboard} == {
        "Fancy Footwork",
        "Yip Yip!",
    }
    for viewer in (actor, 1 - actor):
        raw = env.observation_for_player(viewer)
        canonical = Observation.from_json(env.semantic_observation_json(viewer))
        assert json.loads(raw.toJSON()) == canonical.viewer_state
        assert raw.validate()
        assert len(raw.agent.zone_counts) == 7
        assert all(not hasattr(card, "zone") for card in raw.agent_sideboard)
        assert all(not hasattr(card, "id") for card in raw.agent_sideboard)

    projection = pack.project_observation(before)
    role = pack.schema.object_roles["agent_sideboard"]
    outside = projection.object_roles == role
    assert projection.object_slots[outside].tolist() == [0, 1, 2]
    assert projection.object_definition_rows[outside].tolist() == [
        pack.definition_row(card.registry_key) for card in before.agent_sideboard
    ]
    for row in projection.object_definition_rows[outside]:
        assert pack.program_rows(int(row))
    assert not projection.opaque_identity_valid.any()

    retrieval = contract.frame.find_verb("learn_take_lesson")
    action_index = next(
        i for i, offer in enumerate(contract.frame.offers) if offer == retrieval
    )
    (candidate_id,) = before.action_space.actions[action_index].focus
    selected = next(
        card for card in before.agent_sideboard if card.candidate_id == candidate_id
    )
    assert selected.name == retrieval["public_commitment"]["card"]
    contract.apply(env, retrieval["id"])
    after = env.observation_for_player(actor)
    assert len(after.agent_sideboard) == 2
    assert all(card.candidate_id != candidate_id for card in after.agent_sideboard)
    assert any(
        card.id == candidate_id and int(card.zone) == 1 for card in after.agent_cards
    )
    assert after.agent.sideboard_counts == before.agent.sideboard_counts
    assert after.agent.remaining_sideboard_counts.get(selected.registry_key, 0) == 0
    assert after.agent.known_hand == {selected.registry_key: 1}
    observer = env.observation_for_player(1 - actor)
    assert (
        observer.opponent.remaining_sideboard_counts
        == after.agent.remaining_sideboard_counts
    )
    assert observer.opponent.known_hand == after.agent.known_hand
    assert all(int(card.zone) != 1 for card in observer.opponent_cards)
    assert (
        json.loads(observer.toJSON())
        == Observation.from_json(env.semantic_observation_json(1 - actor)).viewer_state
    )
    assert learn_root.search_witness_json() == retained


def test_learn_bounded_encoders_preserve_kinds_outside_focus_and_overflow(learn_root):
    import numpy as np

    from manabot.env.observation import ObservationEncoder
    from manabot.infra.hypers import ObservationSpaceHypers

    obs = learn_root.observation_for_player(0)
    scalar = ObservationEncoder(ObservationSpaceHypers()).encode(obs)
    native = learn_root.encode_observation(obs)
    for key in scalar:
        np.testing.assert_array_equal(scalar[key], native[key], err_msg=key)
    retrievals = 0
    for index, action in enumerate(obs.action_space.actions):
        kind = int(action.action_type)
        if kind not in (14, 15):
            continue
        assert scalar["actions"][index, kind] == 1
        assert scalar["actions"][index, :-1].sum() == 1
        focus = int(scalar["action_focus"][index, 0])
        assert focus >= 2
        row = scalar["agent_cards"][focus - 2]
        if kind == 14:
            retrievals += 1
            candidate = obs.agent_sideboard[focus - 2 - len(obs.agent_cards)]
            assert candidate.candidate_id == action.focus[0]
            assert row[:7].sum() == 0  # outside is not an eighth game zone
            assert row[-2:].tolist() == [1, 1]
        else:
            assert row[1] == 1  # hand
            assert row[-2] == 0
    assert retrievals == 3
    for hypers in (
        ObservationSpaceHypers(max_actions=2),
        ObservationSpaceHypers(max_cards_per_player=len(obs.agent_cards)),
        ObservationSpaceHypers(max_focus_objects=0),
    ):
        with pytest.raises(ValueError, match="capacity exceeded"):
            ObservationEncoder(hypers).encode(obs)


def test_learn_vector_buffers_match_scalar_at_every_prefix_step():
    import numpy as np

    from manabot.env.observation import ObservationEncoder
    from manabot.infra.hypers import ObservationSpaceHypers

    encoder = ObservationEncoder(ObservationSpaceHypers())
    buffers = encoder.allocate(1)
    buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    configs = [
        managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
        for key in ("ur_lessons", "gw_allies")
    ]
    vector = managym.VectorEnv(1, seed=0, skip_trivial=False)
    vector.set_buffers(buffers)
    vector.reset_all_into_buffers(configs)
    scalar = managym.Env(seed=0, skip_trivial=False)
    obs, _ = scalar.reset(configs)
    rng = random.Random(0)
    for _ in range(2000):
        encoded = encoder.encode(obs)
        for key, value in encoded.items():
            np.testing.assert_array_equal(buffers[key][0], value, err_msg=key)
        retrievals = [
            i
            for i, action in enumerate(obs.action_space.actions)
            if int(action.action_type) == 14
        ]
        if retrievals:
            selected = retrievals[-1]
            after, *_ = scalar.step(selected)
            vector.step_into_buffers([selected])
            for key, value in encoder.encode(after).items():
                np.testing.assert_array_equal(buffers[key][0], value, err_msg=key)
            return
        assert not obs.game_over
        selected = rng.randrange(len(obs.action_space.actions))
        obs, *_ = scalar.step(selected)
        vector.step_into_buffers([selected])
    pytest.fail("Learn not reached within prefix cap")
