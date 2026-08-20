"""Behavioral checks for canonical belief and query operations."""

import numpy as np
import pytest

from manabot.belief import (
    HAND_ZONE_ID,
    LIBRARY_ZONE_ID,
    OPPONENT_OWNER_ROLE_ID,
    BeliefEncodingSchema,
    BeliefError,
    BeliefRow,
    BeliefState,
    CompatibleDealBeliefModel,
    EmptyBeliefSupport,
    ViewerHistory,
    condition_belief,
    encode_belief,
    query_mass,
)
from managym.decision import (
    SEMANTIC_DECISION_VERSION,
    Observation,
    PublicCommitment,
    SemanticContractError,
    TransitionReceipt,
)
from managym.possible_worlds import WorldQuery
from tests.belief.support import fixture_history, fixture_schema, fixture_space


def generated_belief() -> BeliefState:
    history = fixture_history()
    space = fixture_space(history)
    return (
        CompatibleDealBeliefModel()
        .update(previous=None, world_space=space, viewer_history=history)
        .belief
    )


def test_reference_model_and_queries_use_full_canonical_support() -> None:
    belief = generated_belief()

    assert query_mass(belief, WorldQuery.has("Lightning Bolt")) == pytest.approx(0.7)
    has_bolt = condition_belief(belief, WorldQuery.has("Lightning Bolt"))
    lacks_bolt = condition_belief(belief, WorldQuery.lacks("Lightning Bolt"))

    assert isinstance(has_bolt, BeliefState)
    assert isinstance(lacks_bolt, BeliefState)
    assert query_mass(has_bolt, WorldQuery.has("Lightning Bolt")) == 1.0
    assert query_mass(lacks_bolt, WorldQuery.has("Lightning Bolt")) == 0.0
    assert has_bolt.normalization_error < 1e-12
    assert lacks_bolt.normalization_error < 1e-12


def test_equivalent_queries_produce_identical_beliefs_and_encodings() -> None:
    belief = generated_belief()
    schema = fixture_schema(belief.space)
    has_bolt = condition_belief(belief, WorldQuery.has("Lightning Bolt"))
    not_zero_bolt = condition_belief(
        belief, WorldQuery.not_exactly("Lightning Bolt", 0)
    )

    assert isinstance(has_bolt, BeliefState)
    assert isinstance(not_zero_bolt, BeliefState)
    assert has_bolt.digest == not_zero_bolt.digest
    assert (
        encode_belief(has_bolt, schema).encoding_receipt
        == encode_belief(not_zero_bolt, schema).encoding_receipt
    )


def test_empty_support_and_invalid_distributions_fail_closed() -> None:
    belief = generated_belief()

    empty = condition_belief(belief, WorldQuery.has("Lightning Bolt", at_least=3))
    assert isinstance(empty, EmptyBeliefSupport)
    assert query_mass(belief, WorldQuery.has("Lightning Bolt", at_least=3)) == 0.0

    with pytest.raises(BeliefError, match="normalized"):
        BeliefState(
            belief.space,
            belief.model_id,
            np.zeros(belief.space.support_size, dtype=np.float64),
        )
    with pytest.raises(BeliefError, match="finite"):
        BeliefState.from_probabilities(
            belief.space,
            belief.model_id,
            np.full(belief.space.support_size, np.nan, dtype=np.float64),
        )


def test_lifecycle_and_search_import_one_canonical_belief_type() -> None:
    from manabot.belief.range import BeliefState as RangeBeliefState
    from manabot.belief.state import BeliefState as LifecycleBeliefState

    belief = generated_belief()

    assert BeliefState is RangeBeliefState is LifecycleBeliefState
    assert type(belief) is RangeBeliefState


def test_encoding_rejects_content_and_count_schema_mismatches() -> None:
    belief = generated_belief()
    schema = fixture_schema(belief.space)
    wrong_manifest = BeliefEncodingSchema(
        schema_identity=schema.schema_identity,
        world_schema_identity=schema.world_schema_identity,
        content_manifest_identity="another-content-manifest",
        rows=schema.rows,
        count_buckets=schema.count_buckets,
    )
    with pytest.raises(BeliefError, match="content manifest"):
        encode_belief(belief, wrong_manifest)

    missing_card = BeliefEncodingSchema(
        schema_identity=schema.schema_identity,
        world_schema_identity=schema.world_schema_identity,
        content_manifest_identity=schema.content_manifest_identity,
        rows=tuple(row for row in schema.rows if row.card_name != "Mountain"),
        count_buckets=schema.count_buckets,
    )
    with pytest.raises(BeliefError, match="content vocabulary"):
        encode_belief(belief, missing_card)

    shallow_counts = BeliefEncodingSchema(
        schema_identity=schema.schema_identity,
        world_schema_identity=schema.world_schema_identity,
        content_manifest_identity=schema.content_manifest_identity,
        rows=schema.rows,
        count_buckets=2,
    )
    with pytest.raises(BeliefError, match="exceeds"):
        encode_belief(belief, shallow_counts)


def test_repeated_card_identity_projects_distinct_hand_and_library_rows() -> None:
    belief = generated_belief()
    schema = fixture_schema(belief.space)
    has_bolt = condition_belief(belief, WorldQuery.has("Lightning Bolt"))
    lacks_bolt = condition_belief(belief, WorldQuery.lacks("Lightning Bolt"))

    assert isinstance(has_bolt, BeliefState)
    assert isinstance(lacks_bolt, BeliefState)
    has_view = encode_belief(has_bolt, schema)
    lacks_view = encode_belief(lacks_bolt, schema)
    bolt_rows = {
        row.hidden_zone_id: index
        for index, row in enumerate(schema.rows)
        if row.card_name == "Lightning Bolt"
    }
    assert set(bolt_rows) == {LIBRARY_ZONE_ID, HAND_ZONE_ID}
    hand = bolt_rows[HAND_ZONE_ID]
    library = bolt_rows[LIBRARY_ZONE_ID]
    assert has_view.card_def_ids[hand] == has_view.card_def_ids[library]
    assert has_view.count_probabilities[hand].tolist() == pytest.approx(
        [0.0, 6.0 / 7.0, 1.0 / 7.0]
    )
    assert has_view.count_probabilities[library].tolist() == pytest.approx(
        [1.0 / 7.0, 6.0 / 7.0, 0.0]
    )
    assert lacks_view.count_probabilities[hand].tolist() == [1.0, 0.0, 0.0]
    assert lacks_view.count_probabilities[library].tolist() == [0.0, 0.0, 1.0]


def test_schema_rejects_only_duplicate_full_row_keys() -> None:
    space = generated_belief().space
    repeated = BeliefRow(
        owner_role_id=OPPONENT_OWNER_ROLE_ID,
        hidden_zone_id=HAND_ZONE_ID,
        card_def_id=7,
        card_name="Lightning Bolt",
    )
    with pytest.raises(BeliefError, match="unique by owner"):
        BeliefEncodingSchema(
            schema_identity="test",
            world_schema_identity=space.world_schema_identity,
            content_manifest_identity=space.content_manifest_identity,
            rows=(repeated, repeated),
            count_buckets=3,
        )


def test_history_identity_is_derived_from_native_observations_and_receipts() -> None:
    initial = Observation(
        schema_version=SEMANTIC_DECISION_VERSION,
        revision=9,
        viewer=0,
        viewer_state_hash="viewer-nine",
        viewer_state={},
        events=("event-a",),
        decision=None,
    )
    same = ViewerHistory.from_observation(initial)
    assert same.identity == ViewerHistory.from_observation(initial).identity

    receipt = TransitionReceipt(
        schema_version=SEMANTIC_DECISION_VERSION,
        before_revision=9,
        after_revision=10,
        command_id="command-nine",
        public_commitment={"kind": "play_land", "card": "Mountain"},
        events=("event-b",),
        next_decision="next",
    )
    current = Observation(
        schema_version=SEMANTIC_DECISION_VERSION,
        revision=10,
        viewer=0,
        viewer_state_hash="viewer-ten",
        viewer_state={},
        events=(),
        decision=None,
    )
    advanced = same.advance(receipt, current, acting=1)

    assert advanced.events == ("event-a", "event-b")
    assert [event.to_payload() for event in advanced.semantic_events] == [
        {
            "actor_role_id": 1,
            "commitment": {"kind": "play_land", "card": "Mountain"},
        }
    ]
    assert advanced.identity != same.identity
    with pytest.raises(BeliefError, match="does not continue"):
        advanced.advance(receipt, current, acting=1)


@pytest.mark.parametrize(
    "public_commitment, message",
    (
        ({"kind": "declare_attacker"}, "unsupported public commitment"),
        ({"kind": "cast"}, "non-canonical fields"),
        (
            {"kind": "pass_priority", "card": "Mountain"},
            "non-canonical fields",
        ),
    ),
)
def test_history_rejects_unsupported_public_commitments(
    public_commitment: dict[str, str],
    message: str,
) -> None:
    history = fixture_history()
    receipt = TransitionReceipt(
        schema_version=SEMANTIC_DECISION_VERSION,
        before_revision=history.current_revision,
        after_revision=history.current_revision + 1,
        command_id="unsupported-commitment",
        public_commitment=public_commitment,
        events=(),
        next_decision="next",
    )
    observation = Observation(
        schema_version=SEMANTIC_DECISION_VERSION,
        revision=receipt.after_revision,
        viewer=history.viewer,
        viewer_state_hash="next-viewer-state",
        viewer_state={},
        events=(),
        decision=None,
    )

    with pytest.raises(BeliefError, match=message):
        history.advance(receipt, observation, acting=1)


def test_public_commitment_type_cannot_construct_invalid_semantics() -> None:
    with pytest.raises(SemanticContractError, match="canonical card name"):
        PublicCommitment(kind="cast")
    with pytest.raises(SemanticContractError, match="cannot carry a card name"):
        PublicCommitment(kind="pass_priority", card="Mountain")
