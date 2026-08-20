"""Behavioral checks for canonical belief and query operations."""

import numpy as np
import pytest

from manabot.belief import (
    BeliefEncodingSchema,
    BeliefError,
    BeliefRow,
    BeliefState,
    CompatibleDealBeliefModel,
    EmptyBeliefSupport,
    condition_belief,
    encode_belief,
    query_mass,
)
from manabot.belief.demo import retained_history, retained_schema, retained_space
from managym.possible_worlds import WorldQuery


def generated_belief() -> BeliefState:
    history = retained_history()
    space = retained_space(history)
    return CompatibleDealBeliefModel().update(
        previous=None, world_space=space, viewer_history=history
    ).belief


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
    schema = retained_schema(belief.space)
    has_bolt = condition_belief(belief, WorldQuery.has("Lightning Bolt"))
    not_zero_bolt = condition_belief(
        belief, WorldQuery.not_exactly("Lightning Bolt", 0)
    )

    assert isinstance(has_bolt, BeliefState)
    assert isinstance(not_zero_bolt, BeliefState)
    assert has_bolt.identity == not_zero_bolt.identity
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
            belief.model,
            np.full(belief.space.support_size, 0.5, dtype=np.float64),
        )
    with pytest.raises(BeliefError, match="finite"):
        BeliefState(
            belief.space,
            belief.model,
            np.full(belief.space.support_size, np.nan, dtype=np.float64),
        )


def test_encoding_rejects_content_and_count_schema_mismatches() -> None:
    belief = generated_belief()
    schema = retained_schema(belief.space)
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
        rows=tuple(
            BeliefRow(
                row.owner_role_id,
                row.hidden_zone_id,
                row.card_def_id,
                row.card_name,
            )
            for row in schema.rows
        ),
        count_buckets=2,
    )
    with pytest.raises(BeliefError, match="exceeds"):
        encode_belief(belief, shallow_counts)
