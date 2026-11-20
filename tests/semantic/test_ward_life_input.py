"""A ward paid in life is visible as life: columns, fact program and text."""

import numpy as np

from manabot.backends.torch._semantic_cards import SemanticCardEncoder
from manabot.env.observation import (
    FACT_DIM,
    FACT_EFFECT_PROGRAM,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_VALUE_INDEX,
    PERMANENT_DIM,
    WARD,
    ObservationSpace,
    encode_observation,
)
from manabot.infra.config import ObservationConfig
from manabot.policy import admission
from manabot.semantic.text_observation import capture_text_decision
from manabot.verify.text_observation import priority_root
import managym
from tests.model.prepared_routes import record_vocabulary


def test_sire_of_seven_deaths_ward_is_life_in_columns_facts_json_and_text() -> None:
    env = priority_root(("Forest",), ("Sire of Seven Deaths", "Forest"))
    owner = env.current_agent_index()
    assert owner is not None
    token = managym.catalog_token_value(WARD)
    config = ObservationConfig()
    space = ObservationSpace(config)
    columns = managym.observation_layout()["agent_permanents"]["columns"]
    has_ward = columns["has_ward"]["index"]
    ward_cost = columns["ward_cost"]["index"]
    vocabulary = record_vocabulary(SemanticCardEncoder("world", 4, records=True))
    for viewer in (owner, 1 - owner):
        obs = env.observation_for_player(viewer)
        permanents = (
            obs.agent_permanents if viewer == owner else obs.opponent_permanents
        )
        sire = next(
            permanent.characteristics
            for permanent in permanents
            if permanent.characteristics.name == "Sire of Seven Deaths"
        )
        # No mana value states this ward; the life price is its own fact.
        assert (sire.ward_cost, sire.ward_life) == (0, 7)
        program = sire.ward_program
        assert program[1] == (4, 7), "the amount is an integer token"

        expected = encode_observation(space, obs)

        rows = np.concatenate(
            [expected["agent_permanents"], expected["opponent_permanents"]]
        ).reshape(-1, PERMANENT_DIM)
        warded = rows[rows[:, has_ward] == 1.0]
        assert len(warded) == 1, "the Sire has ward"
        assert warded[0, ward_cost] == 0.0, "at no mana value"

        facts = expected["facts"].reshape(-1, FACT_DIM)
        ward = facts[
            (facts[:, FACT_KIND_INDEX] == FACT_EFFECT_PROGRAM)
            & (facts[:, FACT_ID_INDEX] == token >> 16)
            & (facts[:, FACT_ID_INDEX + 1] == token & 0xFFFF)
        ]
        np.testing.assert_array_equal(
            ward[:, FACT_VALUE_INDEX : FACT_VALUE_INDEX + 2],
            [[float(kind), ordinal] for ordinal, (kind, _) in enumerate(program)],
        )

        # The policy admits the rows: "Ward" and the program's tokens are
        # in the record vocabulary.
        admission.facts(expected["facts"][None], config.max_facts, 10_000, vocabulary)

        # The JSON boundary keeps the price.
        restored = managym.Observation.from_json(obs.toJSON())
        restored_permanents = (
            restored.agent_permanents
            if viewer == owner
            else restored.opponent_permanents
        )
        assert any(
            permanent.characteristics.ward_life == 7
            and permanent.characteristics.ward_program == program
            for permanent in restored_permanents
        )

    assert "ward: pay 7 life" in capture_text_decision(env).text
