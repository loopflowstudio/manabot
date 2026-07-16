from __future__ import annotations

import numpy as np
import pytest

from manabot.env.observation import ObservationSpace
from manabot.semantic.learning import BoundSemanticPack
from manabot.verify.util import GW_ALLIES_DECK, UR_LESSONS_DECK
import managym


def _engine() -> managym.Env:
    engine = managym.Env(seed=215, skip_trivial=True)
    engine.reset(
        [
            managym.PlayerConfig("gw", dict(GW_ALLIES_DECK)),
            managym.PlayerConfig("ur", dict(UR_LESSONS_DECK)),
        ]
    )
    return engine


def _projection_bytes(projection) -> tuple[bytes, ...]:
    return (
        projection.object_definition_rows.tobytes(),
        projection.object_roles.tobytes(),
        projection.object_slots.tobytes(),
        projection.opaque_identity_ids.tobytes(),
        projection.opaque_identity_valid.tobytes(),
    )


def test_exact_environment_manifest_binds_the_selected_matchup():
    absent = managym.Env(seed=1)
    with pytest.raises(managym.AgentError, match="before reset"):
        absent.content_pack_manifest()

    engine = _engine()
    manifest = engine.content_pack_manifest()
    pack = BoundSemanticPack.from_env(engine)

    assert manifest["schema_version"] == pack.content_pack_schema_version
    assert manifest["content_digest"] == pack.content_pack_hash
    assert len(manifest["definitions"]) > len(pack.ir.definitions)
    assert len(pack.definition_row_by_card_def_id) == len(pack.ir.definitions)


def test_actual_viewer_projection_hides_determinized_private_cards_and_binds_references():
    engine = _engine()
    engine.scenario_clear_hand(0)
    engine.scenario_force_card_in_hand(0, "Invasion Reinforcements")
    visible = engine.scenario_refresh()
    pack = BoundSemanticPack.from_env(engine)

    projection = pack.project_observation(visible, identity_mode="semantic_only")
    assert not projection.opaque_identity_valid.any()
    assert (projection.opaque_identity_ids == -1).all()
    assert any(card.name == "Invasion Reinforcements" for card in visible.agent_cards)
    assert all(
        card.name != "Invasion Reinforcements" for card in visible.opponent_cards
    )

    program_row = next(
        row
        for row, program in enumerate(pack.ir.programs)
        if program["semantic_key"] == "tla.invasion_reinforcements.create_ally"
    )
    ally_row = next(
        row
        for row, definition in enumerate(pack.ir.definitions)
        if definition["semantic_key"] == "token.ally"
    )
    padded_catalog = pack.pad_catalog()
    targets = padded_catalog["definition_ref_target_rows"][program_row]
    mask = padded_catalog["definition_ref_mask"][program_row]
    assert targets[mask].tolist() == [ally_row]

    perspective = int(visible.agent.player_index)
    hidden_variants = []
    for seed in (215_001, 215_002):
        clone = engine.clone_env()
        clone.determinize(seed, perspective=perspective)
        hidden_obs = clone.scenario_refresh()
        hidden_variants.append(
            pack.project_observation(hidden_obs, identity_mode="semantic_only")
        )
    assert _projection_bytes(hidden_variants[0]) == _projection_bytes(projection)
    assert _projection_bytes(hidden_variants[1]) == _projection_bytes(projection)

    encoded = ObservationSpace().encode(visible)
    assert set(encoded) == set(ObservationSpace().shapes)
    assert all(
        encoded[key].shape == shape for key, shape in ObservationSpace().shapes.items()
    )
    assert np.asarray(encoded["agent_cards_valid"]).sum() >= 1
