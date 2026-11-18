"""Load an earlier world through each public file format and use its saved rows."""

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.catalog import checkpoint_incompatibility
from manabot.policy.binding import packed
import managym
from tests.model.checkpoint_fixture import read_checkpoint, write_checkpoint


@pytest.mark.parametrize("legacy_file", [False, True])
def test_writer_programs_names_and_player_columns_survive_catalog_growth(
    tmp_path: Path, legacy_file: bool
) -> None:
    space = ObservationSpace(ObservationConfig(policy_input_version=9))
    agent = ActorCritic(space, ModelConfig(hidden_dim=8, world="world"))
    path = tmp_path / "pilot"
    save_checkpoint(agent, space, path, player_configs=None)
    record = read_checkpoint(path)
    # Construct a writer with the definition rows reversed and one fewer name.
    # References in the writer's learned token table are its own definition ids.
    writer = record["writer_catalog"]
    writer["definition_keys"].reverse()
    count = len(writer["definition_keys"])
    writer["naming_catalog"].pop()
    for candidate in writer["naming_catalog"]:
        if candidate["definition"] is not None:
            candidate["definition"] = count - 1 - candidate["definition"]
    state = record["model_state_dict"]
    token_keys = state["semantic_cards.record_token_keys"]
    refs = token_keys >> 32 == 7
    runtime_ref_keys = token_keys[refs].clone()
    original_ref_rows = state["semantic_cards.record_token_rows"][refs]
    token_keys[refs] = (7 << 32) + count - 1 - (runtime_ref_keys & 0xFFFFFFFF)
    # One renamed field, one retired variant and one new runtime token.
    field = packed(2, managym.catalog_token_value("picks"))
    token_keys[token_keys == field] = packed(
        2, managym.catalog_token_value("opponent_picks")
    )
    token_rows = state["semantic_cards.record_token_rows"]
    record_rows = set(state["semantic_cards.record_name_rows"].tolist())
    slot = next(
        i
        for i, (key, row) in enumerate(
            zip(token_keys.tolist(), token_rows.tolist(), strict=True)
        )
        if key >> 32 == 3 and row not in record_rows
    )
    new_key = int(token_keys[slot])
    retired_row = int(token_rows[slot])
    token_keys[slot] = packed(
        3, managym.catalog_token_value("TriggerCondition::YouCastSpell")
    )
    order = token_keys.argsort()
    state["semantic_cards.record_token_keys"] = token_keys[order]
    state["semantic_cards.record_token_rows"] = token_rows[order]
    state["semantic_cards.tokens"] = state["semantic_cards.tokens"].flip((0,))
    state["semantic_cards.lengths"] = state["semantic_cards.lengths"].flip((0,))
    state["semantic_cards.tokens"][-1, 0] = retired_row
    weight = state["semantic_cards.token_embedding.weight"]
    expected = weight[
        state["semantic_cards.tokens"][-1, : state["semantic_cards.lengths"][-1]]
    ]
    name_param = "name_candidates.embedding.weight"
    state[name_param] = state[name_param][:-1].clone()
    binding = record["world_binding"]
    binding["world"] = "previous"
    binding["world_digest"] = writer["world_digest"] = "0" * 64
    binding["naming_vocabulary"] = canonical_sha256(writer["naming_catalog"])
    binding["rules"]["possible_world_version"] -= 1
    for key in ("agent_player", "opponent_player"):
        binding["input_schema"][key]["shape"][-1] -= 1
        record["input_columns"][key].pop()
    player = "player_embedding.projection.0.weight"
    state[player] = state[player][:, :-1].contiguous()
    del record["architecture"]
    assert checkpoint_incompatibility(record) is None
    if legacy_file:
        torch.save(record, path)
    else:
        write_checkpoint(record, path)
    loaded, _ = load_checkpoint(str(path))
    cards = loaded.semantic_cards
    assert cards is not None
    runtime_ref_rows = cards.record_token_rows[
        torch.searchsorted(cards.record_token_keys, runtime_ref_keys)
    ]
    torch.testing.assert_close(
        cards.token_embedding(runtime_ref_rows),
        weight[original_ref_rows],
        atol=0,
        rtol=0,
    )
    actual = cards.token_embedding(cards.tokens[0, : cards.lengths[0]])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    row = cards.record_token_rows[cards.record_token_keys == new_key]
    torch.testing.assert_close(
        cards.token_embedding.weight[row.item()], weight[1:].mean(0)
    )
    torch.testing.assert_close(
        loaded.name_candidates.embedding.weight[-1], state[name_param][1:].mean(0)
    )
    torch.testing.assert_close(
        loaded.player_embedding.projection[0].weight, state[player]
    )

    env = managym.Env(seed=17)
    env.reset(
        [
            managym.PlayerConfig("A", {"Forest": 20, "Gray Ogre": 20}),
            managym.PlayerConfig("B", {"Mountain": 20, "Gray Ogre": 20}),
        ]
    )
    batch = {
        key: value[None]
        for key, value in space.encode(env.observation_for_player(0)).items()
    }
    batch["agent_player"][..., -1] = 1
    prepared = loaded.prepare(batch)
    np.testing.assert_array_equal(
        prepared.agent_player, batch["agent_player"][..., :-1]
    )
    assert np.isfinite(loaded.evaluate(prepared)[0]).all()

    keys = state["semantic_cards.record_token_keys"]
    keys[0] = packed(3, 12345)
    write_checkpoint(record, path)
    with pytest.raises(
        ValueError, match=f"unexplained record_token key {keys[0].item()}"
    ):
        load_checkpoint(str(path))
