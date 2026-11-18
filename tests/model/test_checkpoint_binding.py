"""Load an earlier world through each public file format and use its saved rows."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import BACKEND, load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.catalog import checkpoint_incompatibility, inspect_checkpoint
from manabot.policy import buckets
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
    if legacy_file:
        torch.save(record, path)
    else:
        write_checkpoint(record, path)
    with pytest.raises(
        ValueError, match=f"unexplained record_token key {keys[0].item()}"
    ) as refusal:
        load_checkpoint(str(path))
    assert inspect_checkpoint(path, backend=BACKEND).incompatibility == str(
        refusal.value
    )


def test_saved_policy_uses_its_embedded_catalog_and_column_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    space = ObservationSpace(ObservationConfig(policy_history_version=1))
    policy = ActorCritic(
        space, ModelConfig(hidden_dim=8, world="world", recent_events=True)
    ).eval()
    path = tmp_path / "world.safetensors"
    # The writing engine has a digest for which no external catalog exists.
    identity = managym.world_identity
    pool = managym.content_pool
    with monkeypatch.context() as writer:
        writer.setattr(
            managym, "world_identity", lambda: {**identity(), "card_digest": "f" * 64}
        )
        writer.setattr(
            managym,
            "content_pool",
            lambda key: {**pool(key), "content_digest": "f" * 64},
        )
        save_checkpoint(policy, space, path, player_configs=None)
    record = read_checkpoint(path)
    assert record["source"]["commit"]
    assert record["writer_catalog"]["definition_keys"]
    assert record["writer_catalog"]["naming_catalog"][0]["name"]
    env = managym.Env(seed=7)
    observation, _ = env.reset(
        [
            managym.PlayerConfig("A", {"Mountain": 20, "Gray Ogre": 20}),
            managym.PlayerConfig("B", {"Forest": 20, "Gray Ogre": 20}),
        ]
    )
    batch = {key: value[None] for key, value in space.encode(observation).items()}
    expected = policy.evaluate(policy.prepare(batch))

    # Emulate a later engine with appended columns and an unknown card digest.
    # Without the embedded writer catalog the original file cannot bind here.
    native_layout = managym.observation_layout
    growing = (
        "agent_cards",
        "opponent_cards",
        "stack",
        "actions",
        "decklists",
        "events",
    )

    def extended_layout(**options: bool) -> dict[str, Any]:
        rows = native_layout(**options)
        for key in growing:
            columns = rows[key]["columns"]
            columns["appended"] = {"index": len(columns)}
        return rows

    initialize = ObservationSpace.__init__

    def extended_space(
        self: ObservationSpace, config: ObservationConfig | None = None
    ) -> None:
        initialize(self, config)
        self.encoder.card_dim += 1
        self.encoder.action_dim += 1
        for key in growing:
            shape = self.shapes[key]
            self.shapes[key] = (*shape[:-1], shape[-1] + 1)

    monkeypatch.setattr(managym, "observation_layout", extended_layout)
    monkeypatch.setattr(ObservationSpace, "__init__", extended_space)
    monkeypatch.setattr(buckets, "_EVENT_WIDTH", space.shapes["events"][-1] + 1)
    loaded, runtime = load_checkpoint(str(path))
    for key in growing:
        batch[key] = np.concatenate(
            (batch[key], np.full_like(batch[key][..., :1], 999)), axis=-1
        )
    actual = loaded.evaluate(loaded.prepare(batch))
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(before, after)
    rebound = tmp_path / "rebound.safetensors"
    save_checkpoint(loaded, runtime, rebound, player_configs=None)
    assert read_checkpoint(rebound)["input_columns"] == record["input_columns"]
    again, _ = load_checkpoint(str(rebound))
    np.testing.assert_array_equal(again.evaluate(again.prepare(batch))[0], expected[0])

    record["input_columns"]["agent_cards"][0] = "lost_column"
    write_checkpoint(record, path)
    with pytest.raises(ValueError, match="agent_cards columns:.*lost_column"):
        load_checkpoint(str(path))
