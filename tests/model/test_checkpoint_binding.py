"""Load an earlier world through each public file format and use its saved rows."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import BACKEND, load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace, encode_observation
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.catalog import inspect_checkpoint
from manabot.policy import buckets
from manabot.policy.binding import packed
from manabot.policy.checkpoint import read_container, write_container
from manabot.policy.readings import (
    fresh_readings,
    input_width,
    legacy_readings,
    parse_reading,
    read_numeric,
)
from manabot.recipes.builders import elemental_self_play, gray_ogre_self_play
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
        last = input_width(parse_reading(record["input_readings"][key].pop()))
    player = "player_embedding.projection.0.weight"
    state[player] = state[player][:, :-last].contiguous()
    del record["architecture"]
    if legacy_file:
        torch.save(record, path)
    else:
        write_checkpoint(record, path)
    assert inspect_checkpoint(path, backend=BACKEND).compatible
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
        for key, value in encode_observation(
            space, env.observation_for_player(0)
        ).items()
    }
    batch["agent_player"][..., -1] = 1
    prepared = loaded.prepare(batch)
    np.testing.assert_array_equal(
        prepared.agent_player,
        read_numeric(batch["agent_player"][..., :-1], loaded.readings["agent_player"]),
    )
    assert np.isfinite(loaded.evaluate(prepared)[0]).all()

    if not legacy_file:
        carried = tmp_path / "carried.safetensors"
        write_container(carried, read_container(path))
        restored, _ = load_checkpoint(str(carried))
        np.testing.assert_array_equal(
            restored.evaluate_probabilities(restored.prepare(batch))[0],
            loaded.evaluate_probabilities(prepared)[0],
        )

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
    batch = {
        key: value[None]
        for key, value in encode_observation(space, observation).items()
    }
    expected = policy.evaluate(policy.prepare(batch))

    # Emulate a later engine with appended columns and an unknown card digest.
    # Without the embedded writer catalog the original file cannot bind here.
    native_layout = managym.ObservationLayout
    growing = (
        "agent_cards",
        "opponent_cards",
        "stack",
        "actions",
        "decklists",
        "events",
    )

    class ExtendedLayout:
        def __init__(self, **options: Any) -> None:
            self._inner = native_layout(**options)

        @property
        def arrays(self) -> dict[str, Any]:
            rows = self._inner.arrays
            for key in growing:
                columns = rows[key]["columns"]
                columns["appended"] = {"index": len(columns), "kind": "flag"}
                rows[key]["shape"] = [*rows[key]["shape"][:-1], len(columns)]
            return rows

        @property
        def shapes(self) -> dict[str, tuple[int, ...]]:
            shapes = dict(self._inner.shapes)
            for key in growing:
                shapes[key] = (*shapes[key][:-1], shapes[key][-1] + 1)
            return shapes

        @property
        def dtypes(self) -> dict[str, str]:
            return self._inner.dtypes

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    # The space derives its shapes from the layout, so it grows with it.
    monkeypatch.setattr(managym, "ObservationLayout", ExtendedLayout)
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
    rewritten = read_checkpoint(rebound)
    assert rewritten["input_columns"] == record["input_columns"]
    assert rewritten["input_readings"] == record["input_readings"]
    again, _ = load_checkpoint(str(rebound))
    np.testing.assert_array_equal(again.evaluate(again.prepare(batch))[0], expected[0])

    record["input_columns"]["agent_cards"][0] = "lost_column"
    write_checkpoint(record, path)
    with pytest.raises(ValueError, match="agent_cards columns:.*lost_column"):
        load_checkpoint(str(path))


@pytest.mark.parametrize("elemental", [False, True])
@pytest.mark.parametrize("recorded", [False, True])
def test_setup_policy_keeps_its_pre_x_action_scores(
    tmp_path: Path, elemental: bool, recorded: bool
) -> None:
    settings = elemental_self_play() if elemental else gray_ogre_self_play(horizon=2)
    space = ObservationSpace(settings.observation)
    readings = (fresh_readings if recorded else legacy_readings)(space)
    readings["actions"] = readings["actions"][:23]
    writer = ActorCritic(space, settings.model, readings=readings).eval()
    players = Match(settings.match).to_rust()
    path = tmp_path / "earlier.safetensors"
    save_checkpoint(writer, space, path, player_configs=players)
    if not recorded:
        record = read_checkpoint(path)
        del record["input_readings"]
        write_checkpoint(record, path)
    loaded, runtime = load_checkpoint(str(path))
    env = managym.Env(seed=17)
    obs, _ = env.reset(players)
    batch = {
        key: value[None] for key, value in encode_observation(runtime, obs).items()
    }
    expected = writer.evaluate(writer.prepare(batch))
    batch["actions"][..., 23:] = 1
    actual = loaded.evaluate(loaded.prepare(batch))
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(before, after)
