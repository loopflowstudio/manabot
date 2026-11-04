"""Complete name support, native encoding parity and learned categorical scoring."""

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manabot.env.observation import ACTION_MODE_INDEX, ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.inference import PolicyInference
from manabot.model.world import deck_world
from manabot.semantic.text_observation import capture_text_decision
import managym


@dataclass(frozen=True)
class NamingTrace:
    root: managym.Env
    actions: tuple[int, ...]


def _players() -> list[managym.PlayerConfig]:
    return [
        managym.PlayerConfig(
            "Hero", {"Swamp": 24, "The Rise of Sozin": 12}, disclosure="closed"
        ),
        managym.PlayerConfig(
            "Villain", {"Plains": 20, "Gray Ogre": 20}, disclosure="closed"
        ),
    ]


@pytest.fixture(scope="module")
def naming_trace() -> NamingTrace:
    env = managym.Env(seed=165)
    env.reset(_players())
    actions: list[int] = []
    cast = False
    for ordinal in range(200):
        frame = json.loads(env.semantic_decision_frame_json())
        if any(offer["label"] == "Name Gray Ogre" for offer in frame["offers"]):
            return NamingTrace(env, tuple(actions))
        labels = ["Target Villain", "Play Swamp", "Play Plains"]
        if not cast:
            labels.append("Cast The Rise of Sozin")
        labels.extend(["Pass priority", "Decline"])
        offer = next(
            (
                offer
                for label in labels
                for offer in frame["offers"]
                if offer["label"] == label
            ),
            frame["offers"][0],
        )
        cast |= offer["label"] == "Cast The Rise of Sozin"
        actions.append(frame["offers"].index(offer))
        env.execute_semantic_command_json(
            json.dumps(
                {
                    "command_id": f"naming.{ordinal}",
                    "expected_revision": frame["revision"],
                    "offer_id": offer["id"],
                    "answers": [],
                }
            )
        )
    pytest.fail("naming decision was not reached")


@pytest.fixture(scope="module")
def naming_root(naming_trace: NamingTrace) -> managym.Env:
    return naming_trace.root


def _tensor(values: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value)[None] for key, value in values.items()}


def test_catalog_naming_v8_scalar_parity_and_private_text(
    naming_root: managym.Env,
) -> None:
    raw = naming_root.observation_for_player(naming_root.current_agent_index())
    space = ObservationSpace(ObservationConfig(policy_input_version=8))
    reference = space.encode(raw)
    native = naming_root.encode_observation(raw, policy_input_version=8)
    into = space.encoder.allocate()
    naming_root.encode_observation_into(raw, into)
    for key in reference:
        np.testing.assert_allclose(native[key], reference[key], atol=1e-6, err_msg=key)
        np.testing.assert_array_equal(into[key], native[key], err_msg=key)
    catalog = managym.naming_catalog()
    assert len(raw.action_space.actions) == len(catalog)
    assert not reference["actions"][:, ACTION_MODE_INDEX].any()
    for index, candidate in enumerate(catalog):
        definition = candidate.definition
        assert reference["name_candidates"][index].tolist() == [
            index + 1,
            definition + 1 if definition is not None else 0,
        ]
    assert not reference["decklists"][1].any()
    assert not raw.opponent.main_counts
    text = capture_text_decision(naming_root).text
    assert "Fire Lord Sozin" in text and "Gray Ogre" in text
    assert not naming_root.observation_for_player(1).action_space.actions
    # Frozen layouts are still usable within their declared capacity.
    for version in (6, 7):
        legacy = naming_root.encode_observation(raw, policy_input_version=version)
        assert "name_candidates" not in legacy
        np.testing.assert_array_equal(
            legacy["actions"][: len(catalog), ACTION_MODE_INDEX],
            np.arange(1, len(catalog) + 1),
        )
    assert ObservationConfig.from_checkpoint({}).policy_input_version == 1
    assert "max_facts" not in ObservationConfig.from_checkpoint({}).model_dump()


def test_catalog_naming_large_support_learns_and_rejects_wrong_binding(
    naming_root: managym.Env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(243)
    fixture = json.loads(
        Path("managym/tests/fixtures/naming_vocabulary.json").read_text()
    )
    entries: dict[str, int | None] = {entry["name"]: None for entry in fixture["names"]}
    entries.update(
        {candidate.name: candidate.definition for candidate in managym.naming_catalog()}
    )
    catalog = [
        managym.NameCandidate(name, definition)
        for name, definition in sorted(entries.items())
    ]
    assert len(catalog) > 600
    monkeypatch.setattr(managym, "naming_catalog", lambda: catalog)
    space = ObservationSpace(ObservationConfig(policy_input_version=8))
    assert space.shapes["actions"][0] == len(catalog)
    # Native deserialization and scalar encoding exercise the same controlled
    # vocabulary as flow/naming_tests.rs; no fake CardDefinition is installed.
    raw = naming_root.observation_for_player(0)
    payload = json.loads(raw.toJSON())
    template = payload["action_space"]["actions"][0]
    template.pop("name_definition", None)
    payload["action_space"]["actions"] = [
        template
        | {
            "description": f"Name {candidate.name}",
            "mode_index": index,
            "name_index": index,
            **(
                {"name_definition": candidate.definition}
                if candidate.definition is not None
                else {}
            ),
        }
        for index, candidate in enumerate(catalog)
    ]
    raw = managym.Observation.from_json(json.dumps(payload))
    values = space.encoder.allocate()
    naming_root.encode_observation_into(raw, values)
    for key, expected in space.encode(raw).items():
        np.testing.assert_allclose(values[key], expected, atol=1e-6, err_msg=key)
    with pytest.raises(ValueError, match="capacity"):
        ObservationSpace(ObservationConfig(policy_input_version=7)).encode(raw)
    model = ActorCritic(space, ModelConfig(hidden_dim=16, attention_on=False))
    tensors = _tensor(values)
    logits, _ = model(tensors)
    assert logits.shape == (1, len(catalog))
    assert (logits.softmax(-1) > 0).all()
    assert model.name_candidates is not None
    embedded = model.name_candidates(tensors["name_candidates"])
    assert not torch.equal(embedded[:, 511], embedded[:, 512])
    # One supervised gradient witness through the real policy head. This is a
    # bounded unit check, not a game/training experiment or strength claim.
    target = torch.tensor([512])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    before = torch.nn.functional.cross_entropy(logits, target)
    before.backward()
    assert model.name_candidates.embedding.weight.grad is not None
    assert model.name_candidates.embedding.weight.grad[513].abs().sum() > 0
    optimizer.step()
    after, _ = model(tensors)
    assert torch.nn.functional.cross_entropy(after, target) < before.detach()
    # Portable model state preserves candidate preferences exactly.
    restored = ActorCritic(space, ModelConfig(hidden_dim=16, attention_on=False))
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(tensors)[0], after)
    bad = {key: value.clone() for key, value in tensors.items()}
    bad["name_candidates"][0, 512, 1] = managym.CARD_DEFINITION_COUNT + 100
    with pytest.raises(ValueError, match="vocabulary"):
        model(bad)
    assert "naming_vocabulary" in deck_world(space)
    assert "naming_vocabulary" not in deck_world(
        ObservationSpace(ObservationConfig(policy_input_version=7))
    )


def test_catalog_naming_programs_affect_scores(
    naming_root: managym.Env,
) -> None:
    torch.set_num_threads(1)
    space = ObservationSpace(ObservationConfig(policy_input_version=8))
    model = ActorCritic(
        space, ModelConfig(hidden_dim=8, attention_on=False, world=managym.WORLD_SETUP)
    )
    raw = naming_root.observation_for_player(0)
    tensors = _tensor(space.encode(raw))
    assert model.name_candidates is not None
    with torch.no_grad():
        model.name_candidates.embedding.weight.zero_()
    started = perf_counter()
    logits, _ = model(tensors)
    score_seconds = perf_counter() - started
    # These rows have identical action features/focus and zero name-identity
    # vectors. Different scores must therefore read their different programs.
    assert logits[0, : len(managym.naming_catalog())].unique().numel() > 1
    index = next(
        i
        for i, candidate in enumerate(managym.naming_catalog())
        if candidate.name == "Fire Lord Sozin"
    )
    torch.nn.functional.cross_entropy(logits, torch.tensor([index])).backward()
    assert model.semantic_cards is not None
    assert model.semantic_cards.token_embedding.weight.grad is not None
    assert model.semantic_cards.token_embedding.weight.grad.abs().sum() > 0
    print(
        f"naming scoring: {len(raw.action_space.actions)} candidates, {score_seconds:.6f}s (one CPU forward; no speed threshold)"
    )


@pytest.mark.parametrize("naming", [False, True])
def test_naming_shares_program_encoding_and_preserves_padded_action_order(
    naming_root: managym.Env, naming: bool
) -> None:
    torch.set_num_threads(1)
    env = naming_root.clone_env()
    if not naming:
        env.step(0)  # The following private search has no naming candidates.
    space = ObservationSpace(ObservationConfig(policy_input_version=8))
    tensors = _tensor(space.encode(env.observation_for_player(0)))
    model = ActorCritic(
        space, ModelConfig(hidden_dim=8, attention_on=False, world=managym.WORLD_SETUP)
    ).eval()
    semantic = model.semantic_cards
    assert semantic is not None
    with patch.object(semantic, "definitions", wraps=semantic.definitions) as encode:
        model(tensors)
        assert encode.call_count == 1
        rows = encode.call_args.args[0]
        if not naming:
            assert len(rows) < semantic.catalog_shape.definitions

    with torch.no_grad():
        expected = model(tensors)
        with (
            patch.object(semantic, "definitions", wraps=semantic.definitions) as encode,
            PolicyInference(model) as scope,
        ):
            prepared = scope.prepare(tensors)
            for _ in range(2):
                actual = model(prepared)
                for left, right in zip(actual, expected, strict=True):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
            assert encode.call_count == 1

        # Move a legal row to the final slot and padding into the middle. Trimming
        # must preserve action identity even when validity is not a dense prefix.
        order = torch.arange(tensors["actions"].shape[1])
        order[0], order[-1] = order[-1].item(), order[0].item()
        permuted = dict(tensors)
        for key in ("actions", "action_focus", "actions_valid", "name_candidates"):
            permuted[key] = tensors[key][:, order]
        logits, value = model(permuted)
        torch.testing.assert_close(logits, expected[0][:, order])
        torch.testing.assert_close(value, expected[1])


def test_catalog_naming_vector_buffers_match_scalar_and_clear_after_choice(
    naming_trace: NamingTrace,
) -> None:
    space = ObservationSpace(ObservationConfig(policy_input_version=8))
    vector = managym.VectorEnv(1, seed=165)
    scalar = managym.Env(seed=165)
    buffers = space.encoder.allocate(1)
    buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    vector.set_buffers(buffers)
    vector.reset_all_into_buffers(_players())
    raw = scalar.reset(_players())[0]
    # Replay one Command trace through both step APIs so their per-step events
    # have the same lifetime; observation_for_player omits already-drained events.
    for index in naming_trace.actions:
        raw = scalar.step(index)[0]
        vector.step_into_buffers([index])
    assert buffers["name_candidates"][0, :, 0].max() == len(managym.naming_catalog())
    for key, expected in space.encode(raw).items():
        np.testing.assert_allclose(buffers[key][0], expected, atol=1e-6, err_msg=key)
    index = next(
        i
        for i, action in enumerate(raw.action_space.actions)
        if action.description == "Name Gray Ogre"
    )
    raw = scalar.step(index)[0]
    vector.step_into_buffers([index])
    assert not buffers["name_candidates"].any()
    for key, expected in space.encode(raw).items():
        np.testing.assert_allclose(buffers[key][0], expected, atol=1e-6, err_msg=key)
