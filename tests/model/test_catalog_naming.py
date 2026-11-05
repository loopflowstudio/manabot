"""Complete name support, native encoding parity and learned categorical scoring."""

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from naming_reference import DenseActionReference
import numpy as np
import pytest
import torch

from manabot.env.observation import ACTION_MODE_INDEX, ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.inference import PolicyInference
from manabot.model.world import deck_world
from manabot.semantic.text_observation import capture_text_decision
from manabot.sim.action.structured import flatten_projection
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
    naming_trace: NamingTrace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(243)
    root = naming_trace.root.clone_env()
    installed = managym.naming_catalog()
    catalog = root.scenario_naming_vocabulary()
    assert len(catalog) > 600
    assert len(naming_trace.root.observation_for_player(0).action_space.actions) == len(
        installed
    )
    assert [c.name for c in managym.naming_catalog()] == [c.name for c in installed]
    definition_free = next(c.name for c in catalog if c.definition is None)
    assert managym.unsupported_cards([definition_free]) == [definition_free]
    monkeypatch.setattr(managym, "naming_catalog", lambda: catalog)
    space = ObservationSpace(ObservationConfig(policy_input_version=8))
    assert space.shapes["actions"][0] == len(catalog)
    raw = root.observation_for_player(0)
    values = space.encoder.allocate()
    root.encode_observation_into(raw, values)
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
    selected, _, _, _ = model.get_action_and_value(tensors, deterministic=True)
    index = int(selected)
    assert index >= 512
    # The policy's actual choice executes against the same native vocabulary,
    # through vector buffers and semantic Commands, not fabricated action rows.
    vector = managym.VectorEnv(num_envs=1, seed=165)
    vector.reset_all(_players())
    for action in naming_trace.actions:
        vector.step([action])
    buffers = space.encoder.allocate(1)
    buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    vector.set_buffers(buffers)
    vector.scenario_naming_vocabulary_into_buffers()
    for key, expected in values.items():
        np.testing.assert_array_equal(buffers[key][0], expected, err_msg=key)
    frame = json.loads(root.semantic_decision_frame_json())
    assert json.loads(vector.semantic_decision_frames_json()[0]) == frame
    root_digest = root.state_digest()
    back = next(
        i for i, candidate in enumerate(catalog) if candidate.name == "Fire Lord Sozin"
    )
    ogre = next(
        i for i, candidate in enumerate(catalog) if candidate.name == "Gray Ogre"
    )
    for choice in dict.fromkeys([0, 511, 512, back, ogre, index]):
        branch = root.clone_env()
        replay = root.clone_env()
        command = json.dumps(
            {
                "command_id": f"large-name.{choice}",
                "expected_revision": frame["revision"],
                "offer_id": frame["offers"][choice]["id"],
                "answers": [],
            }
        )
        branch.step_semantic_command(command)
        replay.step_semantic_command(command)
        assert branch.state_digest() == replay.state_digest()
        pending = branch.observation_for_player(0)
        searching = any(f.decision == 25 and f.mode == 4 for f in pending.continuation)
        if searching:
            assert not branch.observation_for_player(1).action_space.actions
        assert not pending.opponent.main_counts
        with pytest.raises(Exception, match="revision|stale"):
            branch.step_semantic_command(command)
        assert branch.state_digest() == replay.state_digest()
        offers = json.loads(branch.semantic_decision_frame_json())["offers"]
        if choice == ogre:
            assert searching
            assert len(offers) > 5
            assert all(
                "Gray Ogre" in o["label"] or o["label"] == "Stop searching and shuffle"
                for o in offers
            )
        if choice == index:
            vector.step_into_buffers([choice])
            assert json.loads(vector.semantic_decision_frames_json()[0]) == json.loads(
                branch.semantic_decision_frame_json()
            )
            for key, expected in space.encode(pending).items():
                np.testing.assert_array_equal(buffers[key][0], expected, err_msg=key)
        if searching:
            stop = next(
                i
                for i, offer in enumerate(offers)
                if offer["label"] == "Stop searching and shuffle"
            )
            branch.step(stop)
        else:
            # No matching cards leaves only fail-to-find, skipped natively.
            assert all(
                offer["label"] != "Stop searching and shuffle" for offer in offers
            )
    assert root.state_digest() == root_digest
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


@pytest.mark.parametrize("change", ["name", "order", "definition", "representation"])
def test_naming_checkpoint_admits_exact_vocabulary_before_loading_weights(
    naming_root: managym.Env,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    space = ObservationSpace(ObservationConfig(policy_input_version=8))
    model = ActorCritic(
        space, ModelConfig(hidden_dim=8, attention_on=False, world=managym.WORLD_SETUP)
    ).eval()
    tensors = _tensor(space.encode(naming_root.observation_for_player(0)))
    path = tmp_path / "naming.pt"
    save_checkpoint(model, space, path, player_configs=None)
    restored, restored_space = load_checkpoint(str(path))
    assert restored_space.shapes == space.shapes
    with torch.no_grad():
        for actual, expected in zip(restored(tensors), model(tensors), strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    # Keep tensor shapes identical: admission must bind identities and joins,
    # not merely the number of candidate embedding rows.
    candidates = list(managym.naming_catalog())
    first = candidates[0]
    if change == "name":
        candidates[0] = managym.NameCandidate("Changed name", first.definition)
    elif change == "order":
        candidates[0], candidates[1] = candidates[1], candidates[0]
    elif change == "definition":
        assert first.definition is not None
        candidates[0] = managym.NameCandidate(first.name, None)
    else:
        checkpoint = torch.load(path, weights_only=False)
        del checkpoint["world_binding"]["naming_representation"]
        torch.save(checkpoint, path)
    monkeypatch.setattr(managym, "naming_catalog", lambda: candidates)
    with patch.object(ActorCritic, "load_state_dict") as load_weights:
        with pytest.raises(ValueError, match="world/setup/input binding differs"):
            load_checkpoint(str(path))
        load_weights.assert_not_called()


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


def _mixed_naming_batch(
    naming_root: managym.Env,
) -> tuple[ObservationSpace, dict[str, torch.Tensor]]:
    space = ObservationSpace(ObservationConfig(policy_input_version=8))
    search = naming_root.clone_env()
    raw = search.observation_for_player(search.current_agent_index())
    search.step(
        next(
            i
            for i, a in enumerate(raw.action_space.actions)
            if a.description == "Name Gray Ogre"
        )
    )
    ordinary = managym.Env(seed=243)
    ordinary.reset(
        [
            managym.PlayerConfig("A", {"Forest": 20, "Llanowar Elves": 20}),
            managym.PlayerConfig("B", {"Mountain": 20, "Gray Ogre": 20}),
        ]
    )
    arrays = [
        space.encode(env.observation_for_player(env.current_agent_index()))
        for env in (naming_root, search, ordinary, ordinary)
    ]
    return space, {
        key: torch.from_numpy(np.stack([row[key] for row in arrays]))
        for key in arrays[0]
    }


@pytest.mark.parametrize("film", [False, True])
@pytest.mark.parametrize("attention", [False, True])
def test_legal_row_scoring_preserves_mixed_logits_gradients_and_preparation(
    naming_root: managym.Env,
    film: bool,
    attention: bool,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(243)
    space, obs = _mixed_naming_batch(naming_root)
    config = ModelConfig(
        hidden_dim=16,
        world=managym.WORLD_SETUP,
        policy_film=film,
        attention_on=attention,
    )
    model = ActorCritic(space, config).eval()
    if model.policy_film is not None:
        with torch.no_grad():
            model.policy_film.weight.fill_(0.01)
    reference = DenseActionReference(space, config).eval()
    reference.load_state_dict(model.state_dict())
    # Move a legal offer beyond padding, with all joins kept at its new slot.
    order = torch.arange(obs["actions"].shape[1])
    order[0], order[-1] = order[-1].item(), order[0].item()
    for key in ("actions", "actions_valid", "action_focus", "name_candidates"):
        obs[key] = obs[key][:, order]
    targets = obs["actions_valid"].argmax(-1)
    actual, values = model.forward_distribution(obs)
    expected, expected_values = reference.forward_distribution(obs)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(values, expected_values, atol=2e-6, rtol=2e-5)
    for net, logits, critic in (
        (model, actual, values),
        (reference, expected, expected_values),
    ):
        (
            torch.nn.functional.cross_entropy(logits, targets) + critic.square().mean()
        ).backward()
    for (name, value), (refname, refvalue) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == refname
        if value.grad is None:
            assert refvalue.grad is None
        else:
            torch.testing.assert_close(
                value.grad, refvalue.grad, atol=2e-6, rtol=3e-4, msg=name
            )
    with torch.no_grad(), PolicyInference(model) as scope:
        prepared = scope.prepare(obs)
        direct = model(obs)
        cached = model(prepared)
        for left, right in zip(direct, cached, strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_legal_row_scoring_rejects_malformed_joins(naming_root: managym.Env) -> None:
    space, obs = _mixed_naming_batch(naming_root)
    model = ActorCritic(space, ModelConfig(hidden_dim=8))
    for key, column, value, message in (
        ("actions", -2, 0.5, "continuation reference"),
        ("action_focus", 0, -2, "focus reference"),
        ("name_candidates", 1, 0, "disagrees with vocabulary"),
    ):
        changed = {k: v.clone() for k, v in obs.items()}
        changed[key][0, 0, column] = value
        with pytest.raises(ValueError, match=message):
            model(changed)


def test_equal_rule_names_learn_opposite_public_preferences(
    naming_root: managym.Env,
) -> None:
    """A bounded model fixture, not new card definitions or a strength experiment.

    Supervised targets name the publicly disclosed card. Equalize two installed
    programs at the encoder boundary: the names, native admission and real game
    definitions are untouched. A program-only state cannot solve this pair.
    """
    torch.set_num_threads(1)
    torch.manual_seed(243)
    space, batch = _mixed_naming_batch(naming_root)
    model = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=16,
            world=managym.WORLD_SETUP,
            attention_on=False,
            policy_film=True,
        ),
    ).eval()
    catalog = list(managym.naming_catalog())
    choices = [
        next(i for i, candidate in enumerate(catalog) if candidate.name == name)
        for name in ("Gray Ogre", "Hill Giant")
    ]
    definitions: list[int] = []
    for choice in choices:
        definition = catalog[choice].definition
        assert definition is not None
        definitions.append(definition + 1)
    encoder = model.semantic_cards
    assert encoder is not None
    program_rows = encoder.definition_rows[torch.tensor(definitions)] - 1
    with torch.no_grad():
        encoder.tokens[program_rows[1]] = encoder.tokens[program_rows[0]]
        encoder.lengths[program_rows[1]] = encoder.lengths[program_rows[0]]
    paired = {
        key: value[:1].repeat(2, *([1] * (value.ndim - 1)))
        for key, value in batch.items()
    }
    paired["decklists"].zero_()
    for row, definition in enumerate(definitions):
        paired["decklists"][row, 1, 0] = torch.tensor([definition, 4, 4, 0, 0])
    # Fit a two-example linear probe through the policy's existing final head.
    # This is a deterministic representability check, not SGD convergence or
    # sample-efficiency evidence. Nonzero FiLM exposes public context to actions.
    assert model.policy_film is not None
    with torch.no_grad():
        model.policy_film.weight.normal_(0, 0.5)
    assert model.name_candidates is not None
    with patch.object(
        model.name_candidates,
        "public_definitions",
        side_effect=lambda programs: programs,
    ):
        indistinguishable, _ = model(paired)
        torch.testing.assert_close(
            indistinguishable[0], indistinguishable[1], atol=0, rtol=0
        )
    features: list[torch.Tensor] = []

    def retain_features(
        module: torch.nn.Module, args: tuple[torch.Tensor, ...]
    ) -> None:
        features.append(args[0].detach())

    hook = model.policy_head[-1].register_forward_pre_hook(retain_features)
    before, _ = model(paired)
    hook.remove()
    count = len(catalog)
    encoded = features[0].reshape(2, count, -1)
    differences = (encoded[:, choices[0]] - encoded[:, choices[1]]).double()
    target_margins = torch.tensor([0.001, -0.001], dtype=torch.float64)
    # ReLU features include exact zero columns. Rank-revealing QR (the CPU
    # default gelsy driver) can report rank zero for this underdetermined matrix
    # on Linux LAPACK. SVD handles zero columns without losing the rank-2 signal.
    fit = torch.linalg.lstsq(differences, target_margins, driver="gelsd")
    assert fit.rank == 2
    assert fit.singular_values[-1] / fit.singular_values[0] > 1e-6
    fitted = fit.solution
    torch.testing.assert_close(
        differences @ fitted, target_margins, atol=1e-10, rtol=1e-8
    )
    # Gradients also reach both public identity and executable program inputs.
    loss = torch.nn.functional.cross_entropy(before[:, choices], torch.arange(2))
    loss.backward()
    assert model.name_candidates is not None
    assert model.name_candidates.embedding.weight.grad is not None
    assert model.name_candidates.embedding.weight.grad.abs().sum() > 0
    assert encoder.token_embedding.weight.grad is not None
    assert encoder.token_embedding.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.policy_head[-1].weight.copy_(fitted[None].float())
    logits, _ = model(paired)
    assert torch.equal(logits[:, choices].argmax(-1), torch.arange(2))
    assert (logits.softmax(-1)[:, : len(catalog)] > 0).all()
    # Moving choices retains learned identity, not ordinal preferences.
    order = torch.arange(logits.shape[1] - 1, -1, -1)
    reordered = dict(paired)
    for key in ("actions", "actions_valid", "action_focus", "name_candidates"):
        reordered[key] = paired[key][:, order]
    moved, _ = model(reordered)
    torch.testing.assert_close(moved, logits[:, order], atol=2e-6, rtol=1e-5)
    # Closed information is the native zero decklist projection; both states
    # then have exactly the same observable evidence and must score identically.
    closed = {key: value.clone() for key, value in paired.items()}
    closed["decklists"][:, 1].zero_()
    hidden, _ = model(closed)
    torch.testing.assert_close(hidden[0], hidden[1], atol=0, rtol=0)


def test_compound_naming_keeps_complete_atomic_offer_support(
    naming_root: managym.Env,
) -> None:
    space = ObservationSpace(ObservationConfig(policy_input_version=8))
    raw = naming_root.observation_for_player(naming_root.current_agent_index())
    obs = _tensor(space.encode(raw))
    offers = flatten_projection(
        json.loads(naming_root.compound_offers().projection_json())
    )
    model = ActorCritic(
        space,
        ModelConfig(hidden_dim=8, world=managym.WORLD_SETUP, compound_decisions=True),
    ).eval()
    index = next(
        i
        for i, candidate in enumerate(managym.naming_catalog())
        if candidate.name == "Gray Ogre"
    )
    output = model.compound(obs, offers, tokens=(index,))
    assert len(output.probabilities[0]) == len(managym.naming_catalog())
    assert (output.probabilities[0] > 0).all()
    assert torch.isfinite(output.log_prob)
