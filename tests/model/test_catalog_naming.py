"""Complete hierarchical name support, encoding parity and learned scoring.

Naming is a sequence of ordinary native decisions: prompts over more than
`managym.NAME_BRANCHING` names offer balanced groups of sorted names, and only
the final name choice commits. Policy input v9 encodes group spans and names
at ordinary action capacity; v8 cannot represent groups.
"""

from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch

from naming_reference import DenseActionReference, packed_definitions
import numpy as np
import pytest
import torch

from manabot.env.observation import ACTION_MODE_INDEX, NAME_DIM_V9, ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
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
        if any(offer["label"].startswith("Names ") for offer in frame["offers"]):
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


def _v9() -> ObservationSpace:
    return ObservationSpace(ObservationConfig(policy_input_version=9))


def _offer_toward(env: managym.Env, name: str) -> tuple[int, bool]:
    """The published group containing `name`, or its leaf offer, from labels."""
    actions = env.observation_for_player(env.current_agent_index()).action_space
    for index, action in enumerate(actions.actions):
        label = action.description or ""
        if label == f"Name {name}":
            return index, True
        first, _, last = label.removeprefix("Names ").partition(" through ")
        if label.startswith("Names ") and first <= name <= last:
            return index, False
    raise AssertionError(f"{name} is unreachable")


def _name_path(env: managym.Env, name: str) -> list[int]:
    """Step `env` through group prompts until the leaf offering `name`."""
    path: list[int] = []
    while True:
        index, leaf = _offer_toward(env, name)
        if leaf:
            return path
        env.step(index)
        path.append(index)


def _command(frame: dict, offer: int, tag: str) -> str:
    return json.dumps(
        {
            "command_id": tag,
            "expected_revision": frame["revision"],
            "offer_id": frame["offers"][offer]["id"],
            "answers": [],
        }
    )


def _assert_scalar_parity(env: managym.Env, space: ObservationSpace) -> dict:
    raw = env.observation_for_player(env.current_agent_index())
    reference = space.encode(raw)
    native = env.encode_observation(raw, policy_input_version=9)
    into = space.encoder.allocate()
    env.encode_observation_into(raw, into)
    for key in reference:
        np.testing.assert_allclose(native[key], reference[key], atol=1e-6, err_msg=key)
        np.testing.assert_array_equal(into[key], native[key], err_msg=key)
    return reference


def test_catalog_naming_v9_groups_then_names_with_scalar_parity_and_private_text(
    naming_root: managym.Env,
) -> None:
    space = _v9()
    catalog = managym.naming_catalog()
    # Ordinary capacity, independent of the vocabulary.
    assert space.shapes["name_candidates"] == (512, NAME_DIM_V9)
    raw = naming_root.observation_for_player(naming_root.current_agent_index())
    reference = _assert_scalar_parity(naming_root, space)
    groups = [action.name_group for action in raw.action_space.actions]
    assert all(group is not None for group in groups)
    assert 2 <= len(groups) <= managym.NAME_BRANCHING
    assert groups[0][0] == 0 and groups[-1][1] == len(catalog) - 1
    assert all(a[1] + 1 == b[0] for a, b in zip(groups, groups[1:]))
    sizes = [last - first + 1 for first, last in groups]
    assert max(sizes) - min(sizes) <= 1
    for row, (first, last) in enumerate(groups):
        assert reference["name_candidates"][row].tolist() == [0, 0, first + 1, last + 1]
        assert raw.action_space.actions[row].description == (
            f"Names {catalog[first].name} through {catalog[last].name}"
        )
    assert not reference["actions"][:, ACTION_MODE_INDEX].any()
    assert not reference["decklists"][1].any()
    assert not raw.opponent.main_counts
    text = capture_text_decision(naming_root).text
    assert f"Names {catalog[0].name} through" in text
    assert not naming_root.observation_for_player(1).action_space.actions
    # V8 offered the whole vocabulary at once and cannot represent groups.
    with pytest.raises(ValueError, match="cannot represent naming groups"):
        ObservationSpace(ObservationConfig(policy_input_version=8)).encode(raw)
    with pytest.raises(Exception, match="cannot represent naming groups"):
        naming_root.encode_observation(raw, policy_input_version=8)
    # Frozen inputs without name identity see anonymous modal choices.
    for version in (6, 7):
        legacy = naming_root.encode_observation(raw, policy_input_version=version)
        assert "name_candidates" not in legacy
        np.testing.assert_array_equal(
            legacy["actions"][: len(groups), ACTION_MODE_INDEX],
            np.arange(1, len(groups) + 1),
        )

    leaf = naming_root.clone_env()
    path = _name_path(leaf, "Fire Lord Sozin")
    assert len(path) == 1
    raw = leaf.observation_for_player(0)
    reference = _assert_scalar_parity(leaf, space)
    first, last = groups[path[0]]
    assert [a.name_index for a in raw.action_space.actions] == list(
        range(first, last + 1)
    )
    for row, index in enumerate(range(first, last + 1)):
        definition = catalog[index].definition
        assert reference["name_candidates"][row].tolist() == [
            index + 1,
            definition + 1 if definition is not None else 0,
            0,
            0,
        ]
    text = capture_text_decision(leaf).text
    assert "Fire Lord Sozin" in text
    assert ObservationConfig.from_checkpoint({}).policy_input_version == 1
    assert "max_facts" not in ObservationConfig.from_checkpoint({}).model_dump()


def _supervised_step(
    model: ActorCritic, tensors: dict[str, torch.Tensor], target: int
) -> torch.Tensor:
    """One bounded supervised gradient witness through the real policy head.

    A unit check of learnability, not a game/training experiment or strength claim.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    logits, _ = model(tensors)
    before = torch.nn.functional.cross_entropy(logits, torch.tensor([target]))
    model.zero_grad()
    before.backward()
    optimizer.step()
    after, _ = model(tensors)
    assert torch.nn.functional.cross_entropy(after, torch.tensor([target])) < before
    return after


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
    assert [c.name for c in managym.naming_catalog()] == [c.name for c in installed]
    definition_free = next(c.name for c in catalog if c.definition is None)
    assert managym.unsupported_cards([definition_free]) == [definition_free]
    monkeypatch.setattr(managym, "naming_catalog", lambda: catalog)
    space = _v9()
    assert space.shapes["actions"][0] == 512
    values = _assert_scalar_parity(root, space)
    raw = root.observation_for_player(0)
    groups = [action.name_group for action in raw.action_space.actions]
    assert len(groups) == -(-len(catalog) // managym.NAME_BRANCHING)
    model = ActorCritic(space, ModelConfig(hidden_dim=16, attention_on=False))
    tensors = _tensor(values)
    logits, _ = model(tensors)
    assert logits.shape == (1, 512)
    assert (logits.softmax(-1)[0, : len(groups)] > 0).all()
    assert model.name_candidates is not None
    embedded = model.name_candidates(tensors["name_candidates"])
    assert embedded[0, : len(groups)].unique(dim=0).shape[0] == len(groups)
    # One group contains names on both sides of 512; its preference trains every
    # member's public identity through the group's mean vector.
    target = next(i for i, (a, b) in enumerate(groups) if a <= 512 <= b)
    model.zero_grad()
    _supervised_step(model, tensors, target)
    grad = model.name_candidates.embedding.weight.grad
    assert grad is not None
    first, last = groups[target]
    members = grad[first + 1 : last + 2]
    assert members.abs().sum(-1).gt(0).all()
    # The mean shares one group gradient equally among members; padding is inert.
    torch.testing.assert_close(members, members[:1].expand_as(members))
    assert not grad[0].any()
    selected, _, _, _ = model.get_action_and_value(tensors, deterministic=True)
    assert int(selected) == target

    # The policy's choices execute through scalar Commands and vector buffers,
    # one ordinary decision per level.
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
    leaf = root.clone_env()
    group_command = _command(frame, target, "large-group")
    leaf.step_semantic_command(group_command)
    vector.step_into_buffers([target])
    leaf_frame = json.loads(leaf.semantic_decision_frame_json())
    assert json.loads(vector.semantic_decision_frames_json()[0]) == leaf_frame
    leaf_values = _assert_scalar_parity(leaf, space)
    for key, expected in leaf_values.items():
        np.testing.assert_array_equal(buffers[key][0], expected, err_msg=key)
    with pytest.raises(Exception, match="revision|stale"):
        leaf.step_semantic_command(group_command)
    leaf_tensors = _tensor(leaf_values)
    row = 512 - first
    assert leaf_values["name_candidates"][row, 0] == 513
    model.zero_grad()
    trained = _supervised_step(model, leaf_tensors, row)
    leaf_grad = model.name_candidates.embedding.weight.grad
    assert leaf_grad is not None and leaf_grad[513].any()

    root_digest = root.state_digest()
    back = next(i for i, c in enumerate(catalog) if c.name == "Fire Lord Sozin")
    ogre = next(i for i, c in enumerate(catalog) if c.name == "Gray Ogre")
    for choice in dict.fromkeys([0, 511, 512, len(catalog) - 1, back, ogre]):
        branch = root.clone_env()
        commands: list[str] = []
        while True:
            offer, is_leaf = _offer_toward(branch, catalog[choice].name)
            command = _command(
                json.loads(branch.semantic_decision_frame_json()),
                offer,
                f"large-name.{choice}.{len(commands)}",
            )
            commands.append(command)
            before = branch.state_digest()
            branch.step_semantic_command(command)
            if is_leaf:
                break
            # Group navigation changes the pending prompt and nothing else.
            assert branch.state_digest() != before
            assert not branch.observation_for_player(0).opponent.main_counts
        assert len(commands) == 2
        replay = root.clone_env()
        for command in commands:
            replay.step_semantic_command(command)
        assert branch.state_digest() == replay.state_digest()
        pending = branch.observation_for_player(0)
        searching = any(f.decision == 25 and f.mode == 4 for f in pending.continuation)
        if searching:
            assert not branch.observation_for_player(1).action_space.actions
        assert not pending.opponent.main_counts
        for command in commands:
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
        if choice == 512:
            vector.step_into_buffers([row])
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
    torch.testing.assert_close(restored(leaf_tensors)[0], trained)
    bad = {key: value.clone() for key, value in leaf_tensors.items()}
    bad["name_candidates"][0, row, 1] = managym.CARD_DEFINITION_COUNT + 100
    with pytest.raises(ValueError, match="vocabulary"):
        model(bad)
    bad = {key: value.clone() for key, value in tensors.items()}
    bad["name_candidates"][0, 0, 3] = 0
    with pytest.raises(ValueError, match="group span"):
        model(bad)
    world = deck_world(space)
    assert world["naming_representation"] == "public-groups-v1"
    assert world["naming_branching"] == managym.NAME_BRANCHING
    assert "naming_vocabulary" not in deck_world(
        ObservationSpace(ObservationConfig(policy_input_version=7))
    )


def test_catalog_naming_programs_affect_scores(
    naming_root: managym.Env,
) -> None:
    torch.set_num_threads(1)
    space = _v9()
    model = ActorCritic(
        space, ModelConfig(hidden_dim=8, attention_on=False, world=managym.WORLD_SETUP)
    )
    leaf = naming_root.clone_env()
    _name_path(leaf, "Fire Lord Sozin")
    raw = leaf.observation_for_player(0)
    tensors = _tensor(space.encode(raw))
    assert model.name_candidates is not None
    with torch.no_grad():
        model.name_candidates.embedding.weight.zero_()
    logits, _ = model(tensors)
    # These rows have identical action features/focus and zero name-identity
    # vectors. Different scores must therefore read their different programs.
    assert logits[0, : len(raw.action_space.actions)].unique().numel() > 1
    index = next(
        i
        for i, action in enumerate(raw.action_space.actions)
        if action.description == "Name Fire Lord Sozin"
    )
    torch.nn.functional.cross_entropy(logits, torch.tensor([index])).backward()
    assert model.semantic_cards is not None
    assert model.semantic_cards.token_embedding.weight.grad is not None
    assert model.semantic_cards.token_embedding.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("change", ["name", "order", "definition", "representation"])
def test_naming_checkpoint_admits_exact_vocabulary_before_loading_weights(
    naming_root: managym.Env,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    space = _v9()
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
    _name_path(env, "Gray Ogre")
    if not naming:
        # The following private search has no naming candidates.
        env.step(_offer_toward(env, "Gray Ogre")[0])
    space = _v9()
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
    space = _v9()
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
    catalog = managym.naming_catalog()
    assert buffers["name_candidates"][0, :, 3].max() == len(catalog)
    assert not buffers["name_candidates"][0, :, :2].any()
    # Group prompt, then leaf prompt, then the private search without names.
    for stage in ("group", "leaf"):
        for key, expected in space.encode(raw).items():
            np.testing.assert_allclose(
                buffers[key][0], expected, atol=1e-6, err_msg=f"{stage}:{key}"
            )
        index = next(
            i
            for i, action in enumerate(raw.action_space.actions)
            if action.description == "Name Gray Ogre"
            or (
                action.name_group is not None
                and action.name_group[0]
                <= [c.name for c in catalog].index("Gray Ogre")
                <= action.name_group[1]
            )
        )
        raw = scalar.step(index)[0]
        vector.step_into_buffers([index])
    assert not buffers["name_candidates"].any()
    for key, expected in space.encode(raw).items():
        np.testing.assert_allclose(buffers[key][0], expected, atol=1e-6, err_msg=key)


def _mixed_naming_batch(
    naming_root: managym.Env,
) -> tuple[ObservationSpace, dict[str, torch.Tensor]]:
    """Group prompt, name prompt, private search and an ordinary decision."""
    space = _v9()
    leaf = naming_root.clone_env()
    _name_path(leaf, "Gray Ogre")
    search = leaf.clone_env()
    search.step(_offer_toward(search, "Gray Ogre")[0])
    ordinary = managym.Env(seed=243)
    ordinary.reset(
        [
            managym.PlayerConfig("A", {"Forest": 20, "Llanowar Elves": 20}),
            managym.PlayerConfig("B", {"Mountain": 20, "Gray Ogre": 20}),
        ]
    )
    arrays = [
        space.encode(env.observation_for_player(env.current_agent_index()))
        for env in (naming_root, leaf, search, ordinary)
    ]
    return space, {
        key: torch.from_numpy(np.stack([row[key] for row in arrays]))
        for key in arrays[0]
    }


@pytest.mark.parametrize("film", [False, True])
@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("width", [16, 64])
def test_legal_row_scoring_preserves_mixed_logits_gradients_and_preparation(
    naming_root: managym.Env,
    film: bool,
    attention: bool,
    width: int,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(243)
    space, obs = _mixed_naming_batch(naming_root)
    config = ModelConfig(
        hidden_dim=width,
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
    assert reference.semantic_cards is not None
    with patch.object(
        reference.semantic_cards,
        "definitions",
        side_effect=partial(packed_definitions, reference.semantic_cards),
    ):
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
    # Batch row 0 is a group prompt, row 1 a name prompt.
    for key, row, column, value, message in (
        ("actions", 1, -2, 0.5, "continuation reference"),
        ("action_focus", 1, 0, -2, "focus reference"),
        ("name_candidates", 1, 1, 0, "disagrees with vocabulary"),
        ("name_candidates", 0, 3, 0, "group span"),
        ("name_candidates", 0, 2, 10_000, "unadmitted"),
    ):
        changed = {k: v.clone() for k, v in obs.items()}
        changed[key][row, 0, column] = value
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
    space = _v9()
    leaf = naming_root.clone_env()
    _name_path(leaf, "Gray Ogre")
    offers = leaf.observation_for_player(0).action_space.actions
    batch = _tensor(space.encode(leaf.observation_for_player(0)))
    model = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=16,
            world=managym.WORLD_SETUP,
            attention_on=False,
            policy_film=True,
        ),
    ).eval()
    # Both names share one leaf prompt; choices are its offer rows.
    choices = [
        next(i for i, action in enumerate(offers) if action.description == f"Name {n}")
        for n in ("Gray Ogre", "Hill Giant")
    ]
    definitions: list[int] = []
    for choice in choices:
        definition = offers[choice].name_definition
        assert definition is not None
        definitions.append(definition + 1)
    encoder = model.semantic_cards
    assert encoder is not None
    program_rows = encoder.definition_rows[torch.tensor(definitions)] - 1
    with torch.no_grad():
        encoder.tokens[program_rows[1]] = encoder.tokens[program_rows[0]]
        encoder.lengths[program_rows[1]] = encoder.lengths[program_rows[0]]
    paired = {
        key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in batch.items()
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
            indistinguishable[0], indistinguishable[1], atol=2e-6, rtol=1e-6
        )
    features: list[torch.Tensor] = []

    def retain_features(
        module: torch.nn.Module, args: tuple[torch.Tensor, ...]
    ) -> None:
        features.append(args[0].detach())

    hook = model.policy_head[-1].register_forward_pre_hook(retain_features)
    before, _ = model(paired)
    hook.remove()
    count = len(offers)
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
    assert (logits.softmax(-1)[:, :count] > 0).all()
    # Moving choices retains learned identity, not ordinal preferences.
    order = torch.arange(logits.shape[1] - 1, -1, -1)
    reordered = dict(paired)
    for key in ("actions", "actions_valid", "action_focus", "name_candidates"):
        reordered[key] = paired[key][:, order]
    moved, _ = model(reordered)
    torch.testing.assert_close(moved, logits[:, order], atol=2e-6, rtol=1e-5)
    # Closed information is the native zero decklist projection; both states
    # then have exactly the same observable evidence. Packed GEMMs can round
    # identical rows differently by batch position, so only inputs are bitwise.
    closed = {key: value.clone() for key, value in paired.items()}
    closed["decklists"][:, 1].zero_()
    for value in closed.values():
        torch.testing.assert_close(value[0], value[1], atol=0, rtol=0)
    hidden, _ = model(closed)
    torch.testing.assert_close(hidden[0], hidden[1], atol=2e-6, rtol=1e-6)
