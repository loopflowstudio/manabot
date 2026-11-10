"""Fact and shared-effect ids read the card encoder's exact vocabulary."""

from pathlib import Path

import pytest
import torch

from manabot.env.observation import (
    FACT_DETAIL_INDEX,
    FACT_DIM,
    FACT_EFFECT,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_KINDS_V8,
    FACT_OBJECT_INDEX,
    FACT_SOURCE_INDEX,
    FACT_VALID_INDEX,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.architecture import architecture_identity, architecture_receipt
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.facts import FactEncoder
from manabot.model.semantic_cards import (
    RecordEmbeddings,
    RecordNamespace,
    SemanticCardEncoder,
)
from manabot.model.shared_effects import SharedEffectEncoder
from manabot.verify.representation import kellan_upgrade_root
import managym

WIDTH = 8
_value = managym.learning_token_value


@pytest.fixture(scope="module")
def cards() -> SemanticCardEncoder:
    torch.manual_seed(30)
    return SemanticCardEncoder(managym.WORLD_SETUP, WIDTH, True, records=True)


def _halves(value: int) -> tuple[int, int]:
    word = value & 0xFFFFFFFF
    return word >> 16, word & 0xFFFF


def _card_row(cards: SemanticCardEncoder, kind: int, value: int) -> int:
    return cards.token_vocabulary.index((kind, value)) + 1


def _granted_keyword(name: str, source: int = 0) -> torch.Tensor:
    """One `AddKeywords` effect fact granting `name` to object 1."""
    facts = torch.zeros(1, 4, FACT_DIM)
    row = facts[0, 0]
    row[FACT_OBJECT_INDEX] = 1
    row[FACT_KIND_INDEX] = FACT_EFFECT
    row[FACT_ID_INDEX : FACT_ID_INDEX + 2] = torch.tensor(
        _halves(_value("Modification::AddKeywords"))
    )
    row[FACT_DETAIL_INDEX : FACT_DETAIL_INDEX + 2] = torch.tensor(_halves(_value(name)))
    row[FACT_SOURCE_INDEX] = source
    row[FACT_VALID_INDEX] = 1
    return facts


def _fact_encoder() -> FactEncoder:
    encoder = FactEncoder(WIDTH, 4, kinds=FACT_KINDS_V8, vocabulary="catalog")
    # A fresh fact output is zero; expose the representation and its gradients.
    with torch.no_grad():
        encoder.output.weight.copy_(torch.eye(WIDTH))
    return encoder


def test_every_record_name_reads_exactly_one_row(cards: SemanticCardEncoder) -> None:
    names = managym.learning_record_names()
    rows = []
    for space, name, value, kind, token in names:
        namespace = RecordNamespace[space.upper()]
        key = (int(namespace) << 32) | (value & 0xFFFFFFFF)
        index = int(torch.searchsorted(cards.record_name_keys, torch.tensor(key)))
        assert int(cards.record_name_keys[index]) == key, name
        row = int(cards.record_name_rows[index])
        if (kind, token) in cards.token_vocabulary:
            assert row == _card_row(cards, kind, token), name
        else:
            # Appended after the card tokens, whose rows stay unchanged.
            assert row > len(cards.token_vocabulary), name
        rows.append(row)
    assert len(set(rows)) == len(names)
    assert max(rows) < cards.token_embedding.num_embeddings
    assert len(cards.record_name_keys) == len(names)


def test_granted_fact_and_printed_keyword_share_one_row(
    cards: SemanticCardEncoder,
) -> None:
    printed = _card_row(cards, 3, _value("Keyword::flying"))
    granted = cards.record_names(
        RecordNamespace.TOKEN, torch.tensor([_value("flying")])
    )
    torch.testing.assert_close(granted[0], cards.token_embedding.weight[printed])


def test_fact_and_effect_gradients_reach_shared_rows(
    cards: SemanticCardEncoder,
) -> None:
    torch.set_num_threads(1)
    printed = _card_row(cards, 3, _value("Keyword::flying"))
    modification = _card_row(cards, 3, _value("Modification::AddKeywords"))
    source = int(cards.definition_rows.nonzero()[0])  # a definition id + 1
    cards.zero_grad()
    records = RecordEmbeddings(cards, cards.definitions())
    out = _fact_encoder()(
        _granted_keyword("flying", source), torch.zeros(1, 2, WIDTH), records
    )
    out.square().sum().backward()
    grad = cards.token_embedding.weight.grad
    assert grad is not None
    assert grad[printed].abs().sum() > 0 and grad[modification].abs().sum() > 0
    assert cards.encoder.weight_ih_l0.grad is not None  # through the source row
    assert cards.encoder.weight_ih_l0.grad.abs().sum() > 0

    cards.zero_grad()
    headers = torch.zeros(1, 128, 18)
    details = torch.zeros(1, 4096, 3)
    members = torch.zeros(1, 2048, 2)
    headers[0, 0, :2] = torch.tensor(_halves(_value("Modification::AddKeywords")))
    headers[0, 0, 8:13] = torch.tensor([2, 1, 1, source, 0])
    headers[0, 0, 13:18] = torch.tensor([0, 1, 0, 1, 1])
    details[0, 0] = torch.tensor([1, *_halves(_value("flying"))])
    members[0, 0] = torch.tensor([2, 1])
    effects = SharedEffectEncoder(WIDTH, version=8, vocabulary="catalog")
    records = RecordEmbeddings(cards, cards.definitions())
    effects(
        headers, details, members, torch.zeros(1, 2, WIDTH), records
    ).square().sum().backward()
    grad = cards.token_embedding.weight.grad
    assert grad is not None
    assert grad[printed].abs().sum() > 0 and grad[modification].abs().sum() > 0


@pytest.mark.parametrize("empty", [False, True])
def test_catalog_fact_inference_preserves_sparse_rows(
    cards: SemanticCardEncoder, empty: bool
) -> None:
    torch.set_num_threads(1)
    encoder = _fact_encoder()
    source = int(cards.definition_rows.nonzero()[0])
    records = RecordEmbeddings(cards, cards.definitions())
    facts = torch.cat(
        (_granted_keyword("flying", source), _granted_keyword("trample", source)), 0
    )
    facts[1] = facts[1].flip(0)
    if empty:
        facts.zero_()
    objects = torch.randn(2, 2, WIDTH)
    expected = encoder(facts, objects, records)
    with torch.inference_mode():
        actual = encoder(facts, objects, records)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    if empty:
        torch.testing.assert_close(actual, objects)
    else:
        assert not torch.allclose(actual[:, 0], objects[:, 0])


def test_unknown_ids_fail_instead_of_hashing(cards: SemanticCardEncoder) -> None:
    encoder = _fact_encoder()
    objects = torch.zeros(1, 2, WIDTH)
    records = RecordEmbeddings(cards, cards.definitions())
    with pytest.raises(ValueError, match="outside the record vocabulary"):
        encoder(_granted_keyword("not a keyword"), objects, records)
    facts = _granted_keyword("flying")
    facts[0, 0, FACT_ID_INDEX : FACT_ID_INDEX + 2] = torch.tensor(
        _halves(_value("Modification::NoSuchModification"))
    )
    with pytest.raises(ValueError, match="outside the record vocabulary"):
        encoder(facts, objects, records)
    facts = _granted_keyword("flying", source=len(cards.definition_rows))
    with pytest.raises(ValueError, match="unadmitted"):
        encoder(facts, objects, records)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        cards.record_tokens(torch.tensor([3]), torch.tensor([_value("Effect::None")]))


def test_catalog_effect_batch_preserves_independent_outputs_and_gradients(
    cards: SemanticCardEncoder,
) -> None:
    """Program boundaries and source gradients survive batched vocabulary lookup."""
    torch.set_num_threads(1)
    env = kellan_upgrade_root()
    space = ObservationSpace(ObservationConfig(policy_input_version=9))
    observation = space.encode(env.observation_for_player(env.current_agent_index()))
    headers = torch.from_numpy(observation["effects"]).repeat(3, 1, 1)
    details = torch.from_numpy(observation["effect_details"]).repeat(3, 1, 1)
    members = torch.from_numpy(observation["effect_members"]).repeat(3, 1, 1)
    # An empty observation between live ones must not shift another program's rows.
    headers[1].zero_()
    count = int(
        torch.cat((members[..., 0].flatten(), headers[..., 10:13:2].flatten())).max()
    )
    objects = torch.randn(3, count, WIDTH)
    effects = SharedEffectEncoder(WIDTH, version=9, vocabulary="catalog")
    parameters = [*effects.parameters(), *cards.parameters()]

    def reading(batched: bool) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        effects.zero_grad(set_to_none=True)
        cards.zero_grad(set_to_none=True)
        catalog = RecordEmbeddings(cards, cards.definitions())
        if batched:
            result = effects(headers, details, members, objects, catalog)
        else:
            result = torch.cat(
                [
                    effects(
                        headers[i : i + 1],
                        details[i : i + 1],
                        members[i : i + 1],
                        objects[i : i + 1],
                        catalog,
                    )
                    for i in range(3)
                ]
            )
        result.square().sum().backward()
        return result.detach(), [
            p.grad.clone() if p.grad is not None else None for p in parameters
        ]

    together, joint_gradients = reading(True)
    separate, separate_gradients = reading(False)
    torch.testing.assert_close(together, separate)
    for joint, individual in zip(joint_gradients, separate_gradients, strict=True):
        if joint is None or individual is None:
            assert joint is individual
        else:
            torch.testing.assert_close(joint, individual, rtol=1e-4, atol=1e-5)
    # Admission is still enforced when the invalid id is in a later batch member.
    headers[2, 0, :2] = torch.tensor(
        _halves(_value("Modification::NoSuchModification"))
    )
    with pytest.raises(ValueError, match="outside the record vocabulary"):
        effects(
            headers,
            details,
            members,
            objects,
            RecordEmbeddings(cards, cards.definitions()),
        )


def test_real_program_ids_map_and_checkpoints_keep_their_vocabulary(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    env = kellan_upgrade_root()
    obs = env.observation_for_player(env.current_agent_index())
    space = ObservationSpace(ObservationConfig(policy_input_version=9))
    values = {key: torch.from_numpy(v)[None] for key, v in space.encode(obs).items()}
    assert (values["effect_details"][..., 0] >= 11).any()
    fresh = ModelConfig(hidden_dim=16, world=managym.WORLD_SETUP)
    assert fresh.record_vocabulary == "catalog"
    for vocabulary in ("catalog", "hashed"):
        config = fresh.model_copy(update={"record_vocabulary": vocabulary})
        model = ActorCritic(space, config)
        assert hasattr(model.shared_effects, "modification") == (vocabulary == "hashed")
        logits, _ = model(values)
        assert torch.isfinite(logits).all()
        path = tmp_path / f"{vocabulary}.pt"
        save_checkpoint(model, space, path, player_configs=None)
        if vocabulary == "hashed":
            # Frozen checkpoints predate the field; omission means hashed ids.
            checkpoint = torch.load(path, weights_only=False)
            del checkpoint["hypers"]["agent_hypers"]["record_vocabulary"]
            torch.save(checkpoint, path)
        restored, _ = load_checkpoint(str(path))
        assert restored.config.record_vocabulary == vocabulary
        assert architecture_receipt(restored) == architecture_receipt(model)
        torch.testing.assert_close(model(values)[0], restored(values)[0])


def test_architecture_identity_names_catalog_only_where_it_applies() -> None:
    def identity(version: int, vocabulary: str) -> str:
        config = ModelConfig(world=managym.WORLD_SETUP, record_vocabulary=vocabulary)
        space = ObservationSpace(ObservationConfig(policy_input_version=version))
        return architecture_identity(config, space)

    assert identity(6, "catalog") == identity(6, "hashed")
    assert identity(9, "catalog") != identity(9, "hashed")
    # Regimes keep their bytes: the fresh default is omitted.
    assert "record_vocabulary" not in ModelConfig().model_dump()


def test_fresh_regimes_can_select_hashed_ids() -> None:
    """A fresh regime naming `hashed` keeps it through serialization and builds
    the hashed tables, so one recipe can train both layouts side by side."""
    regime = ModelConfig(
        hidden_dim=16, world=managym.WORLD_SETUP, record_vocabulary="hashed"
    )
    payload = regime.model_dump()
    assert payload["record_vocabulary"] == "hashed"
    assert ModelConfig.model_validate(payload) == regime
    assert ModelConfig.from_checkpoint(regime.checkpoint_payload(), 9) == regime
    space = ObservationSpace(ObservationConfig(policy_input_version=9))
    model = ActorCritic(space, regime)
    assert hasattr(model.shared_effects, "modification")
    assert hasattr(model.facts, "ident")


def test_checkpoints_without_the_field_read_hashed_only_where_it_applies() -> None:
    world = {"world": managym.WORLD_SETUP}
    assert ModelConfig.from_checkpoint(world, 8).record_vocabulary == "hashed"
    assert ModelConfig.from_checkpoint(world, 9).record_vocabulary == "hashed"
    # Elsewhere both layouts build one model; omission equals the regime's.
    assert ModelConfig.from_checkpoint({}, 9) == ModelConfig()
    assert ModelConfig.from_checkpoint(world, 7) == ModelConfig(**world)
    explicit = ModelConfig(**world).checkpoint_payload()
    assert ModelConfig.from_checkpoint(explicit, 9).record_vocabulary == "catalog"


def test_pre_field_v6_checkpoint_equals_its_regime(tmp_path: Path) -> None:
    """A checkpoint saved before the field existed passes the raw
    `agent.config != regime.agent` admission and learning-state checks."""
    torch.set_num_threads(1)
    regime = ModelConfig(hidden_dim=16, world=managym.WORLD_SETUP)
    space = ObservationSpace(ObservationConfig(policy_input_version=6))
    path = tmp_path / "v6.pt"
    save_checkpoint(ActorCritic(space, regime), space, path, player_configs=None)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["hypers"]["agent_hypers"] = regime.model_dump()
    assert "record_vocabulary" not in checkpoint["hypers"]["agent_hypers"]
    torch.save(checkpoint, path)
    restored, restored_space = load_checkpoint(str(path))
    assert restored.config == regime
    assert restored_space.encoder.config == space.encoder.config
