"""Streams summed into an object token start at one scale with `stream_norm`.

Measured at initialization on objects of sampled Elemental games, with names
enabled. Each threshold is also asserted to fail on the historical layout, so
it would notice the norms disappearing.
"""

from functools import cache

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.readings import published_columns
from manabot.verify.action_cases import decision_case
from tests.model.prepared_routes import device_input
from tests.model.test_input_scales import WORLD, sampled_batches

SPACE = ObservationSpace()
COLUMNS = [column.name for column in published_columns(SPACE)["agent_permanents"]]


def _agent(stream_norm: bool, seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(
        SPACE,
        ModelConfig(
            world=WORLD,
            hidden_dim=64,
            attention_layers=2,
            policy_film=True,
            value_aggregation="masked_mean",
            stream_norm=stream_norm,
        ),
    )


@cache
def _games() -> tuple[dict[str, torch.Tensor], ...]:
    return tuple(
        {key: torch.as_tensor(value) for key, value in batch.items()}
        for batch in sampled_batches(SPACE, games=2, stride=10)
    )


def _permanent_norms(agent: ActorCritic) -> dict[str, np.ndarray]:
    """Per live permanent: numeric and definition stream norms, and the whole
    pre-attention token norm split by token status."""
    token_column = COLUMNS.index("is_token")
    numeric, definition, tokens, cards = [], [], [], []
    with torch.no_grad():
        for batch in _games():
            prepared = device_input(agent, batch)
            state = agent._gather_object_embeddings(prepared)
            assert agent.semantic_cards is not None and prepared.programs is not None
            table = agent._definition_table(
                agent.name_candidates.public_definitions(
                    agent.semantic_cards.definitions(prepared.programs)
                )
            )
            slots = prepared.permanents.shape[2]
            for seat, key in enumerate(("agent_permanents", "opponent_permanents")):
                valid = batch[f"{key}_valid"][0] == 1
                is_token = batch[key][0, :, token_column] == 1
                rows = agent.perm_embedding(prepared.permanents[:, seat].contiguous())
                numeric.append(agent._stream("permanent", rows)[0][valid].norm(dim=-1))
                definition.append(
                    table[prepared.permanent_rows[0, seat]][valid].norm(dim=-1)
                )
                start = 2 + 2 * prepared.agent_cards.shape[1] + seat * slots
                objects = state.objects[0, start : start + slots]
                tokens.append(objects[valid & is_token].norm(dim=-1))
                cards.append(objects[valid & ~is_token].norm(dim=-1))
    return {
        name: torch.cat(values).numpy()
        for name, values in (
            ("numeric", numeric),
            ("definition", definition),
            ("tokens", tokens),
            ("cards", cards),
        )
    }


@pytest.mark.parametrize("stream_norm", [True, False])
def test_numeric_definition_and_token_streams_share_a_scale(stream_norm: bool) -> None:
    """Without the norms a card's definition is several times its numeric
    row and a token permanent (no name) enters at under half a card's size."""
    norms = _permanent_norms(_agent(stream_norm))
    ratio = norms["numeric"] / norms["definition"]
    assert len(norms["tokens"]) and len(norms["cards"])
    size = norms["tokens"].mean() / norms["cards"].mean()
    balanced = bool((ratio > 0.5).all() and (ratio < 2).all())
    assert balanced == stream_norm, (ratio.min(), ratio.max())
    assert (2 / 3 < size < 1.5) == stream_norm, size


@pytest.mark.parametrize(("stream_norm", "moves"), [(True, True), (False, False)])
def test_combat_stats_move_the_whole_token(stream_norm: bool, moves: bool) -> None:
    """2/2 to 9/9 moves the attacker's whole pre-attention token by at least
    a quarter; summed with an unnormalized name it moves under a tenth."""
    agent = _agent(stream_norm)
    case = decision_case("attack").observation
    big = {key: value.clone() for key, value in case.items()}
    big["agent_permanents"][
        0, 0, [COLUMNS.index("power"), COLUMNS.index("toughness")]
    ] = 0.9
    with torch.no_grad():
        prepared = device_input(agent, case)
        attacker = int(prepared.focus[0, 0])
        before = agent._gather_object_embeddings(prepared).objects.flatten(0, 1)[
            attacker
        ]
        after = agent._gather_object_embeddings(device_input(agent, big)).objects
        moved = float((after.flatten(0, 1)[attacker] - before).norm() / before.norm())
    assert (moved >= 0.25) == moves, moved


def _with_known_card(batch: ObservationBuffers) -> ObservationBuffers:
    """The opponent's hand holds one known copy of their first listed card."""
    known = {key: value.clone() for key, value in batch.items()}
    assert not known["known_hand"][0, 1].any()
    known["known_hand"][0, 1, 0] = torch.tensor(
        [float(known["decklists"][0, 1, 0, 0]), 1.0]
    )
    return known


@pytest.mark.parametrize("stream_norm", [True, False])
def test_known_card_share_of_the_player_token(stream_norm: bool) -> None:
    """One known hand card moves the opponent's player token by a share at
    least as large under `stream_norm` as in the historical `/ 60` pool."""

    def share(agent: ActorCritic) -> float:
        batch = next(b for b in _games() if not b["known_hand"][0, 1].any())
        with torch.no_grad():
            before = agent._gather_object_embeddings(device_input(agent, batch))
            after = agent._gather_object_embeddings(
                device_input(agent, _with_known_card(batch))
            )
        player = before.objects[0, 1]
        return float((after.objects[0, 1] - player).norm() / player.norm())

    normalized, historical = share(_agent(True)), share(_agent(False))
    assert normalized >= historical, (normalized, historical)
    assert normalized > 0.05


def test_nothing_known_is_exactly_zero() -> None:
    agent = _agent(True)
    assert agent.semantic_cards is not None
    batch = {key: value.clone() for key, value in _games()[0].items()}
    for key in ("known_hand", "known_library", "decklists"):
        batch[key][0, 1] = 0
    with torch.no_grad():
        prepared = device_input(agent, batch)
        assert prepared.programs is not None
        table = agent._definition_table(
            agent.name_candidates.public_definitions(
                agent.semantic_cards.definitions(prepared.programs)
            )
        )
        _, known = agent.semantic_cards(prepared, table)
    assert not known[0, 1].any()
    assert known[0, 0].norm() > 1


def test_no_context_is_exactly_zero() -> None:
    """A decision without relations gets no shared context, not a norm's bias."""
    agent = _agent(True)
    with torch.no_grad():
        norm = agent.stream_norms["context"] if agent.stream_norms else None
        assert norm is not None
        norm.bias.fill_(1.0)
        case = decision_case("attack").observation
        prepared = device_input(agent, case)
        assert not prepared.relation_valid.any()
        assert not agent._gather_object_embeddings(prepared).shared_context.any()
