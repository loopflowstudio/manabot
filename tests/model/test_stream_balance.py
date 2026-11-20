"""Streams summed into an object token start at one scale with `stream_norm`.

Measured at initialization on objects of sampled Elemental games. Each
threshold is also asserted to fail on the historical layout where that layout
still misses it, so it would notice the norms disappearing. Fresh models keep
names off definitions (`definition_names`); names on are the case where a card
and a token permanent drift furthest apart.
"""

from functools import cache

import numpy as np
import pytest
import torch

from manabot.backends.torch._semantic_cards import KnowledgeStreams
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.prepare import Prepared
from manabot.policy.readings import published_columns
from manabot.verify.action_cases import decision_case
from tests.model.prepared_routes import device_input
from tests.model.test_input_scales import WORLD, sampled_batches

SPACE = ObservationSpace()
COLUMNS = [column.name for column in published_columns(SPACE)["agent_permanents"]]


def _agent(
    stream_norm: bool, seed: int = 0, definition_names: bool = False
) -> ActorCritic:
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
            definition_names=definition_names,
        ),
    )


@cache
def _games() -> tuple[dict[str, torch.Tensor], ...]:
    return tuple(
        {key: torch.as_tensor(value) for key, value in batch.items()}
        for batch in sampled_batches(SPACE, games=2, stride=10)
    )


def _object_definitions(
    agent: ActorCritic, prepared: Prepared[torch.Tensor]
) -> torch.Tensor:
    """The definition table object tokens read, as `_gather_object_embeddings` builds it."""
    assert agent.semantic_cards is not None and prepared.programs is not None
    return agent._definition_table(
        agent.name_candidates.public_definitions(
            agent.semantic_cards.definitions(prepared.programs)
        )
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
            table = _object_definitions(agent, prepared)
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
def test_numeric_and_definition_streams_share_a_scale(stream_norm: bool) -> None:
    """Without the norms a card's definition reaches several times its numeric
    row (measured ratio 0.44-1.08); with them each is one LayerNorm's size."""
    norms = _permanent_norms(_agent(stream_norm))
    ratio = norms["numeric"] / norms["definition"]
    balanced = bool((ratio > 0.5).all() and (ratio < 2).all())
    assert balanced == stream_norm, (ratio.min(), ratio.max())


@pytest.mark.parametrize(
    ("stream_norm", "definition_names", "balanced"),
    [(True, False, True), (True, True, True), (False, True, False)],
)
def test_token_and_card_permanents_enter_at_one_scale(
    stream_norm: bool, definition_names: bool, balanced: bool
) -> None:
    """With names on definitions and no norms, a token permanent (no name)
    enters at under half a card's size (measured 3.3 against 8.6). Without
    names the historical layout is balanced too (2.9 against 2.7), so only
    its names-on case is asserted to miss."""
    norms = _permanent_norms(_agent(stream_norm, definition_names=definition_names))
    assert len(norms["tokens"]) and len(norms["cards"])
    size = norms["tokens"].mean() / norms["cards"].mean()
    assert (2 / 3 < size < 1.5) == balanced, size


@pytest.mark.parametrize(
    ("stream_norm", "definition_names", "moves"),
    [(True, False, True), (True, True, True), (False, True, False)],
)
def test_combat_stats_move_the_whole_token(
    stream_norm: bool, definition_names: bool, moves: bool
) -> None:
    """2/2 to 9/9 moves the attacker's whole pre-attention token by at least
    0.3 under `stream_norm`, with names on definitions or off (measured
    0.36-0.53 and 0.38-0.56 over four seeds). With names and no norms it moves
    0.16-0.25. Without names the historical layout moves it 0.48-0.60: the
    stat flags (`steps:8`) already carry the stats there."""
    agent = _agent(stream_norm, definition_names=definition_names)
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
    assert (moved >= 0.3) == moves, moved


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


def _knowledge(agent: ActorCritic, batch: ObservationBuffers) -> KnowledgeStreams:
    assert agent.semantic_cards is not None
    with torch.no_grad():
        prepared = device_input(agent, batch)
        return agent.semantic_cards.knowledge_streams(
            prepared, _object_definitions(agent, prepared)
        )


@pytest.mark.parametrize("seed", range(3))
def test_a_known_card_leaves_the_decklist_stream_untouched(seed: int) -> None:
    """Learning a hand card adds a hand stream; the decklist stream is
    bit-for-bit what it was, and the knowledge vector keeps its direction
    (one norm over the sum measured cosine 0.07-0.29)."""
    agent = _agent(True, seed=seed)
    batch = next(b for b in _games() if not b["known_hand"][0, 1].any())
    before, after = _knowledge(agent, batch), _knowledge(agent, _with_known_card(batch))
    assert before.decklist is not None and after.decklist is not None
    assert torch.equal(before.decklist[0, 1], after.decklist[0, 1])
    assert not before.hand[0, 1].any() and after.hand[0, 1].any()
    cosine = torch.nn.functional.cosine_similarity(
        before.total[0, 1], after.total[0, 1], dim=0
    )
    assert cosine > 0.5, float(cosine)


@pytest.mark.parametrize("source", ["known_hand", "known_library", "decklists"])
def test_each_empty_source_is_exactly_zero(source: str) -> None:
    """An empty source contributes zero, not its norm's bias, and emptying it
    leaves the other streams as they were."""
    agent = _agent(True)
    assert agent.semantic_cards is not None
    for norm in (
        agent.semantic_cards.hand_norm,
        agent.semantic_cards.library_norm,
        agent.semantic_cards.decklist_norm,
    ):
        assert norm is not None
        norm.bias.data.fill_(1.0)
    full = _with_known_card(
        next(b for b in _games() if not b["known_hand"][0, 1].any())
    )
    emptied = {key: value.clone() for key, value in full.items()}
    emptied[source][0, 1] = 0

    def by_source(streams: KnowledgeStreams) -> dict[str, torch.Tensor]:
        assert streams.library is not None and streams.decklist is not None
        return {
            "known_hand": streams.hand[0, 1],
            "known_library": streams.library[0, 1],
            "decklists": streams.decklist[0, 1],
        }

    before, after = (
        by_source(_knowledge(agent, full)),
        by_source(_knowledge(agent, emptied)),
    )
    assert not after[source].any()
    for name in before.keys() - {source}:
        assert torch.equal(before[name], after[name]), name


def test_nothing_known_is_exactly_zero() -> None:
    agent = _agent(True)
    batch = {key: value.clone() for key, value in _games()[0].items()}
    for key in ("known_hand", "known_library", "decklists"):
        batch[key][0, 1] = 0
    known = _knowledge(agent, batch).total
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
