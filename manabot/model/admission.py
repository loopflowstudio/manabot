"""Observation admission: validate a batch once, where it enters a policy.

`admit_observation(agent, obs)` checks everything about an observation batch
that the policy forward relies on, rather than the weights: shapes and binary
validity, definition and vocabulary references, permanent links, fact and
record programs, shared-effect partitions and membership, continuation and
naming references, recent events, belief rows, the legal-action mask and the
object a value aggregation reads. It returns an `AdmittedObservation`, which
keeps the tensor dictionary interface and carries the checked definition rows.

`ActorCritic.forward` and `forward_distribution` accept only admitted input
(plain dictionaries raise `TypeError`) and perform no validation host reads;
they keep the data-dependent compaction and packing (legal rows, used object
and fact columns, packed programs, shared-effect CSR routes) that make CPU
inference cheap. Checks run in the order the forward reads the inputs, so a
batch with several defects reports the first one the forward would reach.

An admitted observation is bound to the `ModelConfig` and observation
configuration that admitted it, not to one model instance: a bootstrap or
restored model with the same configuration reads it too. Admitted tensors are
read-only by contract. `bind_admitted` wraps rows re-read from a source that
was already admitted as a whole (a learner minibatch of an admitted rollout);
it performs no checks and is for that owner only.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from manabot.env.observation import (
    PERMANENT_DEFINITION_INDEX,
    PERMANENT_LINK_INDEXES,
    RELATION_VALID_INDEX,
)
from manabot.model.continuation import admit_relations
from manabot.model.facts import admit_facts
from manabot.model.numerics import buffer_on
from manabot.model.policy_distribution import legal_mask
from manabot.model.recent_events import admit_recent_events
from manabot.model.semantic_cards import (
    CardReferences,
    admit_definition_ids,
    definition_rows,
)
from manabot.model.shared_effects import admit_shared_effects

if TYPE_CHECKING:
    from manabot.model.actor_critic import ActorCritic


# Object sequence order of ActorCritic: validity key per object block.
_OBJECT_KEYS = (
    ("agent_player", "agent_player_valid"),
    ("opponent_player", "opponent_player_valid"),
    ("agent_cards", "agent_cards_valid"),
    ("opponent_cards", "opponent_cards_valid"),
    ("agent_permanents", "agent_permanents_valid"),
    ("opponent_permanents", "opponent_permanents_valid"),
)
_PERMANENT_KEYS = ("agent_permanents", "opponent_permanents")
_BELIEF_KEYS = (
    "belief_card_def_ids",
    "belief_owner_role_ids",
    "belief_hidden_zone_ids",
    "belief_count_probabilities",
    "belief_validity",
    "belief_globals",
)


def admission_binding(agent: "ActorCritic") -> str:
    """The configuration an admitted observation is bound to."""
    return (
        agent.config.model_dump_json()
        + "\n"
        + agent.observation_space.encoder.config.model_dump_json()
    )


class AdmittedObservation(dict[str, torch.Tensor]):
    """An observation batch checked once for one policy configuration.

    `references` holds the checked definition rows of visible cards,
    permanents, public hands and decklists; `legal` is whether every row has a
    legal action. Construct with `admit_observation`, never by hand.
    """

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        binding: str,
        references: CardReferences,
        legal: bool,
    ) -> None:
        super().__init__(tensors)
        self.binding = binding
        self.references = references
        self.legal = legal


def admit_observation(
    agent: "ActorCritic", obs: Mapping[str, torch.Tensor]
) -> AdmittedObservation:
    """Check one observation batch for `agent`; raise on the first defect.

    Run it on CPU tensors before any device transfer: every check reads the
    host. Malformed input raises `ValueError` (or `NumericalError` for the
    action mask) naming the failed check; nothing is truncated or repaired.
    """
    lookup = _semantic_lookup(agent, _device(obs))
    references = _references(agent, obs, lookup, check=True)
    object_count = sum(obs[key].shape[1] for key, _ in _OBJECT_KEYS)
    valid_count = sum(obs[key].shape[1] for _, key in _OBJECT_KEYS)
    if agent.belief_count_buckets > 0:
        _admit_beliefs(agent, obs)
        object_count += obs["belief_card_def_ids"].shape[1] + 1
        valid_count += obs["belief_validity"].shape[1] + 1
    _admit_permanent_links(obs, object_count)
    # Catalog record ids read the card encoder's vocabulary; without one the
    # forward rejects the configuration itself.
    cards = agent.semantic_cards
    catalog = cards if cards is not None and cards.records else None
    admit_facts(
        obs["facts"],
        agent.facts.max_facts,
        object_count,
        catalog if agent.facts.vocabulary == "catalog" else None,
    )
    admit_shared_effects(
        obs["effects"],
        obs["effect_details"],
        obs["effect_members"],
        object_count,
        catalog if agent.shared_effects.vocabulary == "catalog" else None,
    )
    relations = obs["relations"]
    _definition_ids(agent, relations[..., [5, 23]], lookup)
    admit_relations(relations, agent.continuation.relation_dim, object_count)
    if agent.recent_events is not None:
        assert lookup is not None
        admit_recent_events(obs, object_count, lookup)
    if (
        agent.config.value_aggregation != "historical_mean"
        and agent.belief_count_buckets == 0
    ):
        # Belief policies always add a valid global row.
        validity = torch.cat([obs[key] for _, key in _OBJECT_KEYS], dim=1)
        if (validity.sum(1) == 0).any():
            raise ValueError("value aggregation requires a valid object")
    legal = legal_mask(obs["actions_valid"])
    _admit_actions(agent, obs, legal, lookup, valid_count)
    return AdmittedObservation(
        obs, admission_binding(agent), references, bool(legal.any(-1).all())
    )


def bind_admitted(
    agent: "ActorCritic", obs: Mapping[str, torch.Tensor]
) -> AdmittedObservation:
    """Wrap rows of a source already admitted for `agent`, without checks."""
    lookup = _semantic_lookup(agent, _device(obs))
    return AdmittedObservation(
        obs,
        admission_binding(agent),
        _references(agent, obs, lookup, check=False),
        bool(obs["actions_valid"].any(-1).all()),
    )


def _device(obs: Mapping[str, torch.Tensor]) -> torch.device:
    return obs["actions_valid"].device


def _semantic_lookup(agent: "ActorCritic", device: torch.device) -> torch.Tensor | None:
    if agent.semantic_cards is None:
        return None
    return buffer_on(agent.semantic_cards, "definition_rows", device)


def _references(
    agent: "ActorCritic",
    obs: Mapping[str, torch.Tensor],
    lookup: torch.Tensor | None,
    *,
    check: bool,
) -> CardReferences:
    """Definition rows of visible cards, permanents, public hands and decklists."""
    permanent_ids = torch.stack(
        [obs[key][..., PERMANENT_DEFINITION_INDEX] for key in _PERMANENT_KEYS], dim=1
    )

    def rows(ids: torch.Tensor) -> torch.Tensor:
        return (
            _definition_ids(agent, ids, lookup)
            if check
            else definition_rows(ids, lookup)
        )

    cards = agent.semantic_cards
    return CardReferences(
        rows(obs["semantic_cards"]),
        rows(permanent_ids),
        rows(obs["known_hand"][..., 0]) if lookup is not None else None,
        rows(obs["decklists"][..., 0])
        if cards is not None and cards.decklist_projection is not None
        else None,
    )


def _definition_ids(
    agent: "ActorCritic", ids: torch.Tensor, lookup: torch.Tensor | None
) -> torch.Tensor:
    if lookup is not None:
        return admit_definition_ids(ids, len(lookup), lookup)
    assert agent.card_definitions is not None
    return admit_definition_ids(ids, agent.card_definitions.num_embeddings)


def _admit_beliefs(agent: "ActorCritic", obs: Mapping[str, torch.Tensor]) -> None:
    missing = sorted(set(_BELIEF_KEYS).difference(obs))
    if missing:
        raise ValueError(f"belief-enabled ActorCritic is missing inputs: {missing}")
    card_ids = obs["belief_card_def_ids"].long()
    owner_ids = obs["belief_owner_role_ids"].long()
    zone_ids = obs["belief_hidden_zone_ids"].long()
    probabilities = obs["belief_count_probabilities"].float()
    validity = obs["belief_validity"].float()
    globals_ = obs["belief_globals"].float()
    if card_ids.ndim != 2:
        raise ValueError("belief ids must have shape [batch, rows]")
    expected_ids = card_ids.shape
    if owner_ids.shape != expected_ids or zone_ids.shape != expected_ids:
        raise ValueError("belief semantic ids must share shape [batch, rows]")
    if probabilities.shape != (*expected_ids, agent.belief_count_buckets):
        raise ValueError(
            "belief count probabilities must have shape [batch, rows, buckets]"
        )
    if validity.shape != expected_ids:
        raise ValueError("belief validity must have shape [batch, rows]")
    if globals_.shape != (expected_ids[0], 2):
        raise ValueError("belief globals must have shape [batch, 2]")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("belief probabilities must be finite and non-negative")
    if not torch.isfinite(globals_).all():
        raise ValueError("belief globals must be finite")
    valid_rows = validity > 0
    probability_totals = probabilities.sum(dim=-1)
    if valid_rows.any() and not torch.allclose(
        probability_totals[valid_rows],
        torch.ones_like(probability_totals[valid_rows]),
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError("valid belief rows must be normalized")
    for ids, embedding, label in (
        (card_ids, agent.belief_card_embedding, "card definition"),
        (owner_ids, agent.belief_owner_embedding, "owner role"),
        (zone_ids, agent.belief_zone_embedding, "hidden zone"),
    ):
        assert embedding is not None
        if (ids < 0).any() or (ids >= embedding.num_embeddings).any():
            raise ValueError(f"belief {label} id is outside the model vocabulary")


def _admit_permanent_links(obs: Mapping[str, torch.Tensor], object_count: int) -> None:
    """Links are object index + 1 (zero: none) into the object sequence."""
    for column in PERMANENT_LINK_INDEXES:
        links = torch.cat([obs[key][..., column] for key in _PERMANENT_KEYS], dim=1)
        if not (
            (links == links.floor()) & (links >= 0) & (links <= object_count)
        ).all():
            raise ValueError("invalid permanent link object reference")


def _admit_actions(
    agent: "ActorCritic",
    obs: Mapping[str, torch.Tensor],
    legal: torch.Tensor,
    lookup: torch.Tensor | None,
    valid_count: int,
) -> None:
    """Legal rows' continuation, naming and focus references."""
    batches, slots = legal.nonzero(as_tuple=True)
    indexes = obs["actions"][batches, slots][..., -2]
    if not ((indexes == indexes.floor()) & (indexes >= 0) & (indexes <= 64)).all():
        raise ValueError("invalid action continuation reference")
    rows = obs["relations"]
    present = (
        (indexes[:, None] == rows[..., 1][batches])
        & (rows[..., RELATION_VALID_INDEX][batches] == 1)
    ).any(-1)
    if ((indexes != 0) & ~present).any():
        raise ValueError("action continuation reference has no frame")
    candidates = obs["name_candidates"][batches, slots]
    agent.name_candidates.admit(candidates)
    if lookup is not None:
        admit_definition_ids(candidates[..., 1], len(lookup), lookup)
    focus = obs["action_focus"][batches, slots]
    if not ((focus == focus.floor()) & (focus >= -1) & (focus < valid_count)).all():
        raise ValueError("invalid action focus reference")
