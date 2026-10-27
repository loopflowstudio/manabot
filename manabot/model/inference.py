"""Policy-version-scoped preparation for inference from CPU observations.

PolicyInference admits definition references before device transfer and lazily
encodes the fixed semantic catalog once. PreparedObservation keeps the ordinary
tensor dictionary interface used by forward instrumentation. Neither prepared
inputs nor the cached catalog may cross an update boundary or enter autograd.
The caller owns the model exclusively for the scope and must not mutate prepared
observations. No serving state is persisted in the model or its checkpoints.
"""

from types import TracebackType
from typing import TYPE_CHECKING

import torch

from manabot.env.observation import PERMANENT_DEFINITION_INDEX
from manabot.model.semantic_cards import CardReferences, admit_definition_ids

if TYPE_CHECKING:
<<<<<<< HEAD
    from manabot.model.agent import Agent
=======
    from manabot.model.actor_critic import ActorCritic
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f


_VALIDITY_KEYS = (
    "agent_player_valid",
    "opponent_player_valid",
    "agent_cards_valid",
    "opponent_cards_valid",
    "agent_permanents_valid",
    "opponent_permanents_valid",
)


class PreparedObservation(dict[str, torch.Tensor]):
    """Device tensors and CPU-admitted joins, valid only in their owning scope."""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        scope: "PolicyInference",
        references: CardReferences,
        objects_checked: bool,
    ) -> None:
        super().__init__(tensors)
        self.scope = scope
        self.references = references
        self.objects_checked = objects_checked


class PolicyInference:
    """One unchanged behavior policy, ordinarily one collection/update interval.

    Version counters reject optimizer/load_state_dict/in-place updates and tensor
    replacement. As with autograd, mutation through unsafe `.data` aliases is outside
<<<<<<< HEAD
    the contract. Ordinary tensor dictionaries still use Agent's differentiable path.
    """

    def __init__(self, agent: "Agent") -> None:
=======
    the contract. Ordinary tensor dictionaries still use ActorCritic's differentiable path.
    """

    def __init__(self, agent: "ActorCritic") -> None:
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        self.agent = agent
        self.device = next(agent.parameters()).device
        self._closed = False
        self._versions = self._state()
        self._modes = tuple(module.training for module in agent.modules())
<<<<<<< HEAD
        self._spec = agent.hypers.model_dump_json()
=======
        self._spec = agent.config.model_dump_json()
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        self._lookup = (
            agent.semantic_cards.definition_rows.detach().cpu().clone()
            if agent.semantic_cards is not None
            else None
        )
        self._catalog: torch.Tensor | None = None

    def _state(self) -> tuple[tuple[int, int, torch.device, torch.dtype], ...]:
        return tuple(
            (id(tensor), tensor._version, tensor.device, tensor.dtype)
            for tensor in (*self.agent.parameters(), *self.agent.buffers())
        )

<<<<<<< HEAD
    def check(self, agent: "Agent") -> None:
=======
    def check(self, agent: "ActorCritic") -> None:
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        if self._closed:
            raise RuntimeError("inference scope is closed")
        if agent is not self.agent or self._versions != self._state():
            raise RuntimeError("inference policy changed within its scope")
        if (
            self._modes != tuple(module.training for module in agent.modules())
<<<<<<< HEAD
            or self._spec != agent.hypers.model_dump_json()
=======
            or self._spec != agent.config.model_dump_json()
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        ):
            raise RuntimeError("inference policy mode/configuration changed")

    def __enter__(self) -> "PolicyInference":
        self.check(self.agent)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._closed = True
        self._catalog = None

    @torch.no_grad()
    def catalog(self) -> torch.Tensor | None:
        self.check(self.agent)
<<<<<<< HEAD
        # Lazy: recorded-output replay prepares observations but never runs Agent.
=======
        # Lazy: recorded-output replay prepares observations but never runs ActorCritic.
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        if self._catalog is None and self.agent.semantic_cards is not None:
            self._catalog = self.agent.semantic_cards.definitions()
        return self._catalog

    def prepare(self, obs: dict[str, torch.Tensor]) -> PreparedObservation:
        """Admit owned CPU inputs, then transfer them and their checked indexes."""
        self.check(self.agent)
        if any(tensor.device.type != "cpu" for tensor in obs.values()):
            raise ValueError("inference preparation requires CPU observations")
        embedding = self.agent.card_definitions
        size = (
            len(self._lookup)
            if self._lookup is not None
            else (embedding.num_embeddings if embedding is not None else 0)
        )
        permanent_ids = torch.stack(
            [
                obs[key][..., PERMANENT_DEFINITION_INDEX]
                for key in ("agent_permanents", "opponent_permanents")
            ],
            dim=1,
        )
        references = CardReferences(
            admit_definition_ids(obs["semantic_cards"], size, self._lookup),
            admit_definition_ids(permanent_ids, size, self._lookup),
            admit_definition_ids(obs["known_hand"][..., 0], size, self._lookup)
            if self._lookup is not None
            else None,
        )
<<<<<<< HEAD
        # Belief-enabled policies add validity rows inside Agent; retain their
        # complete check there. Plain policies can check the same rows on CPU.
        objects_checked = self.agent.belief_count_buckets == 0
        if objects_checked and self.agent.hypers.value_aggregation != "historical_mean":
=======
        # Belief-enabled policies add validity rows inside ActorCritic; retain their
        # complete check there. Plain policies can check the same rows on CPU.
        objects_checked = self.agent.belief_count_buckets == 0
        if objects_checked and self.agent.config.value_aggregation != "historical_mean":
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
            validity = torch.cat([obs[key] for key in _VALIDITY_KEYS], dim=1)
            if (validity.sum(1) == 0).any():
                raise ValueError("value aggregation requires a valid object")
        return PreparedObservation(
            {key: value.to(self.device) for key, value in obs.items()},
            self,
            references.to(self.device),
            objects_checked,
        )
