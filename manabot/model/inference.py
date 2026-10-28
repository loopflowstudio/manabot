"""Policy-version-scoped preparation for inference from CPU observations.

PolicyInference admits definition references before device transfer and lazily
encodes the fixed semantic catalog once. PreparedObservation keeps the ordinary
tensor dictionary interface used by forward instrumentation. Neither prepared
inputs nor the cached catalog may cross an update boundary or enter autograd.
The caller owns the model exclusively for the scope and must not mutate prepared
observations. No serving state is persisted in the model or its checkpoints.

Device inputs move with one contiguous copy per dtype (`transfer`): serving's
float32 observation keys and int64 catalog references are two copies per batch
instead of one per key.
"""

from collections.abc import Mapping
from types import TracebackType
from typing import TYPE_CHECKING

import torch

from manabot.env.observation import PERMANENT_DEFINITION_INDEX
from manabot.model.semantic_cards import CardReferences, admit_definition_ids

if TYPE_CHECKING:
    from manabot.model.actor_critic import ActorCritic


_VALIDITY_KEYS = (
    "agent_player_valid",
    "opponent_player_valid",
    "agent_cards_valid",
    "opponent_cards_valid",
    "agent_permanents_valid",
    "opponent_permanents_valid",
)


def transfer(
    tensors: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[dict[str, torch.Tensor], int]:
    """Move CPU tensors with one copy per dtype; return views and the copy count.

    Each dtype group is packed into one contiguous host block (pinned for CUDA,
    so the copy can be asynchronous) and the device block is split back into
    views with the original shapes. Values are bitwise identical to per-key
    `.to(device)`. CPU targets return the inputs unchanged, with no copies.
    """
    if device.type == "cpu":
        return dict(tensors), 0
    groups: dict[torch.dtype, list[str]] = {}
    for key, tensor in tensors.items():
        if tensor.device.type != "cpu":
            raise ValueError("packed transfer requires CPU tensors")
        groups.setdefault(tensor.dtype, []).append(key)
    moved: dict[str, torch.Tensor] = {}
    for dtype, keys in groups.items():
        sizes = [tensors[key].numel() for key in keys]
        block = torch.empty(sum(sizes), dtype=dtype, pin_memory=device.type == "cuda")
        torch.cat([tensors[key].reshape(-1) for key in keys], out=block)
        block = block.to(device, non_blocking=device.type == "cuda")
        for key, part in zip(keys, block.split(sizes), strict=True):
            moved[key] = part.view(tensors[key].shape)
    return moved, len(groups)


class PreparedObservation(dict[str, torch.Tensor]):
    """Device tensors and CPU-admitted joins, valid only in their owning scope.

    `transfers` counts the host-to-device copies that produced them.
    """

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        scope: "PolicyInference",
        references: CardReferences,
        objects_checked: bool,
        transfers: int,
    ) -> None:
        super().__init__(tensors)
        self.scope = scope
        self.references = references
        self.objects_checked = objects_checked
        self.transfers = transfers


class PolicyInference:
    """One unchanged behavior policy, ordinarily one collection/update interval.

    Version counters reject optimizer/load_state_dict/in-place updates and tensor
    replacement. As with autograd, mutation through unsafe `.data` aliases is outside
    the contract. Ordinary tensor dictionaries still use ActorCritic's differentiable path.
    """

    def __init__(self, agent: "ActorCritic") -> None:
        self.agent = agent
        self.device = next(agent.parameters()).device
        self._closed = False
        self._versions = self._state()
        self._modes = tuple(module.training for module in agent.modules())
        self._spec = agent.config.model_dump_json()
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

    def check(self, agent: "ActorCritic") -> None:
        if self._closed:
            raise RuntimeError("inference scope is closed")
        if agent is not self.agent or self._versions != self._state():
            raise RuntimeError("inference policy changed within its scope")
        if (
            self._modes != tuple(module.training for module in agent.modules())
            or self._spec != agent.config.model_dump_json()
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
        # Lazy: recorded-output replay prepares observations but never runs ActorCritic.
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
        # Belief-enabled policies add validity rows inside ActorCritic; retain their
        # complete check there. Plain policies can check the same rows on CPU.
        objects_checked = self.agent.belief_count_buckets == 0
        if objects_checked and self.agent.config.value_aggregation != "historical_mean":
            validity = torch.cat([obs[key] for key in _VALIDITY_KEYS], dim=1)
            if (validity.sum(1) == 0).any():
                raise ValueError("value aggregation requires a valid object")
        joins = {"cards": references.cards, "permanents": references.permanents}
        if references.known is not None:
            joins["known"] = references.known
        moved, transfers = transfer(
            {**obs, **{f"references/{key}": value for key, value in joins.items()}},
            self.device,
        )
        return PreparedObservation(
            {key: moved[key] for key in obs},
            self,
            CardReferences(
                moved["references/cards"],
                moved["references/permanents"],
                moved.get("references/known"),
            ),
            objects_checked,
            transfers,
        )
