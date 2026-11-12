"""Policy-version-scoped preparation for inference from CPU observations.

PolicyInference admits each CPU batch (`manabot.model.admission`) before device
transfer and lazily encodes the fixed semantic catalog once. PreparedObservation
is an AdmittedObservation, so it keeps the ordinary tensor dictionary interface
used by forward instrumentation. Neither prepared inputs nor the cached catalog
may cross an update boundary or enter autograd.
The caller owns the model exclusively for the scope and must not mutate prepared
observations. No serving state is persisted in the model or its checkpoints.

Device inputs move with one contiguous copy per dtype (`transfer`): serving's
float32 observation keys and int64 catalog references are two copies per batch
instead of one per key.

With bucketed shapes, preparation measures each batch's buckets on the host
before transfer; the forward trims the device tensors to them without a device
read. The prepared dictionary itself keeps full-capacity tensors. With live
tokens, preparation likewise counts the batch's live attention tokens, so the
forward packs them without reading the count back.
"""

from collections.abc import Mapping
from types import TracebackType
from typing import TYPE_CHECKING

import torch

from manabot.model.admission import AdmittedObservation, admit_observation
from manabot.model.buckets import BucketSizes, bucket_sizes, trim_observation
from manabot.model.semantic_cards import CardReferences

if TYPE_CHECKING:
    from manabot.model.actor_critic import ActorCritic


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


def transfer_admitted(
    admitted: AdmittedObservation, device: torch.device
) -> tuple[AdmittedObservation, int]:
    """Move an admitted CPU batch and its checked references to `device`.

    One copy per dtype covers the observation and its references; the result
    keeps the admission. Returns it with the copy count.
    """
    references = admitted.references
    joins = {"cards": references.cards, "permanents": references.permanents}
    if references.known is not None:
        joins["known"] = references.known
    if references.decklists is not None:
        joins["decklists"] = references.decklists
    moved, transfers = transfer(
        {**admitted, **{f"references/{key}": value for key, value in joins.items()}},
        device,
    )
    return AdmittedObservation(
        {key: moved[key] for key in admitted},
        admitted.binding,
        CardReferences(
            moved["references/cards"],
            moved["references/permanents"],
            moved.get("references/known"),
            moved.get("references/decklists"),
        ),
        admitted.legal,
    ), transfers


def admit_on_device(
    agent: "ActorCritic", obs: Mapping[str, torch.Tensor]
) -> AdmittedObservation:
    """Admit a CPU batch for `agent`, then move it to the agent's device.

    For one-off forwards outside a PolicyInference scope.
    """
    device = next(agent.parameters()).device
    return transfer_admitted(admit_observation(agent, obs), device)[0]


class PreparedObservation(AdmittedObservation):
    """Device tensors and CPU-admitted joins, valid only in their owning scope.

    `transfers` counts the host-to-device copies that produced them; `buckets`
    holds host-measured bucket sizes when the policy uses bucketed shapes;
    `live_token_count` the host-measured number of packed tokens when it uses
    live tokens.
    """

    def __init__(
        self,
        admitted: AdmittedObservation,
        scope: "PolicyInference",
        transfers: int,
        buckets: BucketSizes | None = None,
        live_token_count: int | None = None,
    ) -> None:
        super().__init__(
            admitted, admitted.binding, admitted.references, admitted.legal
        )
        self.scope = scope
        self.transfers = transfers
        self.buckets = buckets
        self.live_token_count = live_token_count


def trim_admitted(
    obs: AdmittedObservation, buckets: BucketSizes
) -> AdmittedObservation:
    """Trim admitted tensors and their checked joins to `buckets`.

    Trimming removes only rows nothing reads, so the admission still holds.
    """
    sizes = buckets.sizes
    references = obs.references
    return AdmittedObservation(
        trim_observation(obs, buckets),
        obs.binding,
        CardReferences(
            references.cards[:, :, : sizes["cards"]],
            references.permanents[:, :, : sizes["permanents"]],
            None
            if references.known is None
            else references.known[:, :, : sizes["known"]],
            None
            if references.decklists is None
            else references.decklists[:, :, : sizes["decklists"]],
        ),
        obs.legal,
    )


def trim_prepared(obs: PreparedObservation) -> PreparedObservation:
    """Trim prepared tensors and their joins to the host-measured buckets."""
    if obs.buckets is None:
        return obs
    return PreparedObservation(
        trim_admitted(obs, obs.buckets),
        obs.scope,
        obs.transfers,
        live_token_count=obs.live_token_count,
    )


class PolicyInference:
    """One unchanged behavior policy, ordinarily one collection/update interval.

    Version counters reject optimizer/load_state_dict/in-place updates and tensor
    replacement. As with autograd, mutation through unsafe `.data` aliases is outside
    the contract. Observations admitted outside a scope keep ActorCritic's
    differentiable path.
    """

    def __init__(self, agent: "ActorCritic") -> None:
        self.agent = agent
        self.device = next(agent.parameters()).device
        self._closed = False
        self._versions = self._state()
        self._modes = tuple(module.training for module in agent.modules())
        self._spec = agent.config.model_dump_json()
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
        admitted = admit_observation(self.agent, obs)
        buckets = (
            bucket_sizes(
                admitted,
                self.agent.observation_space.encoder,
                objects=self.agent.object_trimming(),
            )
            if self.agent.bucketed_shapes
            else None
        )
        live = self.agent.live_token_count(admitted)
        moved, transfers = transfer_admitted(admitted, self.device)
        return PreparedObservation(moved, self, transfers, buckets, live)
