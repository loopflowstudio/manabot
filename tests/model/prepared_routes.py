"""Prepare standalone encoder fixtures through the policy's host routing code."""

from dataclasses import fields, is_dataclass, replace
from typing import Any, cast

import numpy as np
import torch

from manabot.backends.torch._semantic_cards import SemanticCardEncoder
from manabot.backends.torch.arrays import device_of, on_device
from manabot.backends.torch.policy import ActorCritic
from manabot.policy.admission import Catalog
from manabot.policy.prepare import (
    EffectRoutes,
    FactRoutes,
    Layout,
    Prepared,
    Tables,
    _effects,
    _facts,
)


def _tables(catalog: SemanticCardEncoder | None) -> Tables:
    buffers = {
        "name_candidates.definitions": np.zeros(1, np.int64),
        "_is_agent_template": np.zeros(0, bool),
    }
    if catalog is not None:
        buffers.update(
            {
                f"semantic_cards.{name}": value.detach().cpu().numpy()
                for name, value in catalog.named_buffers()
            }
        )
        buffers["name_candidates.program_rows"] = np.zeros(1, np.int64)
    return Tables.from_buffers(buffers, Layout(0, 0, 0, (), ()))


def _device(value: Any, objects: torch.Tensor) -> Any:
    # These fixture dataclasses are generic in their array leaves.
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).to(objects.device)
        return tensor.to(objects.dtype) if tensor.is_floating_point() else tensor
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{f.name: _device(getattr(value, f.name), objects) for f in fields(value)},
        )
    return value


def fact_rows(
    facts: torch.Tensor,
    objects: torch.Tensor,
    catalog: SemanticCardEncoder | None = None,
) -> FactRoutes[torch.Tensor]:
    values = facts.detach().cpu().numpy()
    if not torch.is_grad_enabled():
        values = values[:, (values[..., 11] == 1).any(0)]
    return cast(
        FactRoutes[torch.Tensor],
        _device(_facts(values, _tables(catalog), objects.shape[1]), objects),
    )


def effect_rows(
    headers: torch.Tensor,
    details: torch.Tensor,
    members: torch.Tensor,
    objects: torch.Tensor,
    catalog: SemanticCardEncoder | None = None,
) -> EffectRoutes[torch.Tensor] | None:
    values = {
        name: value.detach().cpu().numpy()
        for name, value in (
            ("effects", headers),
            ("effect_details", details),
            ("effect_members", members),
        )
    }
    return cast(
        EffectRoutes[torch.Tensor] | None,
        _device(_effects(values, _tables(catalog), objects.shape[1]), objects),
    )


def host_inputs(obs: dict[str, torch.Tensor | np.ndarray]) -> dict[str, np.ndarray]:
    """Fixture tensors re-enter the public host boundary before each preparation."""
    return {
        key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        for key, value in obs.items()
    }


def device_input(
    agent: "ActorCritic", obs: dict[str, torch.Tensor | np.ndarray]
) -> Prepared[torch.Tensor]:
    return on_device(
        agent.prepare(host_inputs(obs), inference=not torch.is_grad_enabled()),
        device_of(agent),
    )


def record_vocabulary(cards: SemanticCardEncoder | None) -> Catalog | None:
    if cards is None:
        return None
    tables = _tables(cards)
    assert tables.definition_rows is not None
    return Catalog(
        tables.definition_rows, tables.record_name_keys, tables.record_token_keys
    )
