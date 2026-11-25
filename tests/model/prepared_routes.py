"""Prepare standalone encoder fixtures through the policy's host routing code."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, cast

import numpy as np
import torch

from manabot.backends.torch.arrays import device_of, on_device
from manabot.backends.torch.policy import ActorCritic, _catalog_order
from manabot.backends.torch.semantic_cards import SemanticCardEncoder
from manabot.env.observation import FACT_VALID_INDEX, ObservationSpace
from manabot.policy.admission import Catalog
from manabot.policy.prepare import (
    CatalogPrograms,
    EffectRoutes,
    FactRoutes,
    Layout,
    Prepared,
    Tables,
    _effects,
    _facts,
    catalog_programs,
)
from manabot.policy.readings import Readings, legacy_readings, read_numeric
from manabot.policy.spans import ObservationSelection
from manabot.verify.choice_inputs import encode
import managym


def dense_action_features(agent: ActorCritic, rows: torch.Tensor) -> torch.Tensor:
    """Read every action slot for dense comparisons with the legal-row forward."""
    features = read_numeric(rows.detach().cpu().numpy(), agent.readings["actions"])
    return torch.from_numpy(features).to(rows.device)


def _tables(
    catalog: SemanticCardEncoder | None, readings: Readings | None = None
) -> Tables:
    """Tables over `readings`, by default those of a checkpoint that records none."""
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
    return replace(
        Tables.from_buffers(
            buffers,
            Layout(
                legacy_readings(ObservationSpace()) if readings is None else readings
            ),
        ),
        order_lengths=_catalog_order,
    )


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
    readings: Readings | None = None,
) -> FactRoutes[torch.Tensor]:
    values = facts.detach().cpu().numpy()
    if not torch.is_grad_enabled():
        values = values[:, (values[..., FACT_VALID_INDEX] == 1).any(0)]
    valid = values[..., FACT_VALID_INDEX] == 1
    routes = _facts(values[valid], valid, _tables(catalog, readings), objects.shape[1])
    return cast(FactRoutes[torch.Tensor], _device(routes, objects))


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


def host_inputs(obs: Mapping[str, torch.Tensor | np.ndarray]) -> dict[str, np.ndarray]:
    """Fixture tensors re-enter the public host boundary before each preparation."""
    return {
        key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        for key, value in obs.items()
    }


def fixture_tensors(raw: managym.Observation) -> dict[str, torch.Tensor]:
    """Editable logical fields for tests that construct synthetic model inputs."""
    space = ObservationSpace()
    selection = encode(raw, space)
    return {key: torch.from_numpy(selection.project(key)) for key in space.shapes}


def device_input(
    agent: ActorCritic,
    obs: Mapping[str, torch.Tensor | np.ndarray]
    | managym.Observation
    | ObservationSelection
    | managym.ObservationBatch
    | Prepared[np.ndarray],
) -> Prepared[torch.Tensor]:
    if isinstance(obs, Prepared):
        return on_device(obs, device_of(agent))
    if isinstance(obs, managym.Observation):
        obs = managym.encode_observation_batch([obs], agent.observation_space.layout)
    host = (
        obs
        if isinstance(obs, (managym.ObservationBatch, ObservationSelection))
        else host_inputs(obs)
    )
    return on_device(
        agent.prepare(host, inference=not torch.is_grad_enabled()), device_of(agent)
    )


def record_vocabulary(cards: SemanticCardEncoder | None) -> Catalog | None:
    if cards is None:
        return None
    tables = _tables(cards)
    assert tables.definition_rows is not None
    return Catalog(
        tables.definition_rows, tables.record_name_keys, tables.record_token_keys
    )


def program_input(
    cards: SemanticCardEncoder, rows: torch.Tensor | None = None
) -> CatalogPrograms[torch.Tensor]:
    """Standalone program fixtures use the same host schedule as policy preparation."""
    host = catalog_programs(
        cards.tokens.detach().cpu().numpy(),
        cards.lengths.detach().cpu().numpy(),
        None if rows is None else rows.detach().cpu().numpy(),
        order_lengths=_catalog_order,
    )
    return on_device(host, device_of(cards))
