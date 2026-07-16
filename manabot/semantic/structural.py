"""Small relational encoders for static semantic-program diagnostics.

The rules compiler remains authoritative.  This module consumes its existing
symbolic token stream and adds only deterministic static relations: sequence,
tree ancestry, field ownership, and local target-role links.  It never accepts
runtime observations, objects, legal offers, or definition-reference payloads.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import nn

from manabot.semantic.transfer import masked_mean_max_program_pool

KATA_FAMILIES = (
    "order",
    "hierarchy",
    "field_role",
    "argument_binding",
    "target_choice_role",
)
FAMILY_INDEX = {name: index for index, name in enumerate(KATA_FAMILIES)}

RELATION_NAMES = (
    "parent_to_child",
    "child_to_parent",
    "ancestor_to_descendant",
    "descendant_to_ancestor",
    "field_to_value",
    "value_to_field",
    "role_declaration_to_reference",
    "role_reference_to_declaration",
)
RELATION_INDEX = {name: index for index, name in enumerate(RELATION_NAMES)}


class StructuralSemanticError(ValueError):
    """The static token stream cannot be projected without ambiguity."""


@dataclass(frozen=True)
class StructuralProjection:
    """Static relational metadata aligned one-to-one with program tokens."""

    depth: tuple[int, ...]
    relations: Mapping[str, tuple[tuple[int, int], ...]]

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": list(self.depth),
            "relations": {
                name: [list(edge) for edge in self.relations[name]]
                for name in RELATION_NAMES
            },
        }


def _payload(symbol: str, family: str) -> str | None:
    prefix = f"{family}:"
    return symbol[len(prefix) :] if symbol.startswith(prefix) else None


def _structure_base(name: str, suffix: str) -> str:
    if not name.endswith(suffix):
        raise StructuralSemanticError(f"structure {name!r} is not a {suffix!r} token")
    return name[: -len(suffix)]


def project_static_relations(token_symbols: Sequence[str]) -> StructuralProjection:
    """Derive fail-closed static relations from a compiler-emitted token stream."""

    if not token_symbols:
        raise StructuralSemanticError("cannot project an empty semantic program")
    if any(symbol.startswith("definition_ref:") for symbol in token_symbols):
        raise StructuralSemanticError(
            "definition-reference semantics are excluded from structural katas"
        )

    count = len(token_symbols)
    parents = [-1] * count
    depths = [0] * count
    ancestors: list[tuple[int, ...]] = [()] * count
    stack: list[tuple[str, int]] = []

    for index, symbol in enumerate(token_symbols):
        structure = _payload(symbol, "structure")
        if structure is not None and structure.endswith("_end"):
            base = _structure_base(structure, "_end")
            if not stack or stack[-1][0] != base:
                active = stack[-1][0] if stack else "<empty>"
                raise StructuralSemanticError(
                    f"unbalanced {structure!r} at token {index}; active={active!r}"
                )
            parents[index] = stack[-1][1]
            depths[index] = len(stack)
            ancestors[index] = tuple(item[1] for item in stack)
            stack.pop()
            continue

        parents[index] = stack[-1][1] if stack else -1
        depths[index] = len(stack)
        ancestors[index] = tuple(item[1] for item in stack)
        if structure is not None and structure.endswith("_begin"):
            stack.append((_structure_base(structure, "_begin"), index))

    if stack:
        raise StructuralSemanticError(
            "unclosed semantic structures: " + ", ".join(item[0] for item in stack)
        )

    relation_sets: dict[str, set[tuple[int, int]]] = {
        name: set() for name in RELATION_NAMES
    }
    for child, parent in enumerate(parents):
        if parent >= 0:
            relation_sets["parent_to_child"].add((parent, child))
            relation_sets["child_to_parent"].add((child, parent))
        for ancestor in ancestors[child]:
            relation_sets["ancestor_to_descendant"].add((ancestor, child))
            relation_sets["descendant_to_ancestor"].add((child, ancestor))

    active_field: dict[int, int] = {}
    inherited_field: dict[int, int] = {}
    field_owner: dict[int, int] = {}
    for index, symbol in enumerate(token_symbols):
        container = parents[index]
        field = _payload(symbol, "field")
        if field is not None:
            active_field[container] = index
            continue
        owner = active_field.get(container, inherited_field.get(container))
        if owner is not None:
            field_owner[index] = owner
            relation_sets["field_to_value"].add((owner, index))
            relation_sets["value_to_field"].add((index, owner))
        structure = _payload(symbol, "structure")
        if structure is not None and structure.endswith("_begin") and owner is not None:
            inherited_field[index] = owner

    declarations: dict[str, int] = {}
    references: list[tuple[str, int]] = []
    for index, symbol in enumerate(token_symbols):
        role = _payload(symbol, "role")
        if role is None or not role.startswith("local:"):
            continue
        owner = field_owner.get(index)
        owner_name = _payload(token_symbols[owner], "field") if owner is not None else None
        inside_target = any(
            token_symbols[ancestor] == "structure:target_begin"
            for ancestor in ancestors[index]
        )
        if owner_name == "role" and inside_target:
            if role in declarations:
                raise StructuralSemanticError(f"ambiguous declaration for {role!r}")
            declarations[role] = index
        else:
            references.append((role, index))

    for role, reference in references:
        try:
            declaration = declarations[role]
        except KeyError as error:
            raise StructuralSemanticError(f"dangling local role {role!r}") from error
        relation_sets["role_declaration_to_reference"].add(
            (declaration, reference)
        )
        relation_sets["role_reference_to_declaration"].add(
            (reference, declaration)
        )

    return StructuralProjection(
        depth=tuple(depths),
        relations={
            name: tuple(sorted(relation_sets[name])) for name in RELATION_NAMES
        },
    )


@dataclass(frozen=True)
class KataBatch:
    token_ids: torch.Tensor
    token_kinds: torch.Tensor
    token_mask: torch.Tensor
    depth: torch.Tensor
    relations: torch.Tensor
    families: torch.Tensor
    candidate_orders: torch.Tensor
    labels: torch.Tensor

    def select(self, indexes: torch.Tensor) -> "KataBatch":
        return KataBatch(
            token_ids=self.token_ids[indexes],
            token_kinds=self.token_kinds[indexes],
            token_mask=self.token_mask[indexes],
            depth=self.depth[indexes],
            relations=self.relations[indexes],
            families=self.families[indexes],
            candidate_orders=self.candidate_orders[indexes],
            labels=self.labels[indexes],
        )


class KataProbe(nn.Module):
    """Five identical family-routed two-candidate dot-product heads."""

    def __init__(self, probe_dim: int = 24) -> None:
        super().__init__()
        self.probe_dim = probe_dim
        self.candidate_embeddings = nn.Parameter(
            torch.empty(len(KATA_FAMILIES), 2, probe_dim)
        )
        nn.init.normal_(self.candidate_embeddings, mean=0.0, std=probe_dim**-0.5)

    def forward(
        self,
        representation: torch.Tensor,
        families: torch.Tensor,
        candidate_orders: torch.Tensor,
    ) -> torch.Tensor:
        canonical = self.candidate_embeddings[families]
        ordered = canonical.gather(
            1,
            candidate_orders.unsqueeze(-1).expand(-1, -1, self.probe_dim),
        )
        return torch.einsum("bd,bcd->bc", representation, ordered) / math.sqrt(
            self.probe_dim
        )


class BagSemanticEncoder(nn.Module):
    """The landed invariant pooling path with a fixed 24-wide probe output."""

    def __init__(self, token_count: int, *, hidden_dim: int = 32, probe_dim: int = 24):
        super().__init__()
        self.token_embedding = nn.Embedding(token_count, hidden_dim)
        self.token_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh()
        )
        self.context = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.output_projection = nn.Linear(hidden_dim, probe_dim)

    def forward(self, batch: KataBatch) -> torch.Tensor:
        pooled = masked_mean_max_program_pool(
            self.token_embedding,
            self.token_projection,
            batch.token_ids,
            batch.token_mask,
        )
        return self.output_projection(self.context(pooled))


def _sinusoidal(values: torch.Tensor, width: int) -> torch.Tensor:
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=values.device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    angles = values.to(torch.float32).unsqueeze(-1) * frequencies
    out = torch.zeros((*values.shape, width), device=values.device)
    out[..., 0::2] = torch.sin(angles)
    out[..., 1::2] = torch.cos(angles[..., : out[..., 1::2].shape[-1]])
    return out


class RelationalAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 24,
        heads: int = 2,
        d_ff: int = 28,
        distance_clip: int = 8,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise StructuralSemanticError("d_model must be divisible by heads")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.distance_clip = distance_clip
        self.norm_attention = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attention_output = nn.Linear(d_model, d_model)
        self.distance_bias = nn.Parameter(
            torch.zeros(heads, distance_clip * 2 + 1)
        )
        self.relation_bias = nn.Parameter(
            torch.zeros(heads, len(RELATION_NAMES))
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, _ = values.shape
        normalized = self.norm_attention(values)
        qkv = self.qkv(normalized).reshape(
            batch_size, length, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )

        positions = torch.arange(length, device=values.device)
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).clamp(
            -self.distance_clip, self.distance_clip
        ) + self.distance_clip
        scores = scores + self.distance_bias[:, distances].unsqueeze(0)
        scores = scores + torch.einsum(
            "brij,hr->bhij", relations.to(values.dtype), self.relation_bias
        )
        scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch_size, length, self.d_model
        )
        values = values + self.attention_output(attended)
        return values + self.ff(self.norm_ff(values))


class RelationalSemanticEncoder(nn.Module):
    """One-block static relation-aware semantic-program encoder."""

    def __init__(
        self,
        token_count: int,
        token_kind_count: int,
        *,
        d_model: int = 24,
        heads: int = 2,
        d_ff: int = 28,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(token_count, d_model)
        self.token_kind_embedding = nn.Embedding(token_kind_count, d_model)
        self.summary_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.summary_token, mean=0.0, std=d_model**-0.5)
        self.block = RelationalAttentionBlock(
            d_model=d_model, heads=heads, d_ff=d_ff
        )
        self.final_norm = nn.LayerNorm(d_model)
        # The common-width context projection keeps the model near the landed
        # bag arm's capacity while remaining part of the timed representation.
        self.context = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())

    def forward(self, batch: KataBatch) -> torch.Tensor:
        batch_size, token_count = batch.token_ids.shape
        positions = torch.arange(token_count, device=batch.token_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        values = (
            self.token_embedding(batch.token_ids)
            + self.token_kind_embedding(batch.token_kinds)
            + _sinusoidal(positions, self.d_model)
            + _sinusoidal(batch.depth, self.d_model)
        )
        summary = self.summary_token.reshape(1, 1, -1).expand(batch_size, -1, -1)
        values = torch.cat([summary, values], dim=1)
        mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=values.device),
                batch.token_mask,
            ],
            dim=1,
        )
        relation_count = batch.relations.shape[1]
        relations = torch.zeros(
            (
                batch_size,
                relation_count,
                token_count + 1,
                token_count + 1,
            ),
            dtype=batch.relations.dtype,
            device=batch.relations.device,
        )
        relations[:, :, 1:, 1:] = batch.relations
        encoded = self.block(values, mask, relations)
        return self.context(self.final_norm(encoded[:, 0]))


class KataProbeModel(nn.Module):
    def __init__(self, encoder: nn.Module, probe: KataProbe) -> None:
        super().__init__()
        self.encoder = encoder
        self.probe = probe

    def forward(self, batch: KataBatch) -> torch.Tensor:
        return self.probe(
            self.encoder(batch), batch.families, batch.candidate_orders
        )


def build_matched_models(
    *,
    token_count: int,
    token_kind_count: int,
    seed: int,
) -> tuple[KataProbeModel, KataProbeModel]:
    """Build separately initialized encoders with byte-identical probe heads."""

    torch.manual_seed(seed + 1_000_000)
    probe = KataProbe(probe_dim=24)
    bag_probe = deepcopy(probe)
    structural_probe = deepcopy(probe)
    torch.manual_seed(seed + 2_000_000)
    bag = BagSemanticEncoder(token_count, hidden_dim=32, probe_dim=24)
    torch.manual_seed(seed + 3_000_000)
    structural = RelationalSemanticEncoder(
        token_count, token_kind_count, d_model=24, heads=2, d_ff=28
    )
    return (
        KataProbeModel(bag, bag_probe),
        KataProbeModel(structural, structural_probe),
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_bytes(model: nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
