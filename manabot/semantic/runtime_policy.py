"""Learned policy over viewer-safe semantic runtime decisions.

The model consumes the existing :class:`SemanticDecision` join. Rust remains
the source of legal offers and the only decoder/mutation authority; this module
projects public features, scores ragged offers and candidates, and returns an
ID-only submission for the existing structured command path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from manabot.sim.structured_policy import (
    DecodedSubmission,
    PolicyScores,
    RaggedPolicyDecoder,
)

from .learning import BoundSemanticPack
from .policy import SemanticDecision, SemanticDecisionError

RUNTIME_POLICY_VERSION = 1
PolicyArm = Literal["semantic", "identity_only", "structure_shuffled"]
POLICY_ARMS: tuple[PolicyArm, ...] = (
    "semantic",
    "identity_only",
    "structure_shuffled",
)

VERBS = (
    "cast",
    "play_land",
    "activate",
    "pass_priority",
    "declare_attackers",
    "declare_blockers",
    "choose",
    "pay",
    "special",
)
PROMPTS = (
    "priority",
    "declare_attackers",
    "declare_blockers",
    "choose_target",
    "scry",
    "look_and_select",
    "pay_or_not",
    "modal",
    "discard_then_draw",
    "waterbend",
)
ZONES = ("library", "hand", "battlefield", "graveyard", "stack", "exile", "command")
OBJECT_KINDS = ("card", "permanent", "stack_ability")


class RuntimePolicyError(ValueError):
    """A decision or checkpoint is incompatible with the learned policy."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class RuntimeCatalog:
    """Padded typed-program tensors pinned to one admitted semantic pack."""

    semantic_pack_hash: str
    token_ids: torch.Tensor
    program_ids: torch.Tensor
    depth_ids: torch.Tensor
    mask: torch.Tensor
    shuffled_token_ids: torch.Tensor
    shuffled_program_ids: torch.Tensor
    shuffled_depth_ids: torch.Tensor
    vocabulary: tuple[tuple[int, int] | str, ...]
    definition_keys: tuple[str, ...]
    max_programs: int
    max_depth: int

    @property
    def definition_count(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    @classmethod
    def from_pack(cls, pack: BoundSemanticPack) -> "RuntimeCatalog":
        pairs = sorted(
            {
                (int(kind), int(value))
                for kind, value in zip(
                    pack.catalog.token_kind,
                    pack.catalog.token_value,
                    strict=True,
                )
            }
        )
        vocabulary: tuple[tuple[int, int] | str, ...] = (
            "<pad>",
            "<definition>",
            "<program>",
            *pairs,
        )
        token_by_pair = {pair: index + 3 for index, pair in enumerate(pairs)}
        structure_kind = pack.schema.token_kinds["structure"]
        structure_by_id = {
            value: name for name, value in pack.schema.structures.items()
        }

        rows: list[list[int]] = []
        programs: list[list[int]] = []
        depths: list[list[int]] = []
        shuffled_rows: list[list[int]] = []
        shuffled_programs: list[list[int]] = []
        shuffled_depths: list[list[int]] = []
        max_programs = 0
        max_depth = 0

        for definition_index, definition in enumerate(pack.ir.definitions):
            row = [1]
            program_row = [0]
            depth_row = [0]
            shuffled_row = [1]
            shuffled_program_row = [0]
            shuffled_depth_row = [0]
            definition_programs = [
                int(value) for value in definition["program_indexes"]
            ]
            max_programs = max(max_programs, len(definition_programs))

            for local_program, source_program in enumerate(
                definition_programs, start=1
            ):
                row.append(2)
                program_row.append(local_program)
                depth_row.append(0)
                shuffled_row.append(2)
                shuffled_program_row.append(local_program)
                shuffled_depth_row.append(0)

                start = int(pack.catalog.program_offsets[source_program])
                stop = int(pack.catalog.program_offsets[source_program + 1])
                token_chunk: list[int] = []
                depth_chunk: list[int] = []
                depth = 0
                for absolute in range(start, stop):
                    kind = int(pack.catalog.token_kind[absolute])
                    value = int(pack.catalog.token_value[absolute])
                    structure = (
                        structure_by_id.get(value) if kind == structure_kind else None
                    )
                    if structure is not None and structure.endswith("_end"):
                        depth = max(0, depth - 1)
                    token_chunk.append(token_by_pair[(kind, value)])
                    depth_chunk.append(depth)
                    max_depth = max(max_depth, depth)
                    if structure is not None and structure.endswith("_begin"):
                        depth += 1

                row.extend(token_chunk)
                program_row.extend([local_program] * len(token_chunk))
                depth_row.extend(depth_chunk)

                order = sorted(
                    range(len(token_chunk)),
                    key=lambda position: hashlib.sha256(
                        (
                            f"{pack.semantic_pack_hash}:{definition_index}:"
                            f"{source_program}:{position}"
                        ).encode()
                    ).digest(),
                )
                shuffled_row.extend(token_chunk[position] for position in order)
                shuffled_program_row.extend([local_program] * len(token_chunk))
                shuffled_depth_row.extend(depth_chunk[position] for position in order)

            rows.append(row)
            programs.append(program_row)
            depths.append(depth_row)
            shuffled_rows.append(shuffled_row)
            shuffled_programs.append(shuffled_program_row)
            shuffled_depths.append(shuffled_depth_row)

        width = max(len(row) for row in rows)

        def padded(values: Sequence[Sequence[int]]) -> torch.Tensor:
            return torch.tensor(
                [list(row) + [0] * (width - len(row)) for row in values],
                dtype=torch.long,
            )

        token_ids = padded(rows)
        return cls(
            semantic_pack_hash=pack.semantic_pack_hash,
            token_ids=token_ids,
            program_ids=padded(programs),
            depth_ids=padded(depths),
            mask=token_ids.ne(0),
            shuffled_token_ids=padded(shuffled_rows),
            shuffled_program_ids=padded(shuffled_programs),
            shuffled_depth_ids=padded(shuffled_depths),
            vocabulary=vocabulary,
            definition_keys=tuple(
                str(row["semantic_key"]) for row in pack.ir.definitions
            ),
            max_programs=max_programs,
            max_depth=max_depth,
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "version": RUNTIME_POLICY_VERSION,
                "semantic_pack_hash": self.semantic_pack_hash,
                "vocabulary": [
                    list(value) if isinstance(value, tuple) else value
                    for value in self.vocabulary
                ],
                "definition_keys": list(self.definition_keys),
                "token_ids": self.token_ids.tolist(),
                "program_ids": self.program_ids.tolist(),
                "depth_ids": self.depth_ids.tolist(),
                "shuffled_token_ids": self.shuffled_token_ids.tolist(),
                "shuffled_program_ids": self.shuffled_program_ids.tolist(),
                "shuffled_depth_ids": self.shuffled_depth_ids.tolist(),
            }
        )


@dataclass(frozen=True)
class RuntimePolicyFeatures:
    """Public, ID-invariant tensors for one ragged semantic decision."""

    frame: torch.Tensor
    prompt: torch.Tensor
    object_definitions: torch.Tensor
    object_kinds: torch.Tensor
    object_zones: torch.Tensor
    object_controllers: torch.Tensor
    object_public: torch.Tensor
    offer_verbs: torch.Tensor
    offer_sources: torch.Tensor
    offer_public: torch.Tensor
    candidate_objects: torch.Tensor
    candidate_players: torch.Tensor
    candidate_offers: torch.Tensor
    candidate_public: torch.Tensor

    def payload(self) -> dict[str, Any]:
        return {
            name: getattr(self, name).tolist() for name in self.__dataclass_fields__
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True)
class PolicyTargets:
    offer_index: int
    candidate_selected: tuple[bool, ...]


@dataclass(frozen=True)
class RuntimePolicyOutput:
    offer_logits: torch.Tensor
    candidate_logits: torch.Tensor

    def scores(self) -> PolicyScores:
        return PolicyScores(
            offer_scores=tuple(float(value) for value in self.offer_logits.detach()),
            candidate_scores=tuple(
                float(value) for value in self.candidate_logits.detach()
            ),
        )


class RuntimePolicyProjector:
    """Project a SemanticDecision without exposing runtime addresses."""

    def __init__(self, pack: BoundSemanticPack) -> None:
        self.pack = pack
        self.catalog = RuntimeCatalog.from_pack(pack)
        self._verb = {name: index for index, name in enumerate(VERBS)}
        self._prompt = {name: index for index, name in enumerate(PROMPTS)}
        self._zone = {name: index for index, name in enumerate(ZONES)}
        self._object_kind = {name: index for index, name in enumerate(OBJECT_KINDS)}

    def project(self, decision: SemanticDecision) -> RuntimePolicyFeatures:
        if decision.semantic_pack_hash != self.pack.semantic_pack_hash:
            raise RuntimePolicyError("decision semantic pack does not match projector")
        prompt = decision.frame.prompt
        if prompt is None:
            raise RuntimePolicyError("runtime policy cannot score a terminal frame")
        try:
            prompt_index = self._prompt[prompt.kind]
        except KeyError as error:
            raise RuntimePolicyError(f"unknown prompt kind {prompt.kind!r}") from error

        projection = decision.frame.projection
        frame = torch.tensor(
            [
                min(float(projection.turn.turn_number), 40.0) / 40.0,
                float(projection.agent.is_active),
                float(projection.agent.life) / 20.0,
                float(projection.opponent.life) / 20.0,
                float(projection.agent.library_count) / 40.0,
                float(projection.opponent.library_count) / 40.0,
                float(len(projection.agent.hand)) / 10.0,
                float(projection.opponent.hand_hidden_count) / 10.0,
                float(len(projection.agent.battlefield)) / 20.0,
                float(len(projection.opponent.battlefield)) / 20.0,
                float(len(projection.agent.graveyard)) / 20.0,
                float(len(projection.opponent.graveyard)) / 20.0,
            ],
            dtype=torch.float32,
        )

        public_by_entity: dict[tuple[str, int], tuple[float, ...]] = {}

        def add_cards(player: Any, controller: int) -> None:
            for card in (*player.hand, *player.graveyard, *player.exile):
                public_by_entity[("object", int(card.id))] = (
                    float(controller),
                    float(card.power) / 10.0,
                    float(card.toughness) / 10.0,
                    float(card.mana_value) / 10.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            for permanent in player.battlefield:
                public_by_entity[("object", int(permanent.id))] = (
                    float(controller),
                    float(permanent.power) / 10.0,
                    float(permanent.toughness) / 10.0,
                    0.0,
                    float(permanent.tapped),
                    float(permanent.summoning_sick),
                    float(permanent.damage) / 10.0,
                    float(permanent.plus1_counters) / 10.0,
                )

        add_cards(projection.agent, 0)
        add_cards(projection.opponent, 1)
        object_public: list[tuple[float, ...]] = []
        object_definitions: list[int] = []
        object_kinds: list[int] = []
        object_zones: list[int] = []
        object_controllers: list[int] = []
        for row in decision.objects:
            if not 0 <= row.definition_row < self.catalog.definition_count:
                raise RuntimePolicyError(f"unknown definition row {row.definition_row}")
            try:
                kind = self._object_kind[row.object_kind]
                zone = self._zone[row.zone]
            except KeyError as error:
                raise RuntimePolicyError(
                    "unknown runtime object kind or zone"
                ) from error
            relative_controller = (
                0 if row.controller == int(projection.agent.player_index) else 1
            )
            values = public_by_entity.get((row.subject.kind, row.subject.entity))
            if values is None:
                values = (
                    float(relative_controller),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            object_definitions.append(row.definition_row)
            object_kinds.append(kind)
            object_zones.append(zone)
            object_controllers.append(relative_controller)
            object_public.append(values)

        offer_verbs: list[int] = []
        offer_sources: list[int] = []
        offer_public: list[tuple[float, ...]] = []
        candidate_offers = [-1] * len(decision.batch.candidates)
        for offer_index, offer in enumerate(decision.batch.offers):
            verb = str(offer.get("verb"))
            try:
                offer_verbs.append(self._verb[verb])
            except KeyError as error:
                raise RuntimePolicyError(f"unknown offer verb {verb!r}") from error
            source = decision.offer_sources[offer_index]
            offer_sources.append(
                -1 if source is None or source.object_row is None else source.object_row
            )
            choice_start = decision.batch.choice_offsets[offer_index]
            choice_stop = decision.batch.choice_offsets[offer_index + 1]
            choice_rows = decision.batch.choices[choice_start:choice_stop]
            candidate_count = sum(
                row.candidate_stop - row.candidate_start for row in choice_rows
            )
            minimum = sum(row.minimum for row in choice_rows)
            maximum = sum(row.maximum for row in choice_rows)
            offer_public.append(
                (
                    float(len(choice_rows)) / 4.0,
                    float(candidate_count) / 40.0,
                    float(minimum) / 10.0,
                    float(maximum) / 10.0,
                )
            )
            for choice in choice_rows:
                for candidate_index in range(
                    choice.candidate_start, choice.candidate_stop
                ):
                    if candidate_offers[candidate_index] >= 0:
                        raise RuntimePolicyError("candidate belongs to multiple offers")
                    candidate_offers[candidate_index] = offer_index
        if any(index < 0 for index in candidate_offers):
            raise RuntimePolicyError("candidate is not bound to an offer")

        candidate_objects: list[int] = []
        candidate_players: list[int] = []
        candidate_public: list[tuple[float, ...]] = []
        for binding in decision.candidate_subjects:
            if binding.object_row is not None:
                candidate_objects.append(binding.object_row)
                candidate_players.append(-1)
                row = decision.objects[binding.object_row]
                candidate_public.append(
                    (
                        1.0,
                        float(row.controller == int(projection.agent.player_index)),
                        float(row.zone == "battlefield"),
                    )
                )
            elif binding.player_index is not None:
                candidate_objects.append(-1)
                relative = (
                    0
                    if binding.player_index == int(projection.agent.player_index)
                    else 1
                )
                candidate_players.append(relative)
                candidate_public.append((0.0, float(relative == 0), 0.0))
            else:
                raise RuntimePolicyError("candidate has no public subject binding")

        return RuntimePolicyFeatures(
            frame=frame,
            prompt=torch.tensor(prompt_index, dtype=torch.long),
            object_definitions=torch.tensor(object_definitions, dtype=torch.long),
            object_kinds=torch.tensor(object_kinds, dtype=torch.long),
            object_zones=torch.tensor(object_zones, dtype=torch.long),
            object_controllers=torch.tensor(object_controllers, dtype=torch.long),
            object_public=torch.tensor(object_public, dtype=torch.float32).reshape(
                -1, 8
            ),
            offer_verbs=torch.tensor(offer_verbs, dtype=torch.long),
            offer_sources=torch.tensor(offer_sources, dtype=torch.long),
            offer_public=torch.tensor(offer_public, dtype=torch.float32).reshape(-1, 4),
            candidate_objects=torch.tensor(candidate_objects, dtype=torch.long),
            candidate_players=torch.tensor(candidate_players, dtype=torch.long),
            candidate_offers=torch.tensor(candidate_offers, dtype=torch.long),
            candidate_public=torch.tensor(
                candidate_public, dtype=torch.float32
            ).reshape(-1, 3),
        )


class SemanticRuntimePolicy(nn.Module):
    """Small matched-capacity Transformer policy over semantic decisions."""

    def __init__(
        self,
        catalog: RuntimeCatalog,
        arm: PolicyArm,
        *,
        hidden_dim: int = 32,
        attention_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if arm not in POLICY_ARMS:
            raise RuntimePolicyError(f"unknown policy arm {arm!r}")
        if hidden_dim % attention_heads:
            raise RuntimePolicyError("hidden_dim must be divisible by attention_heads")
        self.catalog = catalog
        self.arm = arm
        self.hidden_dim = hidden_dim
        self.token_embedding = nn.Embedding(
            max(catalog.vocabulary_size, catalog.definition_count + 3), hidden_dim
        )
        self.position_embedding = nn.Embedding(catalog.token_ids.shape[1], hidden_dim)
        self.program_embedding = nn.Embedding(catalog.max_programs + 1, hidden_dim)
        self.depth_embedding = nn.Embedding(catalog.max_depth + 1, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.definition_encoder = nn.TransformerEncoder(
            layer, num_layers=transformer_layers, enable_nested_tensor=False
        )
        self.frame_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.prompt_embedding = nn.Embedding(len(PROMPTS), hidden_dim)
        self.verb_embedding = nn.Embedding(len(VERBS), hidden_dim)
        self.kind_embedding = nn.Embedding(len(OBJECT_KINDS), hidden_dim)
        self.zone_embedding = nn.Embedding(len(ZONES), hidden_dim)
        self.controller_embedding = nn.Embedding(2, hidden_dim)
        self.player_embedding = nn.Embedding(3, hidden_dim)
        self.object_public_encoder = nn.Linear(8, hidden_dim)
        self.offer_public_encoder = nn.Linear(4, hidden_dim)
        self.candidate_public_encoder = nn.Linear(3, hidden_dim)
        self.null_source = nn.Parameter(torch.zeros(hidden_dim))
        self.null_candidate = nn.Parameter(torch.zeros(hidden_dim))
        self.offer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    @property
    def architecture(self) -> dict[str, Any]:
        first_layer = self.definition_encoder.layers[0]
        return {
            "version": RUNTIME_POLICY_VERSION,
            "arm": self.arm,
            "hidden_dim": self.hidden_dim,
            "attention_heads": first_layer.self_attn.num_heads,
            "transformer_layers": len(self.definition_encoder.layers),
            "catalog_digest": self.catalog.digest,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }

    def encode_definitions(self) -> torch.Tensor:
        device = self.token_embedding.weight.device
        if self.arm == "identity_only":
            token_ids = torch.arange(
                3, self.catalog.definition_count + 3, device=device
            ).unsqueeze(1)
            program_ids = torch.zeros_like(token_ids)
            depth_ids = torch.zeros_like(token_ids)
            mask = torch.ones_like(token_ids, dtype=torch.bool)
        elif self.arm == "structure_shuffled":
            token_ids = self.catalog.shuffled_token_ids.to(device)
            program_ids = self.catalog.shuffled_program_ids.to(device)
            depth_ids = self.catalog.shuffled_depth_ids.to(device)
            mask = self.catalog.mask.to(device)
        else:
            token_ids = self.catalog.token_ids.to(device)
            program_ids = self.catalog.program_ids.to(device)
            depth_ids = self.catalog.depth_ids.to(device)
            mask = self.catalog.mask.to(device)
        positions = torch.arange(token_ids.shape[1], device=device).unsqueeze(0)
        encoded = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
            + self.program_embedding(program_ids)
            + self.depth_embedding(depth_ids)
        )
        encoded = self.definition_encoder(encoded, src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        features: RuntimePolicyFeatures,
        definitions: torch.Tensor | None = None,
    ) -> RuntimePolicyOutput:
        return self.forward_many((features,), definitions)[0]

    def forward_many(
        self,
        feature_batch: Sequence[RuntimePolicyFeatures],
        definitions: torch.Tensor | None = None,
    ) -> tuple[RuntimePolicyOutput, ...]:
        """Score a ragged batch without padding runtime objects or offers."""

        if not feature_batch:
            raise RuntimePolicyError("runtime policy batch cannot be empty")
        device = self.token_embedding.weight.device
        definitions = self.encode_definitions() if definitions is None else definitions
        if definitions.shape != (self.catalog.definition_count, self.hidden_dim):
            raise RuntimePolicyError("cached definition representations are misaligned")
        frames = self.frame_encoder(
            torch.stack([features.frame for features in feature_batch]).to(device)
        ) + self.prompt_embedding(
            torch.stack([features.prompt for features in feature_batch]).to(device)
        )

        object_counts = tuple(
            len(features.object_definitions) for features in feature_batch
        )
        object_offsets = []
        offset = 0
        for count in object_counts:
            object_offsets.append(offset)
            offset += count
        object_definitions = definitions[
            torch.cat([features.object_definitions for features in feature_batch]).to(
                device
            )
        ]
        objects = (
            object_definitions
            + self.kind_embedding(
                torch.cat([features.object_kinds for features in feature_batch]).to(
                    device
                )
            )
            + self.zone_embedding(
                torch.cat([features.object_zones for features in feature_batch]).to(
                    device
                )
            )
            + self.controller_embedding(
                torch.cat(
                    [features.object_controllers for features in feature_batch]
                ).to(device)
            )
            + self.object_public_encoder(
                torch.cat([features.object_public for features in feature_batch]).to(
                    device
                )
            )
        )

        offer_counts = tuple(len(features.offer_verbs) for features in feature_batch)
        offer_offsets = []
        offset = 0
        for count in offer_counts:
            offer_offsets.append(offset)
            offset += count
        offer_batch = torch.repeat_interleave(
            torch.arange(len(feature_batch), device=device),
            torch.tensor(offer_counts, device=device),
        )
        source_rows = torch.cat(
            [features.offer_sources for features in feature_batch]
        ).to(device)
        for batch_index, (start, count) in enumerate(
            zip(offer_offsets, offer_counts, strict=True)
        ):
            rows = source_rows[start : start + count]
            rows[rows.ge(0)] += object_offsets[batch_index]
        sources = self.null_source.expand(len(source_rows), -1).clone()
        present_sources = source_rows.ge(0)
        if present_sources.any():
            sources[present_sources] = objects[source_rows[present_sources]]
        offers = (
            frames[offer_batch]
            + self.verb_embedding(
                torch.cat([features.offer_verbs for features in feature_batch]).to(
                    device
                )
            )
            + sources
            + self.offer_public_encoder(
                torch.cat([features.offer_public for features in feature_batch]).to(
                    device
                )
            )
        )
        offer_logits = self.offer_head(offers).squeeze(-1)

        candidate_counts = tuple(
            len(features.candidate_objects) for features in feature_batch
        )
        candidate_offsets = []
        offset = 0
        for count in candidate_counts:
            candidate_offsets.append(offset)
            offset += count
        candidate_batch = torch.repeat_interleave(
            torch.arange(len(feature_batch), device=device),
            torch.tensor(candidate_counts, device=device),
        )
        candidate_rows = torch.cat(
            [features.candidate_objects for features in feature_batch]
        ).to(device)
        candidate_offer_rows = torch.cat(
            [features.candidate_offers for features in feature_batch]
        ).to(device)
        for batch_index, (start, count) in enumerate(
            zip(candidate_offsets, candidate_counts, strict=True)
        ):
            rows = candidate_rows[start : start + count]
            rows[rows.ge(0)] += object_offsets[batch_index]
            candidate_offer_rows[start : start + count] += offer_offsets[batch_index]
        candidates = self.null_candidate.expand(len(candidate_rows), -1).clone()
        object_candidates = candidate_rows.ge(0)
        if object_candidates.any():
            candidates[object_candidates] = objects[candidate_rows[object_candidates]]
        player_rows = torch.cat(
            [features.candidate_players for features in feature_batch]
        ).to(device)
        player_indexes = torch.where(player_rows.ge(0), player_rows + 1, 0)
        candidates = (
            candidates
            + self.player_embedding(player_indexes)
            + self.candidate_public_encoder(
                torch.cat([features.candidate_public for features in feature_batch]).to(
                    device
                )
            )
        )
        if len(candidate_rows):
            candidates = (
                candidates + frames[candidate_batch] + offers[candidate_offer_rows]
            )
            candidate_logits = self.candidate_head(candidates).squeeze(-1)
        else:
            candidate_logits = torch.empty(0, device=device)
        offer_splits = torch.split(offer_logits, offer_counts)
        candidate_splits = torch.split(candidate_logits, candidate_counts)
        return tuple(
            RuntimePolicyOutput(offers_for_decision, candidates_for_decision)
            for offers_for_decision, candidates_for_decision in zip(
                offer_splits, candidate_splits, strict=True
            )
        )

    def loss(
        self,
        output: RuntimePolicyOutput,
        features: RuntimePolicyFeatures,
        targets: PolicyTargets,
    ) -> torch.Tensor:
        if not 0 <= targets.offer_index < len(output.offer_logits):
            raise RuntimePolicyError("target offer index is out of range")
        offer_loss = F.cross_entropy(
            output.offer_logits.unsqueeze(0),
            torch.tensor([targets.offer_index], device=output.offer_logits.device),
        )
        if len(targets.candidate_selected) != len(output.candidate_logits):
            raise RuntimePolicyError("candidate target count does not match decision")
        if not targets.candidate_selected:
            return offer_loss
        candidate_offer = features.candidate_offers.to(output.candidate_logits.device)
        mask = candidate_offer.eq(targets.offer_index)
        if not mask.any():
            return offer_loss
        labels = torch.tensor(
            targets.candidate_selected,
            dtype=output.candidate_logits.dtype,
            device=output.candidate_logits.device,
        )
        return offer_loss + F.binary_cross_entropy_with_logits(
            output.candidate_logits[mask], labels[mask]
        )

    def submission(
        self, decision: SemanticDecision, features: RuntimePolicyFeatures
    ) -> DecodedSubmission:
        self.eval()
        with torch.inference_mode():
            output = self(features)
        try:
            return RaggedPolicyDecoder().decode(decision.batch, output.scores())
        except (ValueError, SemanticDecisionError) as error:
            raise RuntimePolicyError(
                "learned scores could not decode legally"
            ) from error


def targets_from_submission(
    decision: SemanticDecision, submission: DecodedSubmission
) -> PolicyTargets:
    offer_index = next(
        (
            index
            for index, offer in enumerate(decision.batch.offers)
            if int(offer["id"]) == submission.offer_id
        ),
        None,
    )
    if offer_index is None:
        raise RuntimePolicyError("submission offer is absent from decision")
    selected: set[int] = set()
    for answer in submission.answers:
        if answer.get("kind") != "candidates":
            raise RuntimePolicyError("runtime policy admits only candidate answers")
        selected.update(int(value) for value in answer["candidates"])
    return PolicyTargets(
        offer_index=offer_index,
        candidate_selected=tuple(
            int(candidate["id"]) in selected for candidate in decision.batch.candidates
        ),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise RuntimePolicyError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    )


__all__ = [
    "POLICY_ARMS",
    "RUNTIME_POLICY_VERSION",
    "PolicyArm",
    "PolicyTargets",
    "RuntimeCatalog",
    "RuntimePolicyError",
    "RuntimePolicyFeatures",
    "RuntimePolicyOutput",
    "RuntimePolicyProjector",
    "SemanticRuntimePolicy",
    "canonical_json",
    "canonical_sha256",
    "parameter_count",
    "percentile",
    "targets_from_submission",
]
