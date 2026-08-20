"""Supervised exact-world belief learning over viewer-safe history."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from manabot.belief.encoding import (
    HAND_ZONE_ID,
    OPPONENT_OWNER_ROLE_ID,
    BeliefEncodingSchema,
)
from manabot.belief.range import BeliefError, BeliefState
from manabot.belief.state import (
    BeliefUpdate,
    BeliefUpdateReceipt,
    CompatibleDealBeliefModel,
    ViewerHistory,
)
from managym.possible_worlds import PossibleWorldSpace

DEFAULT_HISTORY_FEATURES = 64


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_input(
    space: PossibleWorldSpace,
    history: ViewerHistory,
    schema: BeliefEncodingSchema,
) -> None:
    if history.viewer != space.viewer:
        raise BeliefError("viewer history does not match the world-space viewer")
    if history.current_revision != space.source_revision:
        raise BeliefError("viewer history revision does not match the world space")
    if history.current_viewer_state_hash != space.source_viewer_state_hash:
        raise BeliefError("viewer history observation does not match the world space")
    if space.world_schema_identity != schema.world_schema_identity:
        raise BeliefError("world space does not match the belief-learning schema")
    if space.content_manifest_identity != schema.content_manifest_identity:
        raise BeliefError("world space changed the belief-learning content manifest")


def _card_vocabulary(schema: BeliefEncodingSchema) -> tuple[tuple[int, str], ...]:
    rows = tuple(
        (row.card_def_id, row.card_name)
        for row in schema.rows
        if row.owner_role_id == OPPONENT_OWNER_ROLE_ID
        and row.hidden_zone_id == HAND_ZONE_ID
    )
    if not rows or len({card_id for card_id, _ in rows}) != len(rows):
        raise BeliefError("belief-learning schema needs one opponent-hand row per card")
    if tuple(sorted(rows)) != rows:
        raise BeliefError("belief-learning card vocabulary is not canonical")
    return rows


def _history_features(history: ViewerHistory, *, feature_count: int) -> np.ndarray:
    if feature_count < 8:
        raise BeliefError("belief history feature count must be at least eight")
    features = np.zeros(feature_count, dtype=np.float32)
    features[0] = 1.0
    features[1] = float(history.viewer)
    features[2] = math.log1p(history.current_revision) / 8.0
    features[3] = math.log1p(len(history.events)) / 4.0
    if history.events:
        scale = 1.0 / math.sqrt(len(history.events))
        for position, event in enumerate(history.events):
            event_digest = hashlib.sha256(
                f"{position}:{event}".encode("ascii")
            ).digest()
            bucket = 4 + int.from_bytes(event_digest[:2], "big") % (feature_count - 4)
            sign = 1.0 if event_digest[2] & 1 else -1.0
            features[bucket] += sign * scale
    return features


@dataclass(frozen=True, slots=True)
class BeliefTrainingExample:
    """One viewer-safe candidate set plus one supervision-only world label."""

    world_space: PossibleWorldSpace
    viewer_history: ViewerHistory
    target_world: int
    supervision_receipt: str


def capture_materialized_world_supervision(
    engine: Any,
    *,
    world_space: PossibleWorldSpace,
    viewer_history: ViewerHistory,
    schema: BeliefEncodingSchema,
) -> BeliefTrainingExample:
    """Capture the authority hand behind an access-controlled training boundary."""

    _validate_input(world_space, viewer_history, schema)
    authority_view = engine.observation_for_player(world_space.opponent)
    true_hand: dict[str, int] = {}
    for card in authority_view.agent_cards:
        if int(card.zone) == HAND_ZONE_ID:
            name = str(card.name)
            true_hand[name] = true_hand.get(name, 0) + 1
    if sum(true_hand.values()) != world_space.hand_size:
        raise BeliefError(
            "materialized hand does not match the public hidden-hand size"
        )
    canonical_hand = tuple(sorted(true_hand.items()))
    target = next(
        (world.index for world in world_space.worlds if world.hand == canonical_hand),
        None,
    )
    if target is None:
        raise BeliefError("materialized hand is outside the compatible world space")
    receipt = _digest(
        {
            "schema": "manabot.belief-supervision/materialized-world-v1",
            "world_space_identity": world_space.identity,
            "viewer_history_identity": viewer_history.identity,
            "target_world": target,
            "target_hand": canonical_hand,
        }
    )
    return BeliefTrainingExample(
        world_space=world_space,
        viewer_history=viewer_history,
        target_world=target,
        supervision_receipt=receipt,
    )


@dataclass(frozen=True, slots=True)
class PackedBeliefBatch:
    """Ragged exact-world candidate sets packed into common tensors."""

    world_counts: torch.Tensor
    log_p0: torch.Tensor
    world_batch: torch.Tensor
    offsets: torch.Tensor
    target_world: torch.Tensor
    history_features: torch.Tensor


def _pack_inputs(
    inputs: Sequence[tuple[PossibleWorldSpace, ViewerHistory]],
    schema: BeliefEncodingSchema,
    *,
    feature_count: int,
    targets: Sequence[int] | None,
) -> PackedBeliefBatch:
    if not inputs:
        raise BeliefError("belief training batch must be non-empty")
    vocabulary = _card_vocabulary(schema)
    names = tuple(name for _, name in vocabulary)
    count_rows: list[list[int]] = []
    log_p0: list[float] = []
    world_batch: list[int] = []
    offsets = [0]
    histories: list[np.ndarray] = []
    for decision_index, (space, history) in enumerate(inputs):
        _validate_input(space, history, schema)
        for world in space.worlds:
            count_rows.append([world.count(name) for name in names])
            log_p0.append(math.log(world.weight) - math.log(space.total_weight))
            world_batch.append(decision_index)
        offsets.append(len(count_rows))
        histories.append(_history_features(history, feature_count=feature_count))
    target_values = (
        [-1] * len(inputs) if targets is None else [int(target) for target in targets]
    )
    if len(target_values) != len(inputs):
        raise BeliefError("belief target count differs from the decision batch")
    for index, target in enumerate(target_values):
        support_size = offsets[index + 1] - offsets[index]
        if target < -1 or target >= support_size:
            raise BeliefError("belief target world is outside its local support")
    return PackedBeliefBatch(
        world_counts=torch.tensor(count_rows, dtype=torch.int64),
        log_p0=torch.tensor(log_p0, dtype=torch.float64),
        world_batch=torch.tensor(world_batch, dtype=torch.int64),
        offsets=torch.tensor(offsets, dtype=torch.int64),
        target_world=torch.tensor(target_values, dtype=torch.int64),
        history_features=torch.from_numpy(np.stack(histories)),
    )


def pack_belief_examples(
    examples: Iterable[BeliefTrainingExample],
    schema: BeliefEncodingSchema,
    *,
    feature_count: int = DEFAULT_HISTORY_FEATURES,
) -> PackedBeliefBatch:
    """Pack variable-size canonical supports without exposing labels to inference."""

    rows = tuple(examples)
    return _pack_inputs(
        tuple((row.world_space, row.viewer_history) for row in rows),
        schema,
        feature_count=feature_count,
        targets=tuple(row.target_world for row in rows),
    )


def segment_log_softmax(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Normalize concatenated logits independently within every decision."""

    if values.ndim != 1 or offsets.ndim != 1 or offsets.numel() < 2:
        raise BeliefError("segment log-softmax requires flat values and offsets")
    segments = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        lower = int(start.item())
        upper = int(end.item())
        if lower >= upper:
            raise BeliefError("belief candidate segment must be non-empty")
        segments.append(torch.log_softmax(values[lower:upper], dim=0))
    return torch.cat(segments)


class ExactWorldBeliefModel(nn.Module):
    """Learned likelihood-ratio correction over each exact managym support."""

    architecture_identity = "manabot.exact-world-belief/v1"

    def __init__(
        self,
        schema: BeliefEncodingSchema,
        *,
        hidden_dim: int = 32,
        history_features: int = DEFAULT_HISTORY_FEATURES,
    ) -> None:
        super().__init__()
        if hidden_dim < 2:
            raise BeliefError("exact-world scorer hidden dimension is too small")
        self.schema = schema
        self.history_feature_count = history_features
        card_count = len(_card_vocabulary(schema))
        self.candidate_encoder = nn.Sequential(
            nn.Linear(card_count, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.history_encoder = nn.Sequential(
            nn.Linear(history_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        final = self.history_encoder[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def identity(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.architecture_identity.encode("ascii"))
        digest.update(self.schema.identity.encode("ascii"))
        for name, value in self.state_dict().items():
            digest.update(name.encode("ascii"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return f"{self.architecture_identity}/sha256:{digest.hexdigest()}"

    def correction_logits(self, batch: PackedBeliefBatch) -> torch.Tensor:
        device = next(self.parameters()).device
        counts = batch.world_counts.to(device=device, dtype=torch.float32)
        histories = batch.history_features.to(device=device, dtype=torch.float32)
        world_batch = batch.world_batch.to(device=device)
        candidates = self.candidate_encoder(counts)
        contexts = self.history_encoder(histories)
        return (candidates * contexts[world_batch]).sum(dim=-1) / math.sqrt(
            candidates.shape[-1]
        )

    def log_probabilities(self, batch: PackedBeliefBatch) -> torch.Tensor:
        logits = self.correction_logits(batch).to(torch.float64)
        return segment_log_softmax(
            batch.log_p0.to(logits.device) + logits,
            batch.offsets.to(logits.device),
        )

    def negative_log_likelihood(self, batch: PackedBeliefBatch) -> torch.Tensor:
        if torch.any(batch.target_world < 0):
            raise BeliefError("belief NLL requires supervision targets")
        log_probabilities = self.log_probabilities(batch)
        device = log_probabilities.device
        indexes = batch.offsets[:-1].to(device) + batch.target_world.to(device)
        return -log_probabilities[indexes].mean()

    def update(
        self,
        *,
        previous: BeliefState | None,
        world_space: PossibleWorldSpace,
        viewer_history: ViewerHistory,
    ) -> BeliefUpdate:
        """Infer from viewer-safe history and exact compatible-deal support."""

        _validate_input(world_space, viewer_history, self.schema)
        if previous is not None and previous.space.viewer != world_space.viewer:
            raise BeliefError("previous belief crossed a viewer boundary")
        model_identity = self.identity
        if not viewer_history.events:
            reference = CompatibleDealBeliefModel().update(
                previous=previous,
                world_space=world_space,
                viewer_history=viewer_history,
            )
            belief = BeliefState.from_probabilities(
                world_space, model_identity, reference.belief.probabilities
            )
        else:
            batch = _pack_inputs(
                ((world_space, viewer_history),),
                self.schema,
                feature_count=self.history_feature_count,
                targets=None,
            )
            was_training = self.training
            self.eval()
            try:
                with torch.inference_mode():
                    probabilities = (
                        self.log_probabilities(batch).exp().detach().cpu().numpy()
                    )
            finally:
                self.train(was_training)
            belief = BeliefState.from_probabilities(
                world_space, model_identity, probabilities
            )
        previous_identity = None if previous is None else previous.digest
        receipt = BeliefUpdateReceipt(
            model_identity=model_identity,
            previous_belief=previous_identity,
            viewer_history_identity=viewer_history.identity,
            consumed_history_range=(0, len(viewer_history.events)),
            world_space_identity=world_space.identity,
            normalization_error=belief.normalization_error,
            output_digest=belief.digest,
        )
        return BeliefUpdate(
            previous_belief=previous_identity,
            viewer_history=viewer_history.identity,
            belief=belief,
            update_receipt=receipt,
        )


@dataclass(frozen=True, slots=True)
class BoundedOverfitResult:
    model: ExactWorldBeliefModel
    initial_nll: float
    final_nll: float


def train_bounded_overfit(
    examples: Iterable[BeliefTrainingExample],
    schema: BeliefEncodingSchema,
    *,
    steps: int = 80,
    learning_rate: float = 0.05,
    seed: int = 197,
    hidden_dim: int = 32,
) -> BoundedOverfitResult:
    """Fit a fresh exact-world scorer under a strict local step cap."""

    if steps < 1 or steps > 256:
        raise BeliefError("bounded belief overfit steps must be between 1 and 256")
    torch.manual_seed(seed)
    model = ExactWorldBeliefModel(schema, hidden_dim=hidden_dim)
    batch = pack_belief_examples(examples, schema)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial_nll = float(model.negative_log_likelihood(batch).item())
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.negative_log_likelihood(batch)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_nll = float(model.negative_log_likelihood(batch).item())
    return BoundedOverfitResult(
        model=model,
        initial_nll=initial_nll,
        final_nll=final_nll,
    )


__all__ = [
    "BeliefTrainingExample",
    "BoundedOverfitResult",
    "ExactWorldBeliefModel",
    "PackedBeliefBatch",
    "capture_materialized_world_supervision",
    "pack_belief_examples",
    "segment_log_softmax",
    "train_bounded_overfit",
]
