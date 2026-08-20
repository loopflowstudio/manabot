"""Normalized manabot beliefs over canonical managym possible worlds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from managym.decision import Observation, TransitionReceipt
from managym.possible_worlds import PossibleWorldSpace, WorldQuery


class BeliefError(ValueError):
    """A belief operation violated identity, support, or normalization."""


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ViewerHistory:
    """Identity derived from the canonical viewer Observation stream."""

    schema_version: int
    viewer: int
    initial_observation_identity: str
    current_revision: int
    current_viewer_state_hash: str
    events: tuple[str, ...]

    @classmethod
    def from_observation(cls, observation: Observation) -> "ViewerHistory":
        """Start history at one authoritative viewer Observation."""

        observation_identity = _observation_identity(
            observation.schema_version,
            observation.revision,
            observation.viewer,
            observation.viewer_state_hash,
        )
        return cls(
            schema_version=observation.schema_version,
            viewer=observation.viewer,
            initial_observation_identity=observation_identity,
            current_revision=observation.revision,
            current_viewer_state_hash=observation.viewer_state_hash,
            events=observation.events,
        )

    def advance(
        self,
        receipt: TransitionReceipt,
        observation: Observation,
    ) -> "ViewerHistory":
        """Append one native transition and bind the resulting Observation."""

        if receipt.schema_version != self.schema_version:
            raise BeliefError("transition changed the viewer-history schema")
        if receipt.before_revision != self.current_revision:
            raise BeliefError("transition does not continue the viewer history")
        if observation.schema_version != self.schema_version:
            raise BeliefError("observation changed the viewer-history schema")
        if observation.viewer != self.viewer:
            raise BeliefError("observation changed the viewer-history viewer")
        if observation.revision != receipt.after_revision:
            raise BeliefError("observation does not match the transition revision")
        return ViewerHistory(
            schema_version=self.schema_version,
            viewer=self.viewer,
            initial_observation_identity=self.initial_observation_identity,
            current_revision=observation.revision,
            current_viewer_state_hash=observation.viewer_state_hash,
            events=(*self.events, *receipt.events),
        )

    @property
    def identity(self) -> str:
        return _digest(
            {
                "schema_version": self.schema_version,
                "viewer": self.viewer,
                "initial_observation_identity": self.initial_observation_identity,
                "current_observation_identity": _observation_identity(
                    self.schema_version,
                    self.current_revision,
                    self.viewer,
                    self.current_viewer_state_hash,
                ),
                "events": self.events,
            }
        )


def _observation_identity(
    schema_version: int,
    revision: int,
    viewer: int,
    viewer_state_hash: str,
) -> str:
    return _digest(
        {
            "schema_version": schema_version,
            "revision": revision,
            "viewer": viewer,
            "viewer_state_hash": viewer_state_hash,
        }
    )


@dataclass(frozen=True, slots=True)
class BeliefState:
    """A normalized probability vector bound to one canonical world space."""

    space: PossibleWorldSpace
    model: str
    normalized_distribution: NDArray[np.float64]

    def __post_init__(self) -> None:
        values = np.asarray(self.normalized_distribution, dtype=np.float64).copy()
        if values.shape != (self.space.support_size,):
            raise BeliefError("belief must have one weight per canonical world")
        if np.any(~np.isfinite(values)):
            raise BeliefError("belief weights must be finite")
        if np.any(values < 0.0):
            raise BeliefError("belief weights must be non-negative")
        total = float(values.sum())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise BeliefError(f"belief is not normalized: total={total}")
        values.setflags(write=False)
        object.__setattr__(self, "normalized_distribution", values)

    @property
    def identity(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.space.identity.encode("ascii"))
        digest.update(b"\0")
        digest.update(self.model.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(self.normalized_distribution, dtype="<f8").tobytes())
        return digest.hexdigest()

    @property
    def normalization_error(self) -> float:
        return abs(float(self.normalized_distribution.sum()) - 1.0)

    @property
    def entropy(self) -> float:
        positive = self.normalized_distribution[self.normalized_distribution > 0.0]
        return -float(np.sum(positive * np.log(positive)))

    @property
    def effective_support(self) -> float:
        return 1.0 / float(np.square(self.normalized_distribution).sum())


@dataclass(frozen=True, slots=True)
class BeliefUpdateReceipt:
    model_identity: str
    previous_belief: str | None
    viewer_history_identity: str
    consumed_history_range: tuple[int, int]
    world_space_identity: str
    normalization_error: float
    output_digest: str

    @property
    def identity(self) -> str:
        return _digest(
            {
                "model_identity": self.model_identity,
                "previous_belief": self.previous_belief,
                "viewer_history_identity": self.viewer_history_identity,
                "consumed_history_range": self.consumed_history_range,
                "world_space_identity": self.world_space_identity,
                "normalization_error": self.normalization_error,
                "output_digest": self.output_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class BeliefUpdate:
    previous_belief: str | None
    viewer_history: str
    belief: BeliefState
    update_receipt: BeliefUpdateReceipt


class BeliefModel(Protocol):
    identity: str

    def update(
        self,
        *,
        previous: BeliefState | None,
        world_space: PossibleWorldSpace,
        viewer_history: ViewerHistory,
    ) -> BeliefUpdate: ...


@dataclass(frozen=True, slots=True)
class CompatibleDealBeliefModel:
    """Reference model that recomputes managym's compatible-deal measure."""

    identity: str = "compatible-deal-belief/v1"

    def update(
        self,
        *,
        previous: BeliefState | None,
        world_space: PossibleWorldSpace,
        viewer_history: ViewerHistory,
    ) -> BeliefUpdate:
        if viewer_history.viewer != world_space.viewer:
            raise BeliefError("viewer history does not match the world-space viewer")
        if viewer_history.current_revision != world_space.source_revision:
            raise BeliefError("viewer history revision does not match the world space")
        if (
            viewer_history.current_viewer_state_hash
            != world_space.source_viewer_state_hash
        ):
            raise BeliefError(
                "viewer history observation does not match the world space"
            )
        if previous is not None and previous.space.viewer != world_space.viewer:
            raise BeliefError("previous belief crossed a viewer boundary")
        weights = np.asarray(
            [world.weight for world in world_space.worlds], dtype=np.float64
        )
        if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
            raise BeliefError("compatible-deal weights must be finite and positive")
        distribution = weights / float(weights.sum())
        belief = BeliefState(world_space, self.identity, distribution)
        previous_identity = None if previous is None else previous.identity
        receipt = BeliefUpdateReceipt(
            model_identity=self.identity,
            previous_belief=previous_identity,
            viewer_history_identity=viewer_history.identity,
            consumed_history_range=(0, len(viewer_history.events)),
            world_space_identity=world_space.identity,
            normalization_error=belief.normalization_error,
            output_digest=belief.identity,
        )
        return BeliefUpdate(
            previous_belief=previous_identity,
            viewer_history=viewer_history.identity,
            belief=belief,
            update_receipt=receipt,
        )


@dataclass(frozen=True, slots=True)
class EmptyBeliefSupport:
    """Explicit result for a query carrying no probability mass."""

    belief_identity: str
    query_digest: str
    world_space_identity: str


def query_mass(belief: BeliefState, query: WorldQuery) -> float:
    """Measure a managym query under the canonical belief distribution."""

    indexes, _ = belief.space.condition_indexes(query, allow_empty=True)
    if not indexes:
        return 0.0
    selected = np.asarray(indexes, dtype=np.int64)
    return float(belief.normalized_distribution[selected].sum())


def condition_belief(
    belief: BeliefState, query: WorldQuery
) -> BeliefState | EmptyBeliefSupport:
    """Restrict and normalize a belief without consulting actual truth."""

    indexes, receipt = belief.space.condition_indexes(query, allow_empty=True)
    if not indexes:
        return EmptyBeliefSupport(
            belief_identity=belief.identity,
            query_digest=receipt.query_digest,
            world_space_identity=belief.space.identity,
        )
    selected = np.asarray(indexes, dtype=np.int64)
    mass = float(belief.normalized_distribution[selected].sum())
    if mass <= 0.0:
        return EmptyBeliefSupport(
            belief_identity=belief.identity,
            query_digest=receipt.query_digest,
            world_space_identity=belief.space.identity,
        )
    conditioned = np.zeros_like(belief.normalized_distribution)
    conditioned[selected] = belief.normalized_distribution[selected] / mass
    return BeliefState(belief.space, belief.model, conditioned)


__all__ = [
    "BeliefError",
    "BeliefModel",
    "BeliefState",
    "BeliefUpdate",
    "BeliefUpdateReceipt",
    "CompatibleDealBeliefModel",
    "EmptyBeliefSupport",
    "ViewerHistory",
    "condition_belief",
    "query_mass",
]
