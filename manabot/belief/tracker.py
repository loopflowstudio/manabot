"""Belief tracking from canonical viewer observations and semantic receipts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Mapping, Protocol

from manabot.belief.likelihood import (
    LikelihoodResult,
    RulesProviderGap,
)
from manabot.belief.range import BeliefError, BeliefState
from managym.decision import Observation, SemanticTransition
from managym.possible_worlds import PossibleWorldSpace


class ActionLikelihood(Protocol):
    def evaluate(
        self,
        root_engine: Any,
        *,
        viewer: int,
        commitment: Mapping[str, Any],
        belief: BeliefState,
    ) -> LikelihoodResult: ...


@dataclass(slots=True)
class TrackerStats:
    updates: int = 0
    action_updates: int = 0
    hidden_draws: int = 0
    known_exits: int = 0
    known_returns: int = 0
    update_seconds: float = 0.0
    update_durations: list[float] = field(default_factory=list)
    likelihood_seconds: float = 0.0
    peak_range_bytes: int = 0
    peak_support_size: int = 0


@dataclass(frozen=True, slots=True)
class BeliefTransitionRecord:
    sequence: int
    observation_revision: int
    observation_hash: str
    before_space_id: str
    after_space_id: str
    command_id: str
    public_commitment: Mapping[str, Any] | None
    event_ids: tuple[str, ...]
    hidden_draws: int
    known_exits: tuple[str, ...]
    known_returns: tuple[str, ...]
    posterior_digest: str
    prior_digest: str
    posterior_normalization_error: float
    prior_normalization_error: float
    support_size: int
    positive_support_size: int
    effective_range_size: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "observation": {
                "revision": self.observation_revision,
                "viewer_state_hash": self.observation_hash,
            },
            "before_space_id": self.before_space_id,
            "after_space_id": self.after_space_id,
            "command_id": self.command_id,
            "public_commitment": dict(self.public_commitment)
            if self.public_commitment is not None
            else None,
            "event_ids": list(self.event_ids),
            "hidden_draws": self.hidden_draws,
            "known_exits": list(self.known_exits),
            "known_returns": list(self.known_returns),
            "posterior_digest": self.posterior_digest,
            "prior_digest": self.prior_digest,
            "posterior_normalization_error": self.posterior_normalization_error,
            "prior_normalization_error": self.prior_normalization_error,
            "support_size": self.support_size,
            "positive_support_size": self.positive_support_size,
            "effective_range_size": self.effective_range_size,
        }


class BeliefTracker:
    """Posterior plus matched compatible-prior control for one fixed viewer."""

    def __init__(
        self,
        space: PossibleWorldSpace,
        observation: Observation,
        *,
        likelihood: ActionLikelihood | None,
        epsilon: float,
        model_id: str,
    ) -> None:
        if observation.viewer != space.viewer:
            raise BeliefError("Observation viewer does not match PossibleWorldSpace")
        if observation.revision != space.source_revision:
            raise BeliefError("Observation revision does not match PossibleWorldSpace")
        if observation.viewer_state_hash != space.source_viewer_state_hash:
            raise BeliefError("Observation hash does not match PossibleWorldSpace")
        self.viewer = space.viewer
        self.likelihood = likelihood
        self.epsilon = epsilon
        self.space = space
        self.observation = observation
        self.posterior = BeliefState.compatible_prior(space, model_id=model_id)
        self.prior = BeliefState.compatible_prior(space)
        self.stats = TrackerStats()
        self.initial_space_id = space.identity
        self.initial_observation = observation
        self.initial_posterior_digest = self.posterior.digest
        self.records: list[BeliefTransitionRecord] = []
        self._pending_public_commitment: Mapping[str, Any] | None = None
        self._record_size()

    @classmethod
    def from_engine(
        cls,
        engine: Any,
        *,
        viewer: int,
        likelihood: ActionLikelihood | None,
        epsilon: float,
    ) -> "BeliefTracker":
        space = PossibleWorldSpace.from_engine(engine, viewer)
        observation = Observation.from_json(engine.semantic_observation_json(viewer))
        checkpoint = getattr(likelihood, "checkpoint_sha256", None)
        model_id = (
            f"frozen-policy-likelihood/sha256:{checkpoint}"
            if checkpoint
            else "test-only-likelihood/v1"
        )
        return cls(
            space,
            observation,
            likelihood=likelihood,
            epsilon=epsilon,
            model_id=model_id,
        )

    def observe(
        self,
        after_engine: Any,
        *,
        acting: int,
        transition: SemanticTransition,
        likelihood_root: Any | None = None,
    ) -> None:
        started = time.perf_counter()
        before_space = self.space
        commitment = transition.receipt.public_commitment
        if acting != self.viewer and commitment is not None:
            if self.likelihood is None or likelihood_root is None:
                raise BeliefError(
                    "opponent commitment requires a likelihood model and retained root"
                )
            likelihood = self.likelihood.evaluate(
                likelihood_root,
                viewer=self.viewer,
                commitment=commitment,
                belief=self.posterior,
            )
            self.posterior = self.posterior.condition_likelihood(
                likelihood.likelihoods,
                likelihood.legal_action_counts,
                likelihood.matching_action_counts,
                epsilon=self.epsilon,
            )
            self.stats.action_updates += 1
            self.stats.likelihood_seconds += likelihood.seconds
            if commitment.get("kind") in {"cast", "play_land"}:
                self._pending_public_commitment = commitment
            elif commitment.get("kind") == "pass_priority":
                self._pending_public_commitment = None

        next_observation = Observation.from_json(
            after_engine.semantic_observation_json(self.viewer)
        )
        next_space = PossibleWorldSpace.from_engine(after_engine, self.viewer)
        transport_commitment = (
            commitment or self._pending_public_commitment
            if acting != self.viewer
            else None
        )
        known_exits, known_returns, hidden_draws = self._public_transport_facts(
            before_space, next_space, transport_commitment
        )
        if known_exits:
            self._pending_public_commitment = None
        self.posterior = self.posterior.transport(
            next_space,
            known_exits=known_exits,
            known_returns=known_returns,
            hidden_draws=hidden_draws,
        )
        self.prior = BeliefState.compatible_prior(next_space)
        self.space = next_space
        self.observation = next_observation
        self.stats.hidden_draws += hidden_draws
        self.stats.known_exits += len(known_exits)
        self.stats.known_returns += len(known_returns)
        self.stats.updates += 1
        elapsed = time.perf_counter() - started
        self.stats.update_seconds += elapsed
        self.stats.update_durations.append(elapsed)
        self._record_size()
        self.records.append(
            BeliefTransitionRecord(
                sequence=len(self.records),
                observation_revision=next_observation.revision,
                observation_hash=next_observation.viewer_state_hash,
                before_space_id=before_space.identity,
                after_space_id=next_space.identity,
                command_id=transition.receipt.command_id,
                public_commitment=commitment if acting != self.viewer else None,
                event_ids=transition.receipt.events,
                hidden_draws=hidden_draws,
                known_exits=tuple(known_exits),
                known_returns=tuple(known_returns),
                posterior_digest=self.posterior.digest,
                prior_digest=self.prior.digest,
                posterior_normalization_error=self.posterior.normalization_error,
                prior_normalization_error=self.prior.normalization_error,
                support_size=self.posterior.support_size,
                positive_support_size=self.posterior.positive_support_size,
                effective_range_size=self.posterior.effective_range_size,
            )
        )

    @staticmethod
    def _public_transport_facts(
        before: PossibleWorldSpace,
        after: PossibleWorldSpace,
        commitment: Mapping[str, Any] | None,
    ) -> tuple[list[str], list[str], int]:
        before_pool = dict(before.pool)
        after_pool = dict(after.pool)
        names = sorted(set(before_pool) | set(after_pool))
        exits = [
            name
            for name in names
            for _ in range(max(0, before_pool.get(name, 0) - after_pool.get(name, 0)))
        ]
        returns = [
            name
            for name in names
            for _ in range(max(0, after_pool.get(name, 0) - before_pool.get(name, 0)))
        ]
        if exits:
            if commitment is None or commitment.get("kind") not in {
                "cast",
                "play_land",
            }:
                raise RulesProviderGap(
                    "hidden-pool exit has no canonical public commitment identity"
                )
            card = str(commitment.get("card", ""))
            if exits != [card]:
                raise RulesProviderGap(
                    "public commitment card does not match canonical pool change"
                )
        hidden_draws = after.hand_size - before.hand_size + len(exits) - len(returns)
        if hidden_draws < 0:
            raise BeliefError("canonical space change has an unexplained hand exit")
        return exits, returns, hidden_draws

    def diagnostics(self) -> dict[str, Any]:
        return {
            "space_id": self.space.identity,
            "observation_revision": self.observation.revision,
            "observation_hash": self.observation.viewer_state_hash,
            "model_id": self.posterior.model_id,
            "support_size": self.posterior.support_size,
            "positive_support_size": self.posterior.positive_support_size,
            "effective_range_size": self.posterior.effective_range_size,
            "normalization_error": self.posterior.normalization_error,
            "prior_normalization_error": self.prior.normalization_error,
            "range_bytes": self.posterior.allocated_bytes,
            "probability_bytes": self.posterior.probability_bytes,
            "world_space_bytes": self.posterior.space.allocated_bytes,
            "digest": self.posterior.digest,
            "prior_digest": self.prior.digest,
        }

    def replay_receipt(self) -> dict[str, Any]:
        payload = {
            "schema_version": 2,
            "viewer": self.viewer,
            "initial_observation": {
                "revision": self.initial_observation.revision,
                "viewer_state_hash": self.initial_observation.viewer_state_hash,
            },
            "initial_space_id": self.initial_space_id,
            "initial_posterior_digest": self.initial_posterior_digest,
            "transitions": [record.to_dict() for record in self.records],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["history_digest"] = hashlib.sha256(canonical.encode()).hexdigest()
        return payload

    def _record_size(self) -> None:
        self.stats.peak_range_bytes = max(
            self.stats.peak_range_bytes,
            self.posterior.probability_bytes
            + self.prior.probability_bytes
            + self.posterior.space.allocated_bytes,
        )
        self.stats.peak_support_size = max(
            self.stats.peak_support_size, self.posterior.support_size
        )


__all__ = [
    "ActionLikelihood",
    "BeliefTracker",
    "BeliefTransitionRecord",
    "TrackerStats",
]
