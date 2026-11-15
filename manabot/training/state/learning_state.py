"""Admit portable raw/Adam/EMA state from a hash-bound TrainingRun export.

Admission retains original bytes, validates ordinary policy/world contracts and
the unchanged learning recipe, and derives absolute counters from the producer.
This is a new segment with fresh game/RNG streams, never process recovery. A
declared setup transfer lets the segment continue on different decks from the
same world (a curriculum); nothing else about the recipe may change.

A parent that was interrupted before exporting its final state can still be
continued from one of its monitoring exports (`monitoring_update`): the raw
bytes must be the ones its record's receipt at that update names, the origin
takes that receipt's absolute coordinates, and the EMA and Adam restart (EMA :=
raw, fresh moments). The reset is recorded in the origin as a declared
discontinuity, never inferred.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from manabot.env.match import Match
from manabot.infra.config import MatchConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.world import validate_agent_setup
from manabot.policy.protocol import BackendPolicy
from manabot.training.records import StageRecord, TrainingRun
from manabot.training.regime import (
    ArtifactReference,
    LearningStateOrigin,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.state.admission import admit_setups, copy_verified_artifact
from manabot.training.state.backend import TrainingBackend


@dataclass(frozen=True)
class AdmittedLearningState:
    artifacts: dict[str, ArtifactReference]
    origin: LearningStateOrigin
    raw: BackendPolicy
    ema: BackendPolicy
    # Adam's nested torch serialization boundary; validated before application.
    # None when a raw-only import starts Adam afresh.
    optimizer: dict[str, Any] | None
    seed: int


def _origin(source: TrainingRun, record: StageRecord) -> LearningStateOrigin:
    previous = record.learning_state_origin
    start = previous.iteration if previous is not None else 0
    if not record.updates:
        raise ValueError("learning-state parent lacks completed updates")
    if record.cumulative_seconds is None:
        raise ValueError("learning-state parent cost is unavailable")
    optimizer_updates = record.optimizer_updates
    if previous is not None:
        optimizer_updates = (
            optimizer_updates + previous.optimizer_updates
            if optimizer_updates is not None and previous.optimizer_updates is not None
            else None
        )
    return LearningStateOrigin(
        run_id=source.id,
        stage_id=record.id,
        iteration=start + record.updates,
        optimizer_updates=optimizer_updates,
        **{
            name: getattr(record, name)
            + (getattr(previous, name) if previous is not None else 0)
            for name in (
                "games",
                "environment_decisions",
                "learner_transitions",
                "optimizer_exposures",
            )
        },
        active_training_seconds=record.collection_seconds
        + record.learning_seconds
        + (previous.active_training_seconds if previous is not None else 0),
        cumulative_seconds=record.cumulative_seconds
        + (previous.cumulative_seconds if previous is not None else 0),
    )


def _monitoring_origin(
    source: TrainingRun, record: StageRecord, update: int, raw: ArtifactReference
) -> LearningStateOrigin:
    """Absolute coordinates of the one admitted monitoring export at `update`
    whose bytes are `raw`; its counters already include any origin the parent
    inherited."""
    found = [
        c
        for c in source.monitoring_checkpoints
        if c.stage_id == record.id and c.updates == update
    ]
    if len(found) != 1:
        raise ValueError(f"parent has no single monitoring export at update {update}")
    receipt = found[0]
    if (
        receipt.error is not None
        or receipt.artifact is None
        or any(receipt.artifact.get(key) != raw[key] for key in ("sha256", "bytes"))
    ):
        raise ValueError(
            "learning-state raw differs from the parent's monitoring export"
        )
    previous = record.learning_state_origin
    if receipt.updates <= (previous.iteration if previous is not None else 0):
        raise ValueError("monitoring export precedes the parent's own updates")
    if receipt.active_training_seconds is None:
        raise ValueError("monitoring export lacks active training seconds")
    return LearningStateOrigin(
        run_id=source.id,
        stage_id=record.id,
        iteration=receipt.updates,
        optimizer_updates=receipt.optimizer_updates,
        games=receipt.games,
        environment_decisions=receipt.environment_decisions,
        learner_transitions=receipt.learner_transitions,
        optimizer_exposures=receipt.optimizer_exposures,
        active_training_seconds=receipt.active_training_seconds,
        cumulative_seconds=receipt.training_seconds,
        reset=("ema", "optimizer"),
    )


def _admit_setup_transfer(source: MatchConfig, regime: TrainingRegime) -> None:
    """A deck switch stays inside the agent's world and changes decks."""
    world = regime.agent.world
    if regime.match is None:
        raise ValueError("setup transfer requires an authored match, not a deck source")
    if source == regime.match:
        raise ValueError("setup transfer must change the match")
    if source.world != world or regime.match.world != world:
        raise ValueError("setup transfer must stay inside the agent's world")


def admit_learning_state(
    stage: TrainSelfPlay, regime: TrainingRegime, out: Path, *, backend: TrainingBackend
) -> AdmittedLearningState:
    """Verify relocated copies against an unmodified historical producer export."""
    spec = stage.learning_state
    if spec is None:
        raise ValueError("learning-state input is required")
    receipt = copy_verified_artifact(
        spec.source_run, out / f"{stage.id}-parent-run.json"
    )
    serialized = json.loads(Path(receipt["path"]).read_bytes())
    if not isinstance(serialized, dict) or serialized.get(
        "regime_digest"
    ) != canonical_sha256(serialized.get("regime")):
        raise ValueError("learning-state producer regime digest mismatch")
    source = TrainingRun.model_validate(serialized)
    if not source.identities.get("source_commit") or not source.identities.get(
        "training_source_sha256"
    ):
        raise ValueError("learning-state parent lacks source provenance")
    if len(source.regime.stages) != 1 or len(source.stages) != 1:
        raise ValueError("learning-state parent requires a single self-play stage")
    definition, record = source.regime.stages[0], source.stages[0]
    # A monitoring export was published while training ran, so its parent may
    # have been interrupted afterwards; final exports need a finished parent.
    ended = {"completed", "paused"}
    if spec.monitoring_update is not None:
        ended |= {"interrupted"}
    if (
        not isinstance(definition, TrainSelfPlay)
        or definition.id != spec.source_stage
        or record.id != spec.source_stage
        or record.status not in ended
        or source.status not in ended
        or source.parent_run_id is not None
    ):
        raise ValueError("learning-state parent must have a completed or paused export")
    source_match = regime.match
    if spec.source_match is not None:
        _admit_setup_transfer(spec.source_match, regime)
        source_match = spec.source_match
    if (
        source.regime.world != regime.world
        or source.regime.agent != regime.agent
        or source.regime.observation != regime.observation
        or source.regime.deck_source != regime.deck_source
        or source.regime.match != source_match
        or source.regime.schedule_clock != regime.schedule_clock
    ):
        raise ValueError("learning-state model/world/setup/schedule mismatch")
    for name in (
        "learning",
        "streams",
        "transitions",
        "behavior",
        "opponent",
        "learning_seats",
        "batch_closure",
        "games_per_stream",
    ):
        if getattr(definition, name) != getattr(stage, name):
            raise ValueError(f"learning-state recipe mismatch: {name}")
    origin = (
        _origin(source, record)
        if spec.monitoring_update is None
        else _monitoring_origin(source, record, spec.monitoring_update, spec.raw)
    )
    if origin.iteration >= stage.updates:
        raise ValueError("learning-state target must exceed parent absolute iteration")
    artifacts = {"parent_run": receipt}
    policies: dict[str, BackendPolicy] = {}
    for role in spec.roles:
        reference = getattr(spec, role)
        if spec.monitoring_update is None:
            published = record.artifacts.get(role)
            if published is None or any(
                published.get(key) != reference[key] for key in ("sha256", "bytes")
            ):
                raise ValueError(f"learning-state {role} differs from parent artifact")
        copied = copy_verified_artifact(reference, out / f"{stage.id}-parent-{role}.pt")
        artifacts[f"parent_{role}"] = copied
        if role == "optimizer":
            continue
        with backend.state.preserve_rng():
            agent, space = backend.load_policy(copied["path"])
        # The parent export is bound to what it trained on: a transferred setup's
        # match, else the regime's match or every deck its deck source lists. The
        # segment's own exports are bound to the regime when they are written.
        if spec.source_match is not None:
            validate_agent_setup(agent, Match(source_match).to_rust())
        else:
            admit_setups(agent, regime)
        if agent.config != regime.agent or space.encoder.config != regime.observation:
            raise ValueError("learning-state checkpoint architecture mismatch")
        payload = backend.metadata(copied["path"])
        # Adam stores positional parameter IDs. The historical checkpoint's
        # ordered state dictionary must agree with today's parameter ordering.
        parameter_names = backend.state.parameter_names(agent)
        if [
            name for name in payload["parameter_names"] if name in parameter_names
        ] != parameter_names:
            raise ValueError("learning-state parameter order mismatch")
        metadata = payload.get("bc", {})
        for key, expected in {
            "run_id": source.id,
            "stage_id": record.id,
            "regime_digest": source.regime_digest,
            "weights": role,
            "value_semantic": "signed_outcome",
        }.items():
            if metadata.get(key) != expected:
                raise ValueError(f"learning-state checkpoint metadata mismatch: {key}")
        averaging = metadata.get("averaging")
        if role == "ema" and averaging != {
            "clock": "collect-update-iteration",
            "iteration": origin.iteration,
            "rate": stage.learning.ema,
        }:
            raise ValueError("learning-state EMA iteration/rate mismatch")
        if role == "raw" and averaging is not None:
            raise ValueError("learning-state raw policy cannot be averaged")
        backend.state.validate_policy(agent)
        policies[role] = agent
    if spec.monitoring_update is not None:
        # Declared reset: the EMA restarts as the raw weights at the origin
        # iteration and Adam starts with no moments.
        return AdmittedLearningState(
            artifacts,
            origin,
            policies["raw"],
            backend.state.clone_policy(policies["raw"]),
            None,
            source.seed,
        )
    state = backend.state.admit_adam(
        artifacts["parent_optimizer"]["path"], policies["raw"]
    )
    return AdmittedLearningState(
        artifacts, origin, policies["raw"], policies["ema"], state, source.seed
    )
