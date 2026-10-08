"""Admit portable raw/Adam/EMA state from a hash-bound TrainingRun export.

Admission retains original bytes, validates ordinary policy/world contracts and
the unchanged learning recipe, and derives absolute counters from the producer.
This is a new segment with fresh game/RNG streams, never process recovery.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from manabot.arena.models import canonical_sha256
from manabot.env import Match
from manabot.model.agent import Agent
from manabot.model.world import validate_agent_setup
from manabot.sim.flat_mc import load_checkpoint_agent

from .admission import copy_verified_artifact
from .models import (
    ArtifactReference,
    LearningStateOrigin,
    StageRecord,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)


@dataclass(frozen=True)
class AdmittedLearningState:
    artifacts: dict[str, ArtifactReference]
    origin: LearningStateOrigin
    raw: Agent
    ema: Agent
    # Adam's nested torch serialization boundary; validated before application.
    optimizer: dict[str, Any]
    seed: int


def _origin(source: TrainingRun, record: StageRecord) -> LearningStateOrigin:
    previous = record.learning_state_origin
    start = previous.iteration if previous is not None else 0
    if not record.diagnostics or any(
        diagnostic.get("iteration") != start + offset
        for offset, diagnostic in enumerate(record.diagnostics, 1)
    ):
        raise ValueError("learning-state parent lacks contiguous absolute iterations")
    if record.cumulative_seconds is None:
        raise ValueError("learning-state parent cost is unavailable")
    return LearningStateOrigin(
        run_id=source.id,
        stage_id=record.id,
        iteration=start + len(record.diagnostics),
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


def validate_adam(optimizer: torch.optim.Adam) -> None:
    """Reject absent/corrupt moments; unused parameters may legitimately lack state."""
    if not optimizer.state:
        raise ValueError("learning-state Adam has no populated state")
    for parameter, state in optimizer.state.items():
        if not isinstance(parameter, torch.Tensor) or not isinstance(state, dict):
            raise ValueError("invalid Adam parameter state")
        step = state.get("step")
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not torch.isfinite(step).all()
            or step.item() <= 0
            or step.item() != int(step.item())
        ):
            raise ValueError("invalid Adam step counter")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or not torch.isfinite(value).all()
                or (name == "exp_avg_sq" and (value < 0).any())
            ):
                raise ValueError(f"invalid Adam moment: {name}")


def admit_learning_state(
    stage: TrainSelfPlay, regime: TrainingRegime, out: Path
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
    if (
        not isinstance(definition, TrainSelfPlay)
        or definition.id != spec.source_stage
        or record.id != spec.source_stage
        or record.status not in {"completed", "paused"}
        or source.status not in {"completed", "paused"}
        or source.parent_run_id is not None
    ):
        raise ValueError("learning-state parent must have a completed or paused export")
    if (
        source.regime.world != regime.world
        or source.regime.agent != regime.agent
        or source.regime.observation != regime.observation
        or source.regime.match != regime.match
        or source.regime.schedule_clock != regime.schedule_clock
    ):
        raise ValueError("learning-state model/world/setup/schedule mismatch")
    for name in ("learning", "streams", "transitions", "behavior", "opponent"):
        if getattr(definition, name) != getattr(stage, name):
            raise ValueError(f"learning-state recipe mismatch: {name}")
    origin = _origin(source, record)
    if origin.iteration >= stage.updates:
        raise ValueError("learning-state target must exceed parent absolute iteration")
    artifacts = {"parent_run": receipt}
    policies: dict[str, Agent] = {}
    for role in ("raw", "ema", "optimizer"):
        reference = getattr(spec, role)
        published = record.artifacts.get(role)
        if published is None or any(
            published.get(key) != reference[key] for key in ("sha256", "bytes")
        ):
            raise ValueError(f"learning-state {role} differs from parent artifact")
        copied = copy_verified_artifact(reference, out / f"{stage.id}-parent-{role}.pt")
        artifacts[f"parent_{role}"] = copied
        if role == "optimizer":
            continue
        with torch.random.fork_rng(devices=[]):
            agent, space = load_checkpoint_agent(copied["path"])
        validate_agent_setup(agent, Match(regime.match).to_rust())
        if agent.hypers != regime.agent or space.encoder.hypers != regime.observation:
            raise ValueError("learning-state checkpoint architecture mismatch")
        payload = torch.load(copied["path"], map_location="cpu", weights_only=False)
        # Adam stores positional parameter IDs. The historical checkpoint's
        # ordered state dictionary must agree with today's parameter ordering.
        parameter_names = list(dict(agent.named_parameters()))
        if [
            name for name in payload["model_state_dict"] if name in parameter_names
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
        if any(
            not torch.isfinite(value).all() for value in agent.state_dict().values()
        ):
            raise ValueError("learning-state policy contains nonfinite tensors")
        policies[role] = agent
    state = torch.load(
        artifacts["parent_optimizer"]["path"], map_location="cpu", weights_only=True
    )
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("invalid Adam artifact")
    optimizer = torch.optim.Adam(policies["raw"].parameters(), eps=1e-5)
    groups = state["param_groups"]
    expected_group = optimizer.state_dict()["param_groups"][0]
    if (
        not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], dict)
        or groups[0].get("params") != expected_group["params"]
        or not isinstance(state["state"], dict)
        or not set(state["state"]).issubset(expected_group["params"])
        or any(
            groups[0].get(key) != expected_group[key]
            for key in (
                "betas",
                "eps",
                "weight_decay",
                "amsgrad",
                "maximize",
                "capturable",
                "differentiable",
            )
        )
    ):
        raise ValueError("learning-state Adam parameter groups or settings mismatch")
    optimizer.load_state_dict(state)
    validate_adam(optimizer)
    return AdmittedLearningState(
        artifacts, origin, policies["raw"], policies["ema"], state, source.seed
    )
