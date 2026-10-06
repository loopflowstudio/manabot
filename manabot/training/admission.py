"""Published-policy admission into the existing TrainingRun artifact graph.

ImportPolicy pins a producer's exported TrainingRun and checkpoint. Admission
copies and hashes both before loading, then uses ordinary model/world validation.
The retained producer receipt owns provenance and sunk costs; StageRecord timings
measure only fresh admission. No optimizer or live collector state is imported.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import torch

from manabot.arena.models import canonical_sha256, file_sha256
from manabot.env import Match
from manabot.model.world import validate_agent_setup
from manabot.sim.flat_mc import load_checkpoint_agent

from .models import (
    ArtifactReference,
    ImportPolicy,
    ProducerCost,
    TrainCompound,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
    TrainSupervised,
)


@dataclass(frozen=True)
class AdmittedPolicy:
    checkpoint: ArtifactReference
    source_run: ArtifactReference
    producer_cost: ProducerCost


def _copy_verified(source: ArtifactReference, target: Path) -> ArtifactReference:
    """Pin copied bytes rather than trusting a mutable source pathname."""
    if target.exists():
        raise ValueError(f"import destination already exists: {target}")
    shutil.copyfile(source["path"], target)
    if (
        file_sha256(target) != source["sha256"]
        or target.stat().st_size != source["bytes"]
    ):
        raise ValueError(f"published artifact hash/size mismatch: {source['path']}")
    return {"path": str(target), "sha256": source["sha256"], "bytes": source["bytes"]}


def admit_policy(
    stage: ImportPolicy, regime: TrainingRegime, out: Path
) -> AdmittedPolicy:
    """Validate a completed producer checkpoint, retaining its unmodified export.

    Publication is a caller-supplied immutable receipt, not a promotion decision.
    Incomplete runs can supply a completed stage; missing source identity or cost
    cannot be replaced by a zero-cost synthetic producer. Legacy checkpoints
    without TrainingRun provenance are deliberately unsupported.
    """
    receipt = _copy_verified(stage.source_run, out / f"{stage.id}-source-run.json")
    serialized = json.loads(Path(receipt["path"]).read_bytes())
    if not isinstance(serialized, dict) or serialized.get(
        "regime_digest"
    ) != canonical_sha256(serialized.get("regime")):
        raise ValueError("published producer regime digest mismatch")
    # Hash original evidence before current defaults normalize older recipes.
    source = TrainingRun.model_validate(serialized)
    if any(isinstance(s, ImportPolicy) for s in source.regime.stages) or any(
        s.producer_cost is not None for s in source.stages
    ):
        raise ValueError("nested published producer costs are unsupported")
    if not source.identities.get("source_commit") or not source.identities.get(
        "training_source_sha256"
    ):
        raise ValueError("published producer lacks source provenance")
    definition = next(
        (s for s in source.regime.stages if s.id == stage.source_stage), None
    )
    record = next((s for s in source.stages if s.id == stage.source_stage), None)
    if (
        not isinstance(definition, (TrainSelfPlay, TrainSupervised, TrainCompound))
        or record is None
        or record.status != "completed"
    ):
        raise ValueError("published policy requires a completed producer policy stage")
    if record.cumulative_seconds is None:
        raise ValueError("published producer cost is unavailable")
    published = record.artifacts.get(stage.weights)
    if published is None or any(
        published.get(key) != stage.checkpoint[key] for key in ("sha256", "bytes")
    ):
        raise ValueError("published checkpoint does not match producer weight artifact")
    if (
        source.regime.world != regime.world
        or source.regime.agent != regime.agent
        or source.regime.observation != regime.observation
    ):
        raise ValueError(
            "published producer model/observation/world differs from regime"
        )
    admitted = _copy_verified(stage.checkpoint, out / f"{stage.id}-{stage.weights}.pt")
    # Ordinary admission owns architecture, state keys and native world meaning.
    # Fork RNG so validation never shifts downstream initialization streams.
    with torch.random.fork_rng(devices=[]):
        agent, space = load_checkpoint_agent(admitted["path"])
    if agent.hypers != regime.agent or space.encoder.hypers != regime.observation:
        raise ValueError("published checkpoint model/observation differs from regime")
    validate_agent_setup(agent, Match(source.regime.match).to_rust())
    validate_agent_setup(agent, Match(regime.match).to_rust())
    payload = torch.load(admitted["path"], map_location="cpu", weights_only=False)
    metadata = payload.get("bc", {})
    for key, expected in {
        "run_id": source.id,
        "stage_id": stage.source_stage,
        "regime_digest": source.regime_digest,
        "weights": stage.weights,
    }.items():
        if metadata.get(key) != expected:
            raise ValueError(f"published checkpoint producer metadata mismatch: {key}")
    expected_value = (
        "win_logit"
        if isinstance(definition, TrainSupervised)
        and not definition.target.startswith("local_")
        else "signed_outcome"
    )
    if metadata.get("value_semantic") != expected_value:
        raise ValueError("published checkpoint value semantics mismatch")
    if stage.weights == "ema":
        averaging = metadata.get("averaging")
        if (
            not isinstance(definition, TrainSelfPlay)
            or definition.learning.ema is None
            or not isinstance(averaging, dict)
            or averaging.get("clock") != "collect-update-iteration"
            or averaging.get("rate") != definition.learning.ema
            or not isinstance(averaging.get("iteration"), int)
            or averaging["iteration"] < 1
        ):
            raise ValueError("published EMA averaging identity mismatch")
    elif metadata.get("averaging") is not None:
        raise ValueError("published raw checkpoint declares averaged weights")
    if expected_value != "signed_outcome" and any(
        isinstance(s, TrainSupervised)
        and s.initial == stage.id
        and s.target.startswith("local_")
        for s in regime.stages
    ):
        raise ValueError("local distillation requires signed-outcome initial weights")
    return AdmittedPolicy(
        checkpoint=admitted,
        source_run=receipt,
        producer_cost=ProducerCost(
            run_id=source.id,
            stage_id=stage.source_stage,
            weights=stage.weights,
            cumulative_seconds=record.cumulative_seconds,
        ),
    )
