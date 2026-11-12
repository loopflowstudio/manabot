"""Saved-regime execution: admission, runtime identities and `execute_regime`.

A saved `TrainingRegime` runs as one attempt of `manabot.training.run.Run`
under `attempt`'s writer lease; `stages.run_saved_regime` executes its stages
through the same stage bodies regime functions call. VerifyStore owns state;
manifests are exports.
"""

from collections.abc import Mapping
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.architecture import architecture_identity
from manabot.training.mps import mps_torch_error
from manabot.training.records import TrainingRun
from manabot.training.regime import (
    CheckpointCadence,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.rollout.matchups import elemental_roster
from manabot.training.run import attempt, host_identities, open_run
from manabot.training.stages import run_saved_regime
import managym

if TYPE_CHECKING:
    from manabot.verify.store import VerifyStore


def validate_regime(
    regime: TrainingRegime | Mapping[str, object], *, require_current_world: bool = True
) -> TrainingRegime:
    """Validate a recipe; execution additionally requires the native world.

    Offline authoring can resolve historical recipes without relabeling them.
    Every execution entry point retains the current-world check by default.
    """
    regime = TrainingRegime.model_validate(
        regime.model_dump() if isinstance(regime, TrainingRegime) else regime
    )
    if require_current_world and regime.world != managym.WORLD_VERSION:
        raise ValueError("regime world differs from native runtime")
    if regime.world not in {
        "w2",
        "w3",
        "w4",
        "w5",
        "w6",
        "w7",
        "w8",
        managym.WORLD_VERSION,
    }:
        raise ValueError("unsupported regime world")
    if any(
        isinstance(stage, TrainSelfPlay)
        and stage.matchup_curriculum == "elemental-uniform"
        for stage in regime.stages
    ):
        elemental_roster(Match(regime.root_match))
        if regime.agent.world != "elemental-world-v2":
            raise ValueError(
                "elemental-uniform requires complete Elemental policy input"
            )
    for stage in regime.stages:
        if isinstance(stage, TrainSelfPlay) and stage.adaptive is not None:
            if regime.recovery_max_microsteps is not None:
                raise ValueError("adaptive sampling requires current-game recovery")
            if regime.agent.world == "elemental-world-v2":
                for cell in stage.adaptive.cells:
                    elemental_roster(Match(cell))
    if regime.deck_source is not None:
        if require_current_world:
            regime.deck_source.check()
        if regime.agent.world != managym.WORLD_SETUP:
            raise ValueError(
                "a deck source requires the complete world policy vocabulary"
            )
    if regime.agent.belief_count_buckets:
        raise ValueError("regime stages do not produce belief inputs")
    return regime


def _runtime_identities(
    seed: int, regime: TrainingRegime, space: ObservationSpace
) -> dict[str, object]:
    identities = runtime_fingerprints(
        seed, match_config=regime.root_match, observation_space=space
    )
    if regime.deck_source is not None:
        # Opening the world fixes content and input identities; the run's setup
        # is the supply it samples, never the representative pair it opened.
        del identities["matchup"], identities["matchup_sha256"]
        identities.update(deck_source_sha256=regime.deck_source.digest)
    identities.update(
        architecture=architecture_identity(regime.agent, space), **host_identities()
    )
    if any(stage.execution.device == "cuda" for stage in regime.stages):
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA requested but unavailable; CPU fallback is forbidden"
            )
        gpu = torch.cuda.get_device_properties(0)
        identities["cuda"] = {
            "device": gpu.name,
            "memory_bytes": gpu.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
            "runtime": torch.version.cuda,
        }
    if any(stage.execution.device == "mps" for stage in regime.stages):
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable; CPU fallback is forbidden")
        if (error := mps_torch_error()) is not None:
            raise ValueError(error)
    return identities


def execute_regime(
    regime: TrainingRegime | Mapping[str, object],
    seed: int,
    out: str | Path,
    store: "VerifyStore",
    *,
    resume_from: str | None = None,
    source_repair_reason: str | None = None,
    checkpoint_seconds: float | None = None,
    checkpoint_updates: int | tuple[int, ...] | None = None,
    initial_admission: Path | None = None,
    pause_path: Path | None = None,
    stop_path: Path | None = None,
    stop_reserve_seconds: float | None = None,
    experiment: str | None = None,
    training_exposure: bool = False,
) -> TrainingRun:
    """Execute under a local lease, including admission and crash settlement.

    `stop_reserve_seconds` bounds a single step-target self-play stage by time
    as well as updates: at an update boundary, once the remaining wall budget
    cannot cover the slowest update so far plus this reserve, the stage stops,
    exports raw/EMA/Adam as usual and the run ends `paused` short of its
    target, so a successor can continue its complete learner state.

    Each self-play stage's metrics are an Etude run of `experiment`, or of an
    experiment named for the stage when it is None.
    """
    regime = validate_regime(regime)
    game_error_recovery = any(
        isinstance(stage, TrainSelfPlay) and stage.game_errors is not None
        for stage in regime.stages
    )
    if source_repair_reason is not None and (
        not resume_from or not source_repair_reason.strip() or not game_error_recovery
    ):
        raise ValueError(
            "source repair requires a reason, parent and complete-game error policy"
        )
    # Explicit milestones retain the historical multi-stage/hourly contract;
    # an integer is the prospective, single-stage absolute-update cadence.
    checkpoint_milestones = (
        checkpoint_updates if isinstance(checkpoint_updates, tuple) else ()
    )
    if isinstance(checkpoint_updates, tuple):
        checkpoint_updates = None
    if any(type(n) is not int or n <= 0 for n in checkpoint_milestones):
        raise ValueError("checkpoint milestones must be positive integers")
    if initial_admission is not None and (
        (checkpoint_seconds is None and checkpoint_updates is None)
        or resume_from is not None
        or len(regime.stages) != 1
        or not isinstance(regime.stages[0], TrainSelfPlay)
        or initial_admission.exists()
    ):
        raise ValueError("initial admission requires a fresh single self-play stage")
    if checkpoint_seconds is not None or checkpoint_updates is not None:
        CheckpointCadence(
            checkpoint_updates=checkpoint_updates, checkpoint_seconds=checkpoint_seconds
        ).admit_regime(regime)
    if (
        checkpoint_updates is not None
        and initial_admission is not None
        and any(
            isinstance(stage, TrainSelfPlay) and stage.learning_state is not None
            for stage in regime.stages
        )
    ):
        raise ValueError("step continuation cannot repeat initialization admission")
    if pause_path is not None and not regime.recoverable:
        raise ValueError("safe pause requires complete-state recovery")
    if stop_reserve_seconds is not None and (
        not math.isfinite(stop_reserve_seconds)
        or stop_reserve_seconds <= 0
        or len(regime.stages) != 1
        or not isinstance(regime.stages[0], TrainSelfPlay)
        or regime.stages[0].active_seconds is not None
        or stop_reserve_seconds
        >= min(regime.wall_seconds, regime.stages[0].execution.wall_seconds)
    ):
        raise ValueError(
            "a wall-bounded stop requires one step-target self-play stage"
            " and a positive reserve inside its watchdog"
        )
    digest = canonical_sha256(regime.model_dump(mode="json"))

    def identities() -> dict[str, object]:
        return _runtime_identities(seed, regime, ObservationSpace(regime.observation))

    with attempt(
        store,
        out,
        seed=seed,
        regime_digest=digest,
        identities=identities,
        resume_from=resume_from,
        source_repair_reason=source_repair_reason,
        cadence=(checkpoint_updates, checkpoint_seconds, checkpoint_milestones),
        game_error_recovery=game_error_recovery,
    ) as admitted:
        game_errors = next(
            (
                stage.game_errors
                for stage in reversed(regime.stages)
                if isinstance(stage, TrainSelfPlay) and stage.game_errors is not None
            ),
            None,
        )
        run = open_run(
            store,
            admitted.out,
            seed=seed,
            regime=regime,
            regime_function=None,
            regime_digest=digest,
            wall_seconds=regime.wall_seconds,
            recovery=regime.recovery,
            recoverable=regime.recoverable,
            game_errors=game_errors,
            parent=admitted.parent,
            snapshot=admitted.snapshot,
            source_repair_reason=admitted.source_repair_reason,
            checkpoint_seconds=checkpoint_seconds,
            checkpoint_updates=checkpoint_updates,
            checkpoint_milestones=checkpoint_milestones,
            pause_path=pause_path,
            stop_path=stop_path,
            stop_reserve_seconds=stop_reserve_seconds,
            initial_admission=initial_admission,
            experiment=experiment,
            training_exposure=training_exposure,
        )
        return run.execute(identities, lambda: run_saved_regime(run, regime))
