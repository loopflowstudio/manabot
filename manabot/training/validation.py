"""Admission checks for retained regime definitions, without executing them."""

from collections.abc import Mapping

from manabot.env.match import Match
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.training.rollout.matchups import elemental_roster
import managym


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
