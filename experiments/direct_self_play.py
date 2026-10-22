"""The learning-speed pair: direct PPO self-play against search distillation.

Both train the default model on the selected Allies/Lessons match for 900 seconds.

`uv run --extra notebook python -m experiments.direct_self_play --out .runs/learning-speed`
runs the learning-speed study (smoke by default; see `--help`).
"""

from manabot.arena.match import selected_match
from manabot.infra.hypers import AgentSpec
from manabot.training.builders import PACK, ppo_checkpoints
from manabot.training.models import (
    CollectSearch,
    Learning,
    TrainingRegime,
    TrainSupervised,
)
from manabot.training.regime_study import main
import managym


def direct_self_play() -> TrainingRegime:
    """Two two-update PPO checkpoints with every Ataraxos treatment enabled."""
    return TrainingRegime(
        id="direct-self-play",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=AgentSpec(semantic_pack=PACK),
        stages=ppo_checkpoints(2, Learning()),
        wall_seconds=900,
    )


def search_distillation() -> TrainingRegime:
    """Two rounds of search collection, each followed by supervised training."""
    return TrainingRegime(
        id="search-distillation",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=AgentSpec(semantic_pack=PACK),
        stages=[
            CollectSearch(id="data-0", operation="collect_search", games=2),
            TrainSupervised(
                id="policy-0",
                operation="train_supervised",
                datasets=["data-0"],
                epochs=2,
            ),
            CollectSearch(id="data-1", operation="collect_search", games=2),
            TrainSupervised(
                id="policy-1",
                operation="train_supervised",
                datasets=["data-0", "data-1"],
                initial="policy-0",
                epochs=2,
            ),
        ],
        wall_seconds=900,
    )


# Arm order is the study's; the first arm is the paired reference.
ARMS = (
    search_distillation,
    direct_self_play,
)

if __name__ == "__main__":
    main("learning-speed", ARMS)
