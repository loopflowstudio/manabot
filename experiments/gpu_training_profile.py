"""The large model whose CUDA training-pipeline costs are profiled."""

from manabot.arena.match import selected_match
from manabot.infra.config import ModelConfig
from manabot.training.builders import PACK, move_learning
from manabot.training.models import Execution, TrainingRegime, TrainSelfPlay
import managym


def gpu_training_profile() -> TrainingRegime:
    """Width 384, depth 8 with a value token: 512 CUDA updates of 64 streams."""
    return TrainingRegime(
        id="gpu-training-profile",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=ModelConfig(
            semantic_pack=PACK,
            value_aggregation="value_token",
            attention_layers=8,
            attention_feedforward_dim=1536,
            hidden_dim=384,
        ),
        stages=[
            TrainSelfPlay(
                id="policy",
                operation="train_self_play",
                updates=512,
                streams=64,
                transitions=8,
                learning=move_learning(),
                execution=Execution(device="cuda", wall_seconds=4500),
            )
        ],
        wall_seconds=4560,
        schedule_clock="iteration_fraction",
    )
