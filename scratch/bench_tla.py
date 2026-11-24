"""TLA frontier (small) self-play settings for local measurement: CPU, v11 or v9."""

from experiments.tla_frontier import SMALL, FrontierConfig, settings
from manabot.training.loops.self_play import SelfPlay


def frontier_v9() -> SelfPlay:
    return settings(FrontierConfig(size=SMALL, device="cpu"))


def frontier_v11() -> SelfPlay:
    base = frontier_v9()
    return base.model_copy(
        update={
            "observation": base.observation.model_copy(
                update={"policy_input_version": 11}
            )
        }
    )
