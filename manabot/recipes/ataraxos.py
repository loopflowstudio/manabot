"""Small current-world Ataraxos-inspired experiment, not a historical replay."""

from manabot.experiment.config import Baseline
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.recipes.builders import PACK, ataraxos_baseline
from manabot.sim.matchup import selected_match
import managym


def ataraxos() -> Baseline:
    """Capture a fresh categorical-WDL recipe; historical w4 inputs need pinned source."""
    return Baseline.capture(
        "ataraxos",
        ataraxos_baseline(
            id="ataraxos",
            world=managym.WORLD_VERSION,
            match=selected_match(),
            observation=ObservationConfig(),
            agent=ModelConfig(
                world=PACK,
                value_kind="categorical_wdl",
                value_aggregation="value_token",
                attention_layers=2,
            ),
            checkpoints=2,
            updates=1,
            transitions=64,
            streams=4,
            stage_seconds=80,
            wall_seconds=180,
        ),
    )
