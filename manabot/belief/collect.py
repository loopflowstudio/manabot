"""Step: collect hidden-hand histories from a frozen policy into one file.

`collect_belief` returns at once when its output exists, so a recipe resumed
after a failure skips it. The dataset is written atomically by `save_dataset`.
"""

from pathlib import Path

from manabot.belief.sampling_data import collect_frozen_policy, save_dataset
from manabot.infra.config import MatchConfig


def collect_belief(
    policy: Path,
    out: Path,
    *,
    match: MatchConfig,
    games: int,
    seed: int,
    max_steps: int = 2000,
) -> Path:
    """Play `games` complete games with the checkpoint `policy`; write `out`.

    Each viewer's history is kept with its private hand label. The policy is
    loaded and never changed; `load_checkpoint` refuses one from another world.
    """
    if out.exists():
        return out
    dataset = collect_frozen_policy(
        checkpoint=policy,
        match_config=match,
        games=games,
        seed=seed,
        max_steps=max_steps,
    )
    save_dataset(dataset, out)
    return out
