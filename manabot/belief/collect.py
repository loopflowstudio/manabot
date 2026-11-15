"""Collect hidden-hand histories from a frozen policy into an atomic dataset."""

from pathlib import Path

from manabot.belief.protocol import SamplerBackend
from manabot.belief.sampling_data import collect_frozen_policy, save_dataset
from manabot.infra.config import MatchConfig


def collect_belief(
    policy: Path,
    out: Path,
    *,
    match: MatchConfig,
    games: int,
    seed: int,
    backend: SamplerBackend,
    max_steps: int = 2000,
) -> Path:
    """Play `games` complete games with the checkpoint `policy`; write `out`.

    Each viewer's history is kept with its private hand label. The policy is
    loaded through `backend` and never changed; loading refuses one from another
    world.
    """
    dataset = collect_frozen_policy(
        checkpoint=policy,
        backend=backend,
        match_config=match,
        games=games,
        seed=seed,
        max_steps=max_steps,
    )
    save_dataset(dataset, out)
    return out
