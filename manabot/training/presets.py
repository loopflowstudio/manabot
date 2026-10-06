"""Versioned, complete experiment intentions; no mutable registry or latest alias."""

from pathlib import Path

from manabot.training.experiments import Baseline


def ataraxos_mtg_v1() -> Baseline:
    """Supported S3.4 move learning with a small MTG model; not a reproduction.

    The pinned file includes every serialized default. See docs/training-experiments.md
    for fidelity limits. New meanings require a new function and snapshot version.
    """
    baseline = Baseline(
        "ataraxos-mtg-v1",
        (Path(__file__).with_suffix("") / "ataraxos-mtg-v1.json").read_text(),
        origin="preset",
    )
    baseline.regime()
    return baseline
