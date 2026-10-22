"""Search-generated observation/target shards and their admission contract.

Rows retain game and seat identity, legal masks, chosen actions, and the declared
teacher targets. Loading never fits a model or chooses train/validation games.
"""

import json
from pathlib import Path

import numpy as np

from manabot.env import ObservationSpace
from manabot.infra.config import ObservationConfig

OBS_KEYS: tuple[str, ...] = tuple(ObservationSpace().shapes.keys())
META_KEYS = ("action", "game_index", "seat", "num_valid", "winner")
# "scores" (D, max_actions) float32 — raw flat-MC win-probability estimates
# per action (engine order), -1.0 padding on invalid slots. Present in shards
# generated from exp-07 onward; loaders treat it as optional.
SCORE_KEY = "scores"
# Tree-search teachers write per-edge root visit counts and root values under
# these stable columns. Legacy flat-MC shards omit them.
VISIT_COUNT_KEY = "visit_counts"
ROOT_VALUE_KEY = "root_value"
LOCAL_TARGET_KEY = "local_target"
LOCAL_RECEIPT_KEY = "local_receipt"
# Admission columns: the rules engine's own legal-offer count and decision
# kind (managym.ActionSpaceEnum) at each decision, recorded independently of
# the encoded mask so an omitted legal choice is measurable afterwards.
NUM_LEGAL_KEY = "num_legal"
DECISION_KIND_KEY = "decision_kind"


def load_shards(
    paths: list[str | Path],
    *,
    rounds: list[int] | None = None,
    globally_unique_games: bool = False,
) -> dict[str, np.ndarray]:
    """Load and concatenate dataset shards.

    Adds a per-decision "round" column (expert-iteration staleness
    diagnostic): taken from ``rounds`` (one per shard) when given, else from
    each shard's embedded provenance tag, else -1 for legacy shards.
    """

    shards = [np.load(Path(p)) for p in paths]
    keys = list(OBS_KEYS) + list(META_KEYS)
    for optional_key in (
        SCORE_KEY,
        VISIT_COUNT_KEY,
        ROOT_VALUE_KEY,
        NUM_LEGAL_KEY,
        DECISION_KIND_KEY,
        LOCAL_TARGET_KEY,
        LOCAL_RECEIPT_KEY,
    ):
        if all(optional_key in shard for shard in shards):
            keys.append(optional_key)
    out = {key: np.concatenate([s[key] for s in shards]) for key in keys}

    round_cols = []
    for i, shard in enumerate(shards):
        if rounds is not None:
            shard_round = int(rounds[i])
        elif "provenance" in shard:
            shard_round = int(json.loads(str(shard["provenance"])).get("round", -1))
        else:
            shard_round = -1
        round_cols.append(np.full(len(shard["action"]), shard_round, dtype=np.int16))
    out["round"] = np.concatenate(round_cols)

    # Game indices are only unique within a round's shard set; offset them
    # per round so the by-game train/val split never merges games across
    # rounds.
    unique_rounds = np.unique(out["round"])
    if globally_unique_games:
        # TrainingRun assigns global game offsets when collecting each shard.
        # Re-offsetting those IDs as datasets grow changes split membership.
        seen: set[int] = set()
        for shard in shards:
            games = set(int(g) for g in shard["game_index"])
            if seen.intersection(games):
                raise ValueError("globally unique shard game IDs overlap")
            seen.update(games)
    elif len(unique_rounds) > 1:
        game_index = out["game_index"].astype(np.int64)
        offset = 0
        for r in unique_rounds:
            rows = out["round"] == r
            game_index[rows] += offset
            offset = int(game_index[rows].max()) + 1
        out["game_index"] = game_index
    return out


def dataset_observation_space(
    dataset: dict[str, np.ndarray],
    observation_config: ObservationConfig | None,
) -> ObservationSpace:
    """Build the declared observation space; reject data of another shape."""
    obs_space = ObservationSpace(observation_config or ObservationConfig())
    for key, shape in obs_space.shapes.items():
        if dataset[key].shape[1:] != shape:
            raise ValueError(
                f"Dataset {key} shape differs from observation configuration"
            )
    return obs_space
