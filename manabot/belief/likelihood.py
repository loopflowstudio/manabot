"""Pinned policy likelihoods for managym-owned public commitments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.belief.range import BeliefState
from managym.decision import DecisionFrame
from managym.possible_worlds import PossibleWorldSpace


class RulesProviderGap(RuntimeError):
    """managym cannot identify this commitment at the admitted boundary."""


@dataclass(frozen=True, slots=True)
class LikelihoodResult:
    likelihoods: NDArray[np.float64]
    legal_action_counts: NDArray[np.int64]
    matching_action_counts: NDArray[np.int64]
    seconds: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_commitment_key(value: Mapping[str, Any]) -> str:
    kind = value.get("kind")
    if kind not in {
        "pass_priority",
        "cast",
        "play_land",
        "discard",
        "decline_discard",
    }:
        raise RulesProviderGap(f"unsupported public commitment kind {kind!r}")
    if kind in {"cast", "play_land", "discard"} and not value.get("card"):
        raise RulesProviderGap(f"{kind} commitment has no canonical card name")
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def _matching_offer_indexes(
    frame: DecisionFrame, observed: Mapping[str, Any]
) -> tuple[list[int], int]:
    """Group offers only by managym's provider-owned public identity."""

    observed_key = public_commitment_key(observed)
    groups: dict[str, list[int]] = {}
    for index, offer in enumerate(frame.offers):
        commitment = offer.get("public_commitment")
        key = (
            public_commitment_key(commitment)
            if isinstance(commitment, Mapping)
            else f"unsupported-offer:{offer['id']}"
        )
        groups.setdefault(key, []).append(index)
    return groups.get(observed_key, []), len(frame.offers)


class FrozenPolicyLikelihood:
    """Counterfactual likelihoods from one byte-locked world-w2 policy.

    Each row is materialized by canonical world index through the bound
    ``PossibleWorldSpace``. The evaluator receives no authority hand and never
    calls a direct exact-hand installation API.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        expected_sha256: str,
        batch_size: int = 256,
        device: str = "cpu",
        counterfactual_seed: int = 0,
    ) -> None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"frozen likelihood checkpoint is missing: {path}")
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "frozen likelihood checkpoint SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.checkpoint = path
        self.checkpoint_sha256 = actual_sha256
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.counterfactual_seed = counterfactual_seed
        from manabot.sim.flat_mc import load_checkpoint_agent

        self.agent, self.obs_space = load_checkpoint_agent(str(path))
        self.agent.to(self.device)
        self.agent.eval()

    def evaluate(
        self,
        root_engine: Any,
        *,
        viewer: int,
        commitment: Mapping[str, Any],
        belief: BeliefState,
    ) -> LikelihoodResult:
        started = time.perf_counter()
        root_space = PossibleWorldSpace.from_engine(root_engine, viewer)
        if root_space.identity != belief.space.identity:
            raise ValueError(
                "likelihood root does not match BeliefState space identity"
            )
        likelihoods = np.zeros(belief.support_size, dtype=np.float64)
        legal_counts = np.zeros(belief.support_size, dtype=np.int64)
        matching_counts = np.zeros(belief.support_size, dtype=np.int64)
        encoded_batch: list[dict[str, np.ndarray]] = []
        matching_batch: list[list[int]] = []
        legal_batch: list[int] = []
        row_batch: list[int] = []

        def flush() -> None:
            if not encoded_batch:
                return
            buffers = {
                key: np.stack([encoded[key] for encoded in encoded_batch])
                for key in encoded_batch[0]
            }
            tensors = {
                key: torch.from_numpy(value).to(self.device)
                for key, value in buffers.items()
            }
            with torch.inference_mode():
                logits, _ = self.agent.forward(tensors)
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            for local, row in enumerate(row_batch):
                matching = matching_batch[local]
                legal_counts[row] = legal_batch[local]
                matching_counts[row] = len(matching)
                if matching:
                    likelihoods[row] = float(probabilities[local, matching].sum())
            encoded_batch.clear()
            matching_batch.clear()
            legal_batch.clear()
            row_batch.clear()

        opponent = (viewer + 1) % 2
        for row in range(belief.space.support_size):
            hypothesis = root_space.materialize(
                row,
                seed=self.counterfactual_seed,
                refresh_opponent_commitment=True,
            )
            if hypothesis.current_agent_index() != opponent:
                raise RulesProviderGap(
                    "likelihood materialization did not publish the opponent decision"
                )
            raw = hypothesis.observation_for_player(opponent)
            frame = DecisionFrame.from_json(hypothesis.semantic_decision_frame_json())
            matching, legal_count = _matching_offer_indexes(frame, commitment)
            encoded_batch.append(self.obs_space.encode(raw))
            matching_batch.append(matching)
            legal_batch.append(legal_count)
            row_batch.append(row)
            if len(encoded_batch) >= self.batch_size:
                flush()
        flush()
        return LikelihoodResult(
            likelihoods=likelihoods,
            legal_action_counts=legal_counts,
            matching_action_counts=matching_counts,
            seconds=time.perf_counter() - started,
        )


__all__ = [
    "FrozenPolicyLikelihood",
    "LikelihoodResult",
    "RulesProviderGap",
    "_matching_offer_indexes",
    "file_sha256",
    "public_commitment_key",
]
