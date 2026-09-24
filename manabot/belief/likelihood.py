"""Pinned policy likelihoods for managym-owned public commitments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.belief.range import BeliefState
from managym.decision import DecisionFrame, PublicCommitment, SemanticContractError


class RulesProviderGap(RuntimeError):
    """managym cannot identify this commitment at the admitted boundary."""


@dataclass(frozen=True, slots=True)
class LikelihoodResult:
    likelihoods: NDArray[np.float64]
    legal_action_counts: NDArray[np.int64]
    matching_action_counts: NDArray[np.int64]
    seconds: float
    batches: int = 0
    space_constructions: int = 0
    max_batch_size: int = 0


@dataclass(frozen=True, slots=True)
class LikelihoodBatchProgress:
    batch_ordinal: int
    batch_size: int
    completed_rows: int
    total_rows: int
    evaluator_wall_seconds: float
    prepared_diagnostics: Mapping[str, int]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_commitment_key(value: Mapping[str, Any]) -> str:
    try:
        commitment = PublicCommitment.from_payload(value)
    except SemanticContractError as error:
        raise RulesProviderGap(str(error)) from error
    return json.dumps(
        commitment.to_payload(),
        sort_keys=True,
        separators=(",", ":"),
    )


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
        batch_observer: Callable[[LikelihoodBatchProgress], None] | None = None,
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
        self.batch_observer = batch_observer
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
        prepared = root_engine.prepare_possible_world_materializer(
            viewer,
            belief.space.identity,
            self.batch_size,
        )
        if prepared.space_identity != belief.space.identity:
            raise ValueError(
                "likelihood root does not match BeliefState space identity"
            )
        if prepared.viewer != viewer:
            raise ValueError("prepared likelihood materializer changed viewer")
        if prepared.support_size != belief.support_size:
            raise ValueError("prepared likelihood materializer changed support size")
        if prepared.construction_count != 1:
            raise RuntimeError("likelihood preparation must enumerate exactly once")
        likelihoods = np.zeros(belief.support_size, dtype=np.float64)
        legal_counts = np.zeros(belief.support_size, dtype=np.int64)
        matching_counts = np.zeros(belief.support_size, dtype=np.int64)

        def infer_batch(
            encoded_batch: list[dict[str, np.ndarray]],
            matching_batch: list[list[int]],
            legal_batch: list[int],
            row_batch: list[int],
        ) -> None:
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

        opponent = (viewer + 1) % 2
        completed_rows = 0
        batches = 0
        for start in range(0, belief.support_size, self.batch_size):
            stop = min(start + self.batch_size, belief.support_size)
            row_batch = list(range(start, stop))
            batch_size = len(row_batch)
            branches = prepared.materialize_indexes(
                row_batch,
                [self.counterfactual_seed] * batch_size,
                True,
            )
            encoded_batch: list[dict[str, np.ndarray]] = []
            matching_batch: list[list[int]] = []
            legal_batch: list[int] = []
            for hypothesis in branches:
                if hypothesis.current_agent_index() != opponent:
                    raise RulesProviderGap(
                        "likelihood materialization did not publish the opponent decision"
                    )
                raw = hypothesis.observation_for_player(opponent)
                frame = DecisionFrame.from_json(
                    hypothesis.semantic_decision_frame_json()
                )
                matching, legal_count = _matching_offer_indexes(frame, commitment)
                encoded_batch.append(self.obs_space.encode(raw))
                matching_batch.append(matching)
                legal_batch.append(legal_count)
            del hypothesis
            del branches
            infer_batch(encoded_batch, matching_batch, legal_batch, row_batch)
            del encoded_batch, matching_batch, legal_batch, row_batch
            completed_rows = stop
            batches += 1
            if self.batch_observer is not None:
                diagnostics = {
                    str(key): int(value)
                    for key, value in json.loads(prepared.diagnostics_json()).items()
                }
                self.batch_observer(
                    LikelihoodBatchProgress(
                        batch_ordinal=batches - 1,
                        batch_size=batch_size,
                        completed_rows=completed_rows,
                        total_rows=belief.support_size,
                        evaluator_wall_seconds=time.perf_counter() - started,
                        prepared_diagnostics=diagnostics,
                    )
                )
        diagnostics = json.loads(prepared.diagnostics_json())
        return LikelihoodResult(
            likelihoods=likelihoods,
            legal_action_counts=legal_counts,
            matching_action_counts=matching_counts,
            seconds=time.perf_counter() - started,
            batches=batches,
            space_constructions=int(diagnostics["space_constructions"]),
            max_batch_size=int(diagnostics["maximum_returned_batch"]),
        )


__all__ = [
    "FrozenPolicyLikelihood",
    "LikelihoodBatchProgress",
    "LikelihoodResult",
    "RulesProviderGap",
    "_matching_offer_indexes",
    "file_sha256",
    "public_commitment_key",
]
