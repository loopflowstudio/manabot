"""Local-search configuration and validated evidence without policy execution.

Serialized receipt fields and numerical checks retain their historical identity;
search execution and Academy replay consume this one representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from manabot.belief.sampling_types import SamplerInput, _digest
from manabot.sim.search.rollout_receipt import RolloutReceipt
from manabot.sim.search.sampling_receipt import (
    PHYSICAL_SAMPLER_IDENTITY,
    HandBatch,
    HandSample,
    validate_hand,
)
from managym.decision import Command, Observation

FloatArray = NDArray[np.float64]


class LocalSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    worlds: int = Field(default=1, ge=1)
    depth: int = Field(default=2, ge=1)
    full_probability: float = Field(default=1, ge=0, le=1)
    cheap_worlds: int = Field(default=1, ge=1)
    alpha: float = Field(default=0.02, ge=0)
    beta: float = Field(default=1.0, gt=0)
    decision_seconds: float = Field(default=1, gt=0)
    max_support: int = Field(default=20000, ge=1)
    sampling: Literal["belief", "compatible_prior", "learned"] = "compatible_prior"
    # Compact keeps only reconciled Command counts (v4 receipts); full keeps
    # per-Command native witnesses and actor viewer hashes (v3 receipts).
    rollout_evidence: Literal["full", "compact"] = "compact"

    @classmethod
    def from_recorded(cls, config: Mapping[str, object]) -> LocalSearchConfig:
        """Read a receipt or registration config; one omitting the field is full.

        Receipts and registrations written before compact evidence existed, and
        full-evidence receipts since, omit `rollout_evidence`.
        """
        return cls.model_validate({"rollout_evidence": "full", **config})

    def receipt_config_json(self) -> str:
        """Full-evidence receipts keep the config bytes written before the field."""
        return self.model_dump_json(
            exclude={"rollout_evidence"} if self.rollout_evidence == "full" else None
        )


class SamplerArtifact(BaseModel):
    """Exact learned-belief provenance; policy and world are checked at admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: Path
    sha256: str
    dataset_identity: str
    schema_identity: str
    policy_identity: str
    world_identity: str


@dataclass(frozen=True)
class LearnedBeliefReceipt:
    artifact: SamplerArtifact
    history_identity: str
    inputs: SamplerInput
    constraints_json: str
    sampling_seed: int
    hands: tuple[HandSample, ...]


def regularized_update(
    base: FloatArray,
    values: FloatArray,
    reference: FloatArray,
    *,
    alpha: float,
    beta: float,
) -> FloatArray:
    """Maximize p·Q - alpha KL(p||reference) - beta KL(p||base).

    All vectors have shape [legal_actions]. Base zeros restrict the feasible
    simplex; values outside its support may be NaN and are never consulted.
    The reference remains strictly positive. Values are signed expected outcomes;
    beta is the inverse step scale. This is one detached local simplex update,
    not a gradient through a determinized game or an argmax improvement claim.
    """
    if not np.isfinite([alpha, beta]).all() or alpha < 0 or beta <= 0:
        raise ValueError("local update requires finite alpha >= 0 and beta > 0")
    arrays = [np.asarray(item, dtype=np.float64) for item in (base, values, reference)]
    base, values, reference = arrays
    if base.ndim != 1 or not base.size or any(a.shape != base.shape for a in arrays):
        raise ValueError("local update vectors must align with legal offers")
    support = base > 0
    if not np.isfinite(values[support]).all() or any(
        not np.isfinite(a).all() for a in (base, reference)
    ):
        raise ValueError("local update vectors must be finite")
    for probabilities in (base, reference):
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
            raise ValueError(
                "base and reference require normalized nonnegative support"
            )
    if np.any(reference <= 0):
        raise ValueError("reference requires positive support")
    # beta > 0 restricts the feasible simplex to the base support. Missing Q
    # outside it is deliberately unused, never treated as a zero-valued rollout.
    logits = (
        values[support]
        + alpha * np.log(reference[support])
        + beta * np.log(base[support])
    ) / (alpha + beta)
    if not np.isfinite(logits).all():
        raise ValueError("local update logits are nonfinite")
    target = np.exp(logits - logits.max())
    target /= target.sum()
    if np.any(target == 0):
        raise ValueError("local update numerically lost mixed support")
    aligned = np.zeros_like(base)
    aligned[support] = target
    return aligned


@dataclass(frozen=True)
class CompoundPrefixReceipt:
    """Decoder prefix retained by receipts from removed compound policies.

    Current searches never write one. The type remains so historical receipts
    still validate and keep their replay identities.
    """

    root_observation_json: str
    projection_json: str
    tokens: tuple[int, ...]
    commands: tuple[Command, ...]


@dataclass(frozen=True)
class LocalUpdateReceipt:
    schema: str
    policy_sha256: str
    runtime_sha256: str
    teacher_source_sha256: str
    world_identity: str
    belief_digest: str
    belief_model: str
    sampling_probabilities: tuple[float, ...]
    query_json: str
    condition_mass: float
    viewer: int
    revision: int
    viewer_state_hash: str
    offer_ids: tuple[int, ...]
    base: tuple[float, ...]
    values: tuple[float | None, ...]
    reference: tuple[float, ...]
    target: tuple[float, ...]
    allocation_counts: tuple[int, ...]
    config_json: str
    seed: int
    seconds: float
    rollouts: tuple[RolloutReceipt, ...]
    belief_seconds: float = 0.0
    learned_belief: LearnedBeliefReceipt | None = None
    direct_belief: HandBatch | None = None
    compound_prefix: CompoundPrefixReceipt | None = None

    @classmethod
    def from_json(cls, encoded: str) -> LocalUpdateReceipt:
        receipt = _RECEIPT_ADAPTER.validate_json(encoded)
        if receipt.schema not in (
            "regularized-local-update/v1",
            "regularized-local-update/v2",
            "regularized-local-update/v3",
            "regularized-local-update/v4",
        ):
            raise ValueError("unsupported local update receipt")
        config = LocalSearchConfig.from_recorded(json.loads(receipt.config_json))
        compact = config.rollout_evidence == "compact"
        if compact != (receipt.schema == "regularized-local-update/v4"):
            raise ValueError("compact rollout evidence requires the v4 receipt schema")
        target = regularized_update(
            np.asarray(receipt.base),
            np.asarray(receipt.values, dtype=np.float64),
            np.asarray(receipt.reference),
            alpha=config.alpha,
            beta=config.beta,
        )
        support = np.asarray(receipt.base) > 0
        if receipt.schema in (
            "regularized-local-update/v1",
            "regularized-local-update/v2",
        ) and (not support.all() or any(value is None for value in receipt.values)):
            raise ValueError("historical receipts require complete positive support")
        if any(
            (value is None) == bool(active)
            for value, active in zip(receipt.values, support)
        ):
            raise ValueError("local values must be available exactly on policy support")
        count = len(target)
        if len(receipt.offer_ids) != count or len(set(receipt.offer_ids)) != count:
            raise ValueError("local receipt offers do not align")
        if (
            len(receipt.target) != count
            or not np.allclose(target, receipt.target)
            or not np.array_equal(np.asarray(receipt.target) > 0, support)
            or not np.isfinite(receipt.target).all()
            or np.any(np.asarray(receipt.target) < 0)
        ):
            raise ValueError("local target differs from retained update")
        if any(
            row.action_index < 0 or row.action_index >= count
            for row in receipt.rollouts
        ):
            raise ValueError("local rollout references an absent action")
        counts = np.bincount(
            [row.action_index for row in receipt.rollouts], minlength=count
        )
        if not np.array_equal(counts > 0, support) or not np.array_equal(
            counts, receipt.allocation_counts
        ):
            raise ValueError("local update lacks complete policy-support coverage")
        sums = np.bincount(
            [row.action_index for row in receipt.rollouts],
            weights=[row.signed_value for row in receipt.rollouts],
            minlength=count,
        )
        if not np.allclose(
            sums[support] / counts[support],
            np.asarray(receipt.values, dtype=np.float64)[support],
        ):
            raise ValueError("local values differ from retained rollouts")
        if receipt.schema != "regularized-local-update/v1" and any(
            (
                not compact
                and row.canonical_commands != len(row.actor_observation_hashes)
            )
            or not 1 <= row.canonical_commands <= config.depth
            for row in receipt.rollouts
        ):
            raise ValueError("local receipt canonical Command count differs")
        if compact:
            for row in receipt.rollouts:
                if row.actor_observation_hashes:
                    raise ValueError(
                        "compact rollout must omit actor observation hashes"
                    )
                _validate_rollout_audit(
                    row.branch_audit_json, row.canonical_commands, full=False
                )
        prefix = receipt.compound_prefix
        if any(row.decoder_factors < 0 for row in receipt.rollouts):
            raise ValueError("negative compound decoder-factor count")
        if prefix is None:
            if any(row.decoder_factors for row in receipt.rollouts):
                raise ValueError("compound factors lack retained prefix provenance")
        else:
            root = Observation.from_json(prefix.root_observation_json)
            revisions = [command.expected_revision for command in prefix.commands]
            if (
                receipt.schema == "regularized-local-update/v1"
                or root.viewer != receipt.viewer
                or root.revision > receipt.revision
                or any(a >= b for a, b in zip(revisions, revisions[1:]))
                or (
                    revisions
                    and (
                        revisions[0] != root.revision
                        or revisions[-1] >= receipt.revision
                    )
                )
                or (
                    not revisions
                    and (
                        root.revision != receipt.revision
                        or root.viewer_state_hash != receipt.viewer_state_hash
                    )
                )
                or (bool(prefix.tokens) != bool(prefix.commands))
            ):
                raise ValueError("compound prefix does not bind this decision")
        learned = receipt.learned_belief
        direct = receipt.direct_belief
        if (config.sampling == "learned") != (learned is not None):
            raise ValueError("local receipt sampling provenance differs")
        if direct is not None:
            if (
                receipt.schema == "regularized-local-update/v1"
                or config.sampling == "belief"
            ):
                raise ValueError("direct sampling requires a production sampling mode")
            prepared = direct.prepared
            if not prepared.history_identity or not prepared.distribution_identity:
                raise ValueError("direct sampling identities are missing")
            constraints_json, hands, sampling_seed = (
                prepared.constraints_json,
                direct.hands,
                direct.seed,
            )
            if (
                learned is not None
                or prepared.distribution_identity != PHYSICAL_SAMPLER_IDENTITY
            ):
                raise ValueError("physical sampler identity differs")
            source = json.loads(constraints_json)
            names = prepared.card_names
            if (
                len(set(names)) != len(names)
                or set(source["pool"]) - set(names)
                or prepared.inputs.pool_counts
                != tuple(source["pool"].get(n, 0) for n in names)
                or prepared.inputs.known_minima
                != tuple(source["known_hand"].get(n, 0) for n in names)
                or prepared.inputs.hand_size != source["hand_size"]
            ):
                raise ValueError(
                    "direct sampling coordinates differ from native constraints"
                )
        elif learned is not None:
            constraints_json, hands, sampling_seed = (
                learned.constraints_json,
                learned.hands,
                learned.sampling_seed,
            )
        else:
            if (
                receipt.schema != "regularized-local-update/v1"
                and config.sampling != "belief"
            ):
                raise ValueError("production receipt lacks direct sampling evidence")
            if any(
                row.world_index is None or row.sampled_hand is not None
                for row in receipt.rollouts
            ):
                raise ValueError("enumerated local receipt lacks world indexes")
            return receipt
        if (
            learned is not None
            and learned.artifact.policy_identity != receipt.policy_sha256
        ):
            raise ValueError("local receipt sampler and policy identities differ")
        source = json.loads(constraints_json)
        identity = source["source_observation"]
        if (
            identity["viewer"] != receipt.viewer
            or identity["revision"] != receipt.revision
            or identity["viewer_state_hash"] != receipt.viewer_state_hash
            or _digest(source) != receipt.world_identity
            or sampling_seed != receipt.seed
            or receipt.sampling_probabilities
            or receipt.condition_mass != 1.0
            or json.loads(receipt.query_json) != {"kind": "true"}
            or len(receipt.rollouts) != len(hands) * int(support.sum())
        ):
            raise ValueError("local receipt sampled root identity differs")
        width = int(support.sum())
        for ordinal, hand in enumerate(hands):
            validate_hand(hand, constraints_json)
            rows = receipt.rollouts[ordinal * width : (ordinal + 1) * width]
            if [row.action_index for row in rows] != list(np.flatnonzero(support)):
                raise ValueError("sampled hand lacks policy-support coverage")
            if any(
                row.world_index is not None or row.sampled_hand != hand for row in rows
            ):
                raise ValueError("local receipt rollout sampled hand differs")
        return receipt

    def to_json(self) -> str:
        payload = _RECEIPT_ADAPTER.dump_python(self, mode="json")
        if self.schema == "regularized-local-update/v1":
            payload.pop("direct_belief")
            payload.pop("compound_prefix")
            for rollout in payload["rollouts"]:
                rollout.pop("decoder_factors")
                rollout.pop("canonical_commands")
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def replay_identity(self) -> str:
        payload = json.loads(self.to_json())
        payload.pop("seconds")
        payload.pop("belief_seconds")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


_RECEIPT_ADAPTER = TypeAdapter(LocalUpdateReceipt)


class _RolloutCounters(BaseModel):
    applies: dict[str, int]


class _RolloutAudit(BaseModel):
    """The fields of a native session snapshot that rollout evidence binds."""

    audit: bool
    counters: _RolloutCounters
    tape_lengths: dict[str, int]
    tapes: dict[str, list[dict[str, object]]]


def _validate_rollout_audit(encoded: str, commands: int, *, full: bool) -> None:
    """A rollout forks once at the world site and applies only child Commands.

    Native counters count every applied Command in either mode; only an audited
    session also retains one tape witness per Command.
    """
    audit = _RolloutAudit.model_validate_json(encoded)
    applies = {"world": 0, "child": commands, "leaf": 0}
    lengths = applies if full else dict.fromkeys(applies, 0)
    if (
        audit.audit != full
        or audit.counters.applies != applies
        or audit.tape_lengths != lengths
        or {site: len(tape) for site, tape in audit.tapes.items()} != lengths
    ):
        raise ValueError(
            "rollout evidence differs from native Command counters or tapes"
        )
