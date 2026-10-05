"""Frozen-policy local search with private replay receipts and public evidence.

`LocalUpdateTeacher` owns no learning state. It estimates signed root-action
values using canonical worlds, actor-view policy rollouts and signed leaf values.
`LocalUpdatePlayer` maintains viewer history, or an explicitly selected exact
reference tracker. The closed-form
update follows Ataraxos supplement S3.7 (7)-(8); model mismatch and truncated
rollouts preclude an equilibrium guarantee. Receipts are private training data.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Literal

import managym._managym as native
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
import torch

from manabot.belief.likelihood import (
    FrozenPolicyLikelihood,
    RulesProviderGap,
    file_sha256,
)
from manabot.belief.player import ExactRangePlayer, RangeSampling
from manabot.belief.range import BeliefState
from manabot.belief.sampling import SamplerInput
from manabot.belief.sampling_data import _digest
from manabot.belief.sampling_fit import load_belief_sampler
from manabot.belief.state import (
    EmptyBeliefSupport,
    ViewerHistory,
    condition_belief,
    query_mass,
)
from manabot.env import Env
from manabot.model.world import validate_agent_setup
from manabot.sim.local_compound import (
    CompoundCursor,
    CompoundPrefixReceipt,
    CompoundProjection,
    CompoundRollout,
    advance_compound,
    project_compound,
    sample_suffix,
)
from manabot.sim.local_sampling import (
    HandBatch,
    HandSample,
    LearnedHandSampler,
    PhysicalHandSampler,
    SearchHandSampler,
    validate_hand,
)
from manabot.sim.search_branch import SelectedFullCloneBackend
import managym
from managym.decision import Command, DecisionFrame, Observation, SemanticTransition
from managym.possible_worlds import WorldQuery

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

    All vectors have shape [legal_actions]. Values are signed expected outcomes;
    beta is the inverse step scale. This is one detached local simplex update,
    not a gradient through a determinized game or an argmax improvement claim.
    """
    if not np.isfinite([alpha, beta]).all() or alpha < 0 or beta <= 0:
        raise ValueError("local update requires finite alpha >= 0 and beta > 0")
    arrays = [np.asarray(item, dtype=np.float64) for item in (base, values, reference)]
    base, values, reference = arrays
    if base.ndim != 1 or not base.size or any(a.shape != base.shape for a in arrays):
        raise ValueError("local update vectors must align with legal offers")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("local update vectors must be finite")
    for probabilities in (base, reference):
        if np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1):
            raise ValueError("base and reference require normalized positive support")
    logits = (values + alpha * np.log(reference) + beta * np.log(base)) / (alpha + beta)
    if not np.isfinite(logits).all():
        raise ValueError("local update logits are nonfinite")
    target = np.exp(logits - logits.max())
    target /= target.sum()
    if np.any(target == 0):
        raise ValueError("local update numerically lost mixed support")
    return target


@dataclass(frozen=True)
class RolloutReceipt:
    world_index: int | None
    world_seed: int
    action_index: int
    rollout_seed: int
    signed_value: float
    terminal: bool
    actor_observation_hashes: tuple[str, ...]
    branch_audit_json: str
    sampled_hand: HandSample | None = None
    decoder_factors: int = 0
    canonical_commands: int = 0


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
    values: tuple[float, ...]
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
        ):
            raise ValueError("unsupported local update receipt")
        config = LocalSearchConfig.model_validate_json(receipt.config_json)
        target = regularized_update(
            np.asarray(receipt.base),
            np.asarray(receipt.values),
            np.asarray(receipt.reference),
            alpha=config.alpha,
            beta=config.beta,
        )
        count = len(target)
        if len(receipt.offer_ids) != count or len(set(receipt.offer_ids)) != count:
            raise ValueError("local receipt offers do not align")
        if len(receipt.target) != count or not np.allclose(target, receipt.target):
            raise ValueError("local target differs from retained update")
        if any(
            row.action_index < 0 or row.action_index >= count
            for row in receipt.rollouts
        ):
            raise ValueError("local rollout references an absent action")
        counts = np.bincount(
            [row.action_index for row in receipt.rollouts], minlength=count
        )
        if np.any(counts == 0) or not np.array_equal(counts, receipt.allocation_counts):
            raise ValueError("local update lacks complete action coverage")
        sums = np.bincount(
            [row.action_index for row in receipt.rollouts],
            weights=[row.signed_value for row in receipt.rollouts],
            minlength=count,
        )
        if not np.allclose(sums / counts, receipt.values):
            raise ValueError("local values differ from retained rollouts")
        if receipt.schema == "regularized-local-update/v2" and any(
            row.canonical_commands != len(row.actor_observation_hashes)
            or not 1 <= row.canonical_commands <= config.depth
            for row in receipt.rollouts
        ):
            raise ValueError("local receipt canonical Command count differs")
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
                receipt.schema != "regularized-local-update/v2"
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
                receipt.schema != "regularized-local-update/v2"
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
                or prepared.distribution_identity != PhysicalHandSampler.identity
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
                receipt.schema == "regularized-local-update/v2"
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
            or len(receipt.rollouts) != len(hands) * count
        ):
            raise ValueError("local receipt sampled root identity differs")
        for ordinal, hand in enumerate(hands):
            validate_hand(hand, constraints_json)
            rows = receipt.rollouts[ordinal * count : (ordinal + 1) * count]
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

    def mixing_diagnostics(self) -> dict[str, float]:
        """Root-local nats and probability distances, not bluffing quality."""
        target, base = np.asarray(self.target), np.asarray(self.base)
        return {
            "target_entropy": float(-(target * np.log(target)).sum()),
            "base_entropy": float(-(base * np.log(base)).sum()),
            "kl_to_base": float((target * np.log(target / base)).sum()),
            "policy_l1": float(np.abs(target - base).sum()),
        }


_RECEIPT_ADAPTER = TypeAdapter(LocalUpdateReceipt)


def local_search_source_sha256() -> str:
    """Bind the search and sampling implementation owners for arena admission."""
    root = Path(__file__).resolve().parents[2]
    paths = (
        "manabot/sim/local_update.py",
        "manabot/sim/local_sampling.py",
        "manabot/sim/local_compound.py",
        "manabot/model/compound.py",
        "manabot/model/agent.py",
        "manabot/belief/sampling.py",
        "manabot/belief/sampling_data.py",
        "manabot/belief/sampling_fit.py",
    )
    return _digest({path: file_sha256(root / path) for path in paths})


class LocalUpdateTeacher:
    """One admitted policy supplies rollouts, V and the belief-generating identity."""

    def __init__(
        self,
        checkpoint: Path,
        sha256: str,
        config: LocalSearchConfig,
        *,
        sampler: SamplerArtifact | None = None,
        hand_sampler: SearchHandSampler | None = None,
    ) -> None:
        self.likelihood = FrozenPolicyLikelihood(checkpoint, expected_sha256=sha256)
        self.agent = self.likelihood.agent
        if self.agent.belief_count_buckets:
            raise ValueError("local search requires an observation-only policy")
        if self.agent.hypers.compound_decisions and config.sampling == "belief":
            raise ValueError(
                "compound exact-history likelihood is unsupported; use a direct sampler"
            )
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            not self.agent.hypers.compound_decisions
            and self.agent.hypers.value_kind != "categorical_wdl"
            and metadata.get("bc", {}).get("value_semantic") != "signed_outcome"
        ):
            raise ValueError(
                "scalar local-search checkpoint must declare signed_outcome values"
            )
        self.agent.requires_grad_(False)
        self.space = self.likelihood.obs_space
        self.config = config
        if file_sha256(checkpoint) != sha256:
            raise ValueError("checkpoint changed during local-search admission")
        if (config.sampling == "learned") != (sampler is not None):
            raise ValueError("learned search requires exactly one sampler artifact")
        self.sampler_artifact = sampler
        self._sampler_authority: SearchHandSampler | None = None
        if sampler is not None:
            if sampler.policy_identity != sha256 or sampler.world_identity != _digest(
                self.agent.world_binding
            ):
                raise ValueError(
                    "sampler generating policy or world differs from rollout policy"
                )
            model = load_belief_sampler(
                sampler.path,
                expected_checkpoint_identity=sampler.sha256,
                expected_dataset_identity=sampler.dataset_identity,
                expected_schema_identity=sampler.schema_identity,
                expected_policy_identity=sha256,
                expected_world_identity=sampler.world_identity,
            )
            model.requires_grad_(False)
            self._sampler_authority = LearnedHandSampler(model, sampler.sha256)
        elif config.sampling == "compatible_prior":
            self._sampler_authority = PhysicalHandSampler()
        self.hand_sampler = self._sampler_authority
        if hand_sampler is not None:
            if (
                self.hand_sampler is None
                or hand_sampler.kind != self.hand_sampler.kind
                or hand_sampler.identity != self.hand_sampler.identity
            ):
                raise ValueError("injected sampler differs from admitted distribution")
            self.hand_sampler = hand_sampler
        self.runtime_sha256 = file_sha256(Path(native.__file__))
        self.source_sha256 = local_search_source_sha256()

    def predict(self, engine: managym.Env) -> tuple[FloatArray, float]:
        if self.agent.hypers.compound_decisions:
            raise ValueError("compound inference requires a retained prefix projection")
        actor = int(engine.current_agent_index())
        raw = engine.observation_for_player(actor)
        encoded = self.space.encode(raw)
        count = len(raw.action_space.actions)
        if int(encoded["actions_valid"].sum()) != count or count == 0:
            raise ValueError("policy cannot represent complete legal offers")
        tensors = {
            key: torch.as_tensor(value).unsqueeze(0) for key, value in encoded.items()
        }
        with torch.inference_mode():
            logits, value = self.agent(tensors)
            probabilities = logits[0, :count].double().softmax(-1).numpy()
        signed_value = float(value[0])
        if not np.isfinite(signed_value):
            raise ValueError("nonfinite leaf value")
        # Scalar PPO heads are unconstrained; retain their actual estimate rather
        # than silently turning it into a sigmoid probability or clipping it.
        return probabilities, signed_value

    def verify_replay(
        self,
        engine: managym.Env,
        belief: BeliefState | ViewerHistory,
        receipt: LocalUpdateReceipt,
        *,
        query: WorldQuery | None = None,
        compound_cursor: CompoundCursor | None = None,
    ) -> None:
        """Re-execute the sampled branches and compare every retained witness.

        The caller supplies the exact historical root and its tracked belief;
        receipts are not a substitute for the canonical source trajectory.
        Elapsed time is the only excluded field.
        """
        if (
            receipt.compound_prefix is not None
            and receipt.compound_prefix.commands
            and compound_cursor is None
        ):
            raise ValueError(
                "compound replay requires the retained original-root cursor"
            )
        replay = self.search(
            engine,
            belief,
            seed=receipt.seed,
            query=query,
            compound_cursor=compound_cursor,
        )
        if replay.replay_identity() != receipt.replay_identity():
            raise ValueError("local update replay differs from retained evidence")

    def search(
        self,
        engine: managym.Env,
        belief: BeliefState | ViewerHistory,
        *,
        seed: int,
        query: WorldQuery | None = None,
        deadline: float | None = None,
        compound_cursor: CompoundCursor | None = None,
    ) -> LocalUpdateReceipt:
        started = time.perf_counter()
        limit = min(
            started + self.config.decision_seconds,
            deadline if deadline is not None else float("inf"),
        )

        def check() -> None:
            if time.perf_counter() >= limit:
                raise TimeoutError("local search deadline exceeded; target unavailable")

        check()
        viewer = int(engine.current_agent_index())
        observation = Observation.from_json(engine.semantic_observation_json(viewer))
        selected_query = query or WorldQuery.true()
        learned = None
        direct = None
        if self.hand_sampler is not None:
            if not isinstance(belief, ViewerHistory) or belief.viewer != viewer:
                raise ValueError("direct search requires acting-viewer history")
            if selected_query != WorldQuery.true():
                raise ValueError("direct conditional query mass is unsupported")
            assert self._sampler_authority is not None
            prepared = self._sampler_authority.prepare(engine, belief)
            if (
                self.hand_sampler is not self._sampler_authority
                and self.hand_sampler.prepare(engine, belief) != prepared
            ):
                raise ValueError(
                    "injected sampler preparation differs from admitted distribution"
                )
            # Compare against an independent native snapshot before trusting an
            # injected sampler's preparation or executing any branch.
            if (
                prepared.distribution_identity != self.hand_sampler.identity
                or prepared.history_identity != belief.identity
                or json.loads(prepared.constraints_json)
                != json.loads(engine.hidden_hand_constraints_json(viewer))
            ):
                raise ValueError(
                    "injected sampler preparation differs from source root"
                )
            world_identity = _digest(json.loads(prepared.constraints_json))
            belief_digest = _digest(asdict(prepared))
            belief_model = (
                f"approximate-learned-hand/sha256:{self.hand_sampler.identity}"
                if self.config.sampling == "learned"
                else self.hand_sampler.identity
            )
            sampling_probabilities = ()
            mass = 1.0
        else:
            if not isinstance(belief, BeliefState):
                raise ValueError("enumerated search requires a belief state")
            if (
                belief.space.viewer != viewer
                or belief.space.source_revision != observation.revision
                or belief.space.source_viewer_state_hash
                != observation.viewer_state_hash
            ):
                raise ValueError(
                    "local search belief is stale or belongs to another viewer"
                )
            if belief.support_size > self.config.max_support:
                raise ValueError("local search exact support exceeds declared cap")
            expected_model = (
                f"frozen-policy-likelihood/sha256:{self.likelihood.checkpoint_sha256}"
            )
            if self.config.sampling == "belief" and belief.model_id != expected_model:
                raise ValueError(
                    "belief likelihood and rollout policy identities differ"
                )
            mass = min(1.0, query_mass(belief, selected_query))
            conditioned = condition_belief(belief, selected_query)
            if isinstance(conditioned, EmptyBeliefSupport):
                raise ValueError("local search query has zero posterior mass")
            belief = conditioned
            world_identity, belief_digest, belief_model = (
                belief.space.identity,
                belief.digest,
                belief.model_id,
            )
            sampling_probabilities = tuple(float(p) for p in belief.probabilities)
        frame = DecisionFrame.from_json(engine.semantic_decision_frame_json())
        compound = (
            project_compound(self.agent, engine, compound_cursor, check)
            if self.agent.hypers.compound_decisions
            else None
        )
        if compound is not None:
            base = compound.probabilities
        else:
            base, _ = self.predict(engine)
        reference = np.full(len(base), 1.0 / len(base))
        if len(frame.offers) != len(base):
            raise ValueError("semantic offers and policy rows differ")
        check()
        # Independent streams prevent allocation from conditioning the first
        # sampled world; cheap/full variants still share their sample prefix.
        rng = np.random.default_rng(seed ^ 0x345ABC)
        allocation_rng = np.random.default_rng(seed ^ 0x98BCDA)
        world_count = (
            self.config.worlds
            if allocation_rng.random() < self.config.full_probability
            else min(self.config.cheap_worlds, self.config.worlds)
        )
        indexes: list[int | None]
        if self.hand_sampler is not None:
            direct = self.hand_sampler.sample(
                prepared, count=world_count, seed=seed, check=check
            )
            check()
            if (
                direct.prepared != prepared
                or direct.seed != seed
                or len(direct.hands) != world_count
            ):
                raise ValueError(
                    "injected sampler batch differs from requested preparation"
                )
            # Validate every hand before materializing even the first one.
            for hand in direct.hands:
                validate_hand(hand, prepared.constraints_json)
            if self.sampler_artifact is not None:
                learned = LearnedBeliefReceipt(
                    self.sampler_artifact,
                    prepared.history_identity,
                    prepared.inputs,
                    prepared.constraints_json,
                    seed,
                    direct.hands,
                )
            indexes = [None] * world_count
        else:
            assert isinstance(belief, BeliefState)
            indexes = list(belief.sample_indexes(world_count, seed=seed))
        sums = np.zeros(len(base), dtype=np.float64)
        counts = np.zeros(len(base), dtype=np.int64)
        receipts: list[RolloutReceipt] = []
        for sample_number, index in enumerate(indexes):
            check()
            world_seed = int(rng.integers(0, 2**63))
            sampled_hand = None
            if direct is not None:
                sampled_hand = direct.hands[sample_number]
                world = engine.materialize_sampled_hand(
                    viewer,
                    direct.prepared.constraints_json,
                    dict(sampled_hand.counts),
                    world_seed,
                )
            else:
                assert isinstance(belief, BeliefState) and index is not None
                world = belief.space.materialize(index, seed=world_seed)
            # All root actions see the same sampled deal; subsequent decisions
            # use only their own viewer observation, never determinized truth.
            if (
                Observation.from_json(
                    world.semantic_observation_json(viewer)
                ).viewer_state_hash
                != observation.viewer_state_hash
            ):
                raise ValueError("materialization changed acting viewer information")
            world_frame = DecisionFrame.from_json(world.semantic_decision_frame_json())
            if world_frame.offers != frame.offers:
                raise ValueError("materialization changed semantic root offers")
            for action in range(len(base)):
                check()
                rollout_seed = int(rng.integers(0, 2**63))
                policy_rng = np.random.default_rng(rollout_seed)
                session = SelectedFullCloneBackend().open_session(
                    match_id=f"local-{seed}-{len(receipts)}", audit=True
                )
                branch = session.fork_exact(world, "world")
                hashes: list[str] = []
                probabilities, value = base, 0.0
                compound_rollout = (
                    CompoundRollout(self.agent, compound, rollout_seed, check)
                    if compound is not None
                    else None
                )
                projection = compound
                for ply in range(self.config.depth):
                    check()
                    if branch.is_game_over():
                        break
                    actor = int(branch.current_agent_index())
                    hashes.append(
                        Observation.from_json(
                            branch.semantic_observation_json(actor)
                        ).viewer_state_hash
                    )
                    if compound_rollout is not None:
                        assert projection is not None
                        choice = (
                            action if ply == 0 else compound_rollout.choose(projection)
                        )
                        compound_rollout.advance(projection, choice)
                    else:
                        choice = (
                            action
                            if ply == 0
                            else int(
                                policy_rng.choice(len(probabilities), p=probabilities)
                            )
                        )
                    _, _, _, truncated, _ = session.apply_policy_choice(
                        branch, site="child", policy_index=choice
                    )
                    if truncated:
                        raise RuntimeError(
                            "local rollout truncated before an authoritative outcome"
                        )
                    if not branch.is_game_over():
                        if compound_rollout is not None:
                            projection = compound_rollout.project(branch)
                            probabilities, value = (
                                projection.probabilities,
                                projection.value,
                            )
                        else:
                            probabilities, value = self.predict(branch)
                terminal = bool(branch.is_game_over())
                if terminal:
                    winner = branch.winner_index()
                    value = (
                        0.0 if winner is None else (1.0 if winner == viewer else -1.0)
                    )
                elif int(branch.current_agent_index()) != viewer:
                    value = -value
                sums[action] += value
                counts[action] += 1
                receipts.append(
                    RolloutReceipt(
                        canonical_commands=len(hashes),
                        decoder_factors=compound_rollout.factors
                        if compound_rollout is not None
                        else 0,
                        world_index=index,
                        sampled_hand=sampled_hand,
                        world_seed=world_seed,
                        action_index=action,
                        rollout_seed=rollout_seed,
                        signed_value=value,
                        terminal=terminal,
                        actor_observation_hashes=tuple(hashes),
                        branch_audit_json=json.dumps(
                            session.snapshot(), sort_keys=True
                        ),
                    )
                )
        check()
        values = sums / counts
        target = regularized_update(
            base, values, reference, alpha=self.config.alpha, beta=self.config.beta
        )
        receipt = LocalUpdateReceipt(
            schema="regularized-local-update/v2",
            policy_sha256=self.likelihood.checkpoint_sha256,
            runtime_sha256=self.runtime_sha256,
            teacher_source_sha256=self.source_sha256,
            world_identity=world_identity,
            belief_digest=belief_digest,
            belief_model=belief_model,
            sampling_probabilities=sampling_probabilities,
            learned_belief=learned,
            direct_belief=direct if learned is None else None,
            compound_prefix=compound.cursor.receipt() if compound is not None else None,
            query_json=json.dumps(selected_query.to_dict(), sort_keys=True),
            condition_mass=mass,
            viewer=viewer,
            revision=observation.revision,
            viewer_state_hash=observation.viewer_state_hash,
            offer_ids=tuple(int(offer["id"]) for offer in frame.offers),
            base=tuple(base),
            values=tuple(values),
            reference=tuple(reference),
            target=tuple(target),
            allocation_counts=tuple(int(n) for n in counts),
            config_json=self.config.model_dump_json(),
            seed=seed,
            seconds=time.perf_counter() - started,
            rollouts=tuple(receipts),
        )
        check()
        return receipt


class LocalUpdatePlayer(ExactRangePlayer):
    """Arena lifecycle with zero smoothing and explicit policy mismatch evidence.

    Exact ranges and learned joint-hand approximations model the frozen policy.
    Neither becomes the actual opponent's posterior when opponent behavior differs.
    Production samplers never initialize an exact tracker or enumerate support.
    """

    def __init__(self, teacher: LocalUpdateTeacher, *, seed: int = 0) -> None:
        super().__init__(
            1,
            likelihood=teacher.likelihood,
            epsilon=0,
            sampling=RangeSampling.BELIEF
            if teacher.config.sampling == "learned"
            else RangeSampling(teacher.config.sampling),
            seed=seed,
        )
        self.teacher = teacher
        self.last_receipt: LocalUpdateReceipt | None = None
        self.deadline: float | None = None
        self._belief_seconds = 0.0
        self.history: ViewerHistory | None = None
        self.compound_cursor: CompoundCursor | None = None
        self.compound_projection: CompoundProjection | None = None
        self.base_pending: deque[Command] = deque()
        self._decision_limit: float = float("inf")

    def start_game(self, env: Env, seat: int) -> None:
        validate_agent_setup(self.teacher.agent, env.match.to_rust())
        started = time.perf_counter()
        self.last_receipt = None
        self.compound_cursor = None
        self.compound_projection = None
        self.base_pending.clear()
        self.history = None
        self.tracker = None
        if self.teacher.hand_sampler is not None:
            self.seat = seat
            self.history = ViewerHistory.from_observation(
                Observation.from_json(env._engine.semantic_observation_json(seat))
            )
        else:
            super().start_game(env, seat)
        self._belief_seconds = time.perf_counter() - started

    def finish_game(self, *, game_index: int, seed: int) -> None:
        if self.teacher.hand_sampler is None:
            super().finish_game(game_index=game_index, seed=seed)
        else:
            if self.history is None:
                raise RuntimeError("start_game is required")
            self.completed_replays.append(
                {
                    "game_index": game_index,
                    "seed": seed,
                    "history_identity": self.history.identity,
                    "belief_model": self.teacher.hand_sampler.identity,
                }
            )

    def prepare_step(self, env: Env, acting: int, action: int) -> None:
        if acting == self.seat and self.compound_projection is not None:
            self.compound_cursor = advance_compound(
                self.teacher.agent,
                self.compound_projection,
                action,
                self._check_deadline,
            )
            self.compound_projection = None
        if self.teacher.hand_sampler is None and self.sampling is RangeSampling.BELIEF:
            super().prepare_step(env, acting, action)

    def observe_step(
        self, env: Env, acting: int, transition: SemanticTransition
    ) -> None:
        if self.seat is None:
            raise RuntimeError("start_game is required")
        if self.teacher.hand_sampler is not None:
            assert self.history is not None
            started = time.perf_counter()
            self.history = self.history.advance(
                transition.receipt,
                Observation.from_json(env._engine.semantic_observation_json(self.seat)),
                acting=acting,
            )
            self._belief_seconds += time.perf_counter() - started
            return
        if acting != self.seat:
            if transition.receipt.public_commitment is None:
                raise RulesProviderGap(
                    "exact local posterior unavailable: opponent choice has no public likelihood identity"
                )
            root = self._pending_likelihood_root
            if root is None:
                raise RulesProviderGap(
                    "exact local posterior requires the pre-action root"
                )
            kind = int(
                root.observation_for_player(acting).action_space.action_space_type
            )
            if kind not in (
                int(managym.ActionSpaceEnum.PRIORITY),
                int(managym.ActionSpaceEnum.LEARN),
            ):
                raise RulesProviderGap(
                    "exact local posterior unavailable: materializer cannot refresh this opponent prompt"
                )
        super().observe_step(env, acting, transition)

    def _check_deadline(self) -> None:
        if time.perf_counter() >= self._decision_limit:
            raise TimeoutError("local search deadline exceeded; target unavailable")

    def base_action(self, rng: np.random.Generator) -> int:
        """Collection samples one whole base declaration and drains its Commands."""
        if self.last_receipt is None:
            raise RuntimeError("search must precede base-policy collection")
        projection = self.compound_projection
        if projection is None:
            return int(
                rng.choice(len(self.last_receipt.base), p=self.last_receipt.base)
            )
        if not self.base_pending:
            generator = torch.Generator().manual_seed(int(rng.integers(0, 2**63)))
            self.base_pending.extend(
                sample_suffix(
                    self.teacher.agent,
                    projection.cursor,
                    generator,
                    self._check_deadline,
                )
            )
        command = self.base_pending.popleft()
        return projection.action_index(command)

    def act(self, env: Env, obs: dict[str, NDArray[np.float32]]) -> int:
        del obs
        if self.tracker is None and self.history is None:
            raise RuntimeError("start_game is required")
        self.calls += 1
        # Charge history filtering since the previous decision to this envelope.
        update_seconds = (
            self.tracker.stats.update_seconds if self.tracker is not None else 0.0
        )
        spent = update_seconds + self._belief_seconds
        limit = time.perf_counter() + self.teacher.config.decision_seconds - spent
        if self.deadline is not None:
            limit = min(limit, self.deadline)
        belief: BeliefState | ViewerHistory
        if self.history is not None:
            belief = self.history
        else:
            assert self.tracker is not None
            belief = self.tracker.posterior
        self.last_receipt = None
        search_started = time.perf_counter()
        self.last_receipt = self.teacher.search(
            env._engine,
            belief,
            seed=(self.seed * 1_000_003 + self.calls) % (2**63),
            deadline=limit,
            compound_cursor=self.compound_cursor,
        )
        if self.teacher.agent.hypers.compound_decisions:
            self._decision_limit = limit
            self.compound_projection = project_compound(
                self.teacher.agent,
                env._engine,
                self.compound_cursor,
                self._check_deadline,
            )
        self.last_receipt = replace(
            self.last_receipt,
            belief_seconds=spent,
            seconds=time.perf_counter() - search_started,
        )
        self._belief_seconds = -update_seconds
        self.last_scores = np.asarray(self.last_receipt.values)
        self.stats.search.decisions += 1
        self.stats.search.seconds += self.last_receipt.seconds + spent
        self.stats.search.simulations += len(self.last_receipt.rollouts)
        self.stats.search.decision_seconds.append(self.last_receipt.seconds + spent)
        rng = np.random.default_rng(self.last_receipt.seed ^ 0x345678)
        return int(
            rng.choice(len(self.last_receipt.target), p=self.last_receipt.target)
        )
