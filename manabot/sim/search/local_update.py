"""Frozen-policy local search with private replay receipts and public evidence.

`LocalUpdateTeacher` owns no learning state. It estimates signed root-action
values using canonical worlds, actor-view policy rollouts and signed leaf values.
Live branches share one policy/value forward per ply. `Rollout` owns each
branch's execution state and independent RNG; immutable `RolloutReceipt` values
carry its evidence into the local-update receipt. Full rollout evidence (v3
receipts) keeps per-Command native witnesses and actor viewer hashes; opt-in
compact evidence (v4) keeps Command counts reconciled with native counters.
Opt-in native frontier encoding writes every live branch's view into
teacher-owned buffers with one `managym.encode_observations_into` call.
`LocalUpdatePlayer` maintains viewer history, or an explicitly selected exact
reference tracker. The closed-form
update follows Ataraxos supplement S3.7 (7)-(8); model mismatch and truncated
rollouts preclude an equilibrium guarantee. Receipts are private training data.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.belief.checkpoint import load_belief_sampler
from manabot.belief.likelihood import FrozenPolicyLikelihood, RulesProviderGap
from manabot.belief.player import ExactRangePlayer, RangeSampling
from manabot.belief.range import BeliefState
from manabot.belief.sampling_types import _digest
from manabot.belief.state import (
    EmptyBeliefSupport,
    ViewerHistory,
    condition_belief,
    query_mass,
)
from manabot.env.env import Env
from manabot.infra.config import COMPOUND_DECISIONS_REMOVED
from manabot.infra.identity import file_sha256
from manabot.model.admission import admit_observation
from manabot.model.world import validate_agent_setup
from manabot.sim.search.branch import SelectedFullCloneBackend
from manabot.sim.search.local_update_receipt import (
    FloatArray,
    LearnedBeliefReceipt,
    LocalSearchConfig,
    LocalUpdateReceipt,
    SamplerArtifact,
    _validate_rollout_audit,
    regularized_update,
)
from manabot.sim.search.rollout import Rollout
from manabot.sim.search.rollout_receipt import RolloutReceipt
from manabot.sim.search.sampling import (
    LearnedHandSampler,
    PhysicalHandSampler,
    SearchHandSampler,
)
from manabot.sim.search.sampling_receipt import validate_hand
import managym
import managym._managym as native
from managym.decision import DecisionFrame, Observation, SemanticTransition
from managym.possible_worlds import WorldQuery


def local_search_source_sha256() -> str:
    """Bind the search and sampling implementation owners for arena admission."""
    root = Path(__file__).resolve().parents[3]
    paths = (
        "manabot/sim/search/local_update.py",
        "manabot/sim/search/local_update_receipt.py",
        "manabot/sim/search/sampling_receipt.py",
        "manabot/belief/sampling_types.py",
        "manabot/sim/search/rollout.py",
        "manabot/sim/search/rollout_receipt.py",
        "manabot/sim/search/sampling.py",
        "manabot/backends/torch/policy.py",
        "manabot/model/facts.py",
        "manabot/belief/sampling.py",
        "manabot/belief/sampling_data.py",
        "manabot/belief/checkpoint.py",
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
        native_encoder: bool = True,
    ) -> None:
        self.likelihood = FrozenPolicyLikelihood(checkpoint, expected_sha256=sha256)
        self.agent = self.likelihood.agent
        if self.agent.belief_count_buckets:
            raise ValueError("local search requires an observation-only policy")
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            self.agent.config.value_kind != "categorical_wdl"
            and metadata.get("bc", {}).get("value_semantic") != "signed_outcome"
        ):
            raise ValueError(
                "scalar local-search checkpoint must declare signed_outcome values"
            )
        self.agent.requires_grad_(False)
        self.space = self.likelihood.obs_space
        # Native frontier encoding reuses these teacher-owned buffers across
        # forwards, so one teacher serves one search at a time.
        self.native_encoder = native_encoder
        self._frontier_buffers: dict[str, NDArray[np.generic]] = {}
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
        return self.predict_batch([engine])[0]

    def predict_batch(
        self, engines: Sequence[managym.Env]
    ) -> list[tuple[FloatArray, float]]:
        """Evaluate actor views together; each row keeps its complete legal support.

        Float32 matrix kernels can round differently from singleton inference.
        Probability normalization remains float64, over each row's legal offers.
        """
        if not engines:
            return []
        if self.native_encoder:
            encodings, counts = self._encode_native(engines)
        else:
            encodings, counts = self._encode_rows(engines)
        tensors = admit_observation(
            self.agent,
            {key: torch.as_tensor(value) for key, value in encodings.items()},
        )
        with torch.inference_mode():
            logits, values = self.agent(tensors)
            predictions = [
                (logits[i, :count].double().softmax(-1).numpy(), float(values[i]))
                for i, count in enumerate(counts)
            ]
        if any(not np.isfinite(value) for _, value in predictions):
            raise ValueError("nonfinite leaf value")
        # Scalar PPO heads are unconstrained; retain their actual signed estimate.
        return predictions

    def _encode_native(
        self, engines: Sequence[managym.Env]
    ) -> tuple[dict[str, NDArray[np.generic]], list[int]]:
        """One native call writes every actor view into the reused frontier buffers.

        The returned arrays are views of the first `len(engines)` rows and are
        valid until the next frontier encoding.
        """
        capacity = len(self._frontier_buffers.get("actions_valid", ()))
        if len(engines) > capacity:
            capacity = len(engines)
            self._frontier_buffers = self.space.encoder.allocate(capacity)
        counts = managym.encode_observations_into(
            engines, self._frontier_buffers, capacity
        )
        encodings = {
            key: value[: len(engines)] for key, value in self._frontier_buffers.items()
        }
        for index, count in enumerate(counts):
            if count == 0 or int(encodings["actions_valid"][index].sum()) != count:
                raise ValueError("policy cannot represent complete legal offers")
        return encodings, counts

    def _encode_rows(
        self, engines: Sequence[managym.Env]
    ) -> tuple[dict[str, NDArray[np.generic]], list[int]]:
        # The selected space owns every axis, including historical inputs and
        # nondefault capacities. Each row is a contiguous view of this call's
        # fresh batch; no observation buffers survive into the next frontier.
        encodings = self.space.encoder.allocate(len(engines))
        counts: list[int] = []
        for index, engine in enumerate(engines):
            actor = int(engine.current_agent_index())
            raw = engine.observation_for_player(actor)
            encoded = {key: value[index] for key, value in encodings.items()}
            count = len(raw.action_space.actions)
            if count == 0 or count > self.space.encoder.max_actions:
                raise ValueError("policy cannot represent complete legal offers")
            # Search requires every legal offer, even if a checkpoint permits
            # first-legal truncation during training.
            engine.encode_observation_into(raw, encoded)
            if int(encoded["actions_valid"].sum()) != count:
                raise ValueError("policy cannot represent complete legal offers")
            counts.append(count)
        return encodings, counts

    def _run_rollouts(self, rollouts: list[Rollout], check: Callable[[], None]) -> None:
        """Advance one Command per live branch, then one forward for the frontier.

        The last forward supplies leaf values too. Finished branches leave the
        frontier immediately; independent RNGs make scheduling immaterial to
        sampling, apart from floating-point differences in batched inference.
        """
        live = [row for row in rollouts if not row.branch.is_game_over()]
        for ply in range(self.config.depth):
            pending: list[Rollout] = []
            for row in live:
                check()
                row.advance(ply)
                if not row.branch.is_game_over():
                    pending.append(row)
            if not pending:
                break
            check()
            predictions = self.predict_batch([row.branch for row in pending])
            check()
            for row, (probabilities, value) in zip(pending, predictions, strict=True):
                row.probabilities, row.value = probabilities, value
            live = pending

    def verify_replay(
        self,
        engine: managym.Env,
        belief: BeliefState | ViewerHistory,
        receipt: LocalUpdateReceipt,
        *,
        query: WorldQuery | None = None,
    ) -> None:
        """Re-execute the sampled branches and compare the retained evidence.

        The caller supplies the exact historical root and its tracked belief;
        receipts are not a substitute for the canonical source trajectory.
        Elapsed time is the only excluded field. Compact receipts check values
        and Command counts, not the per-Command witnesses they omit. Receipts of removed compound
        policies are readable but not replayable here.
        """
        if receipt.compound_prefix is not None:
            raise ValueError(COMPOUND_DECISIONS_REMOVED)
        replay = self.search(engine, belief, seed=receipt.seed, query=query)
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
            constraints = json.loads(prepared.constraints_json)
            if (
                prepared.distribution_identity != self.hand_sampler.identity
                or prepared.history_identity != belief.identity
                or constraints
                != json.loads(engine.hidden_hand_constraints_json(viewer))
            ):
                raise ValueError(
                    "injected sampler preparation differs from source root"
                )
            world_identity = _digest(constraints)
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
        # The acting viewer's observation already carries this revision's frame.
        frame = observation.decision
        if frame is None:
            raise ValueError("local search requires an acting-viewer decision")
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
        supported_actions = [int(action) for action in np.flatnonzero(base > 0)]
        full_evidence = self.config.rollout_evidence == "full"
        rollouts: list[Rollout] = []
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
                world.semantic_viewer_state_hash(viewer)
                != observation.viewer_state_hash
            ):
                raise ValueError("materialization changed acting viewer information")
            world_frame = DecisionFrame.from_json(world.semantic_decision_frame_json())
            if world_frame.offers != frame.offers:
                raise ValueError("materialization changed semantic root offers")
            for action in supported_actions:
                check()
                rollout_seed = int(rng.integers(0, 2**63))
                session = SelectedFullCloneBackend().open_session(
                    match_id=f"local-{seed}-{len(rollouts)}", audit=full_evidence
                )
                rollouts.append(
                    Rollout(
                        branch=session.fork_exact(world, "world"),
                        session=session,
                        world_index=index,
                        world_seed=world_seed,
                        action_index=action,
                        rollout_seed=rollout_seed,
                        sampled_hand=sampled_hand,
                        probabilities=base,
                        record_hashes=full_evidence,
                    )
                )
        self._run_rollouts(rollouts, check)
        sums = np.zeros(len(base), dtype=np.float64)
        counts = np.zeros(len(base), dtype=np.int64)
        receipts: list[RolloutReceipt] = []
        for row in rollouts:
            check()
            receipt = row.receipt(viewer)
            _validate_rollout_audit(
                receipt.branch_audit_json,
                receipt.canonical_commands,
                full=full_evidence,
            )
            sums[row.action_index] += receipt.signed_value
            counts[row.action_index] += 1
            receipts.append(receipt)
        check()
        values = np.divide(
            sums, counts, out=np.full_like(sums, np.nan), where=counts > 0
        )
        target = regularized_update(
            base, values, reference, alpha=self.config.alpha, beta=self.config.beta
        )
        receipt = LocalUpdateReceipt(
            schema=(
                "regularized-local-update/v3"
                if full_evidence
                else "regularized-local-update/v4"
            ),
            policy_sha256=self.likelihood.checkpoint_sha256,
            runtime_sha256=self.runtime_sha256,
            teacher_source_sha256=self.source_sha256,
            world_identity=world_identity,
            belief_digest=belief_digest,
            belief_model=belief_model,
            sampling_probabilities=sampling_probabilities,
            learned_belief=learned,
            direct_belief=direct if learned is None else None,
            query_json=json.dumps(selected_query.to_dict(), sort_keys=True),
            condition_mass=mass,
            viewer=viewer,
            revision=observation.revision,
            viewer_state_hash=observation.viewer_state_hash,
            offer_ids=tuple(int(offer["id"]) for offer in frame.offers),
            base=tuple(base),
            values=tuple(float(v) if counts[i] else None for i, v in enumerate(values)),
            reference=tuple(reference),
            target=tuple(target),
            allocation_counts=tuple(int(n) for n in counts),
            config_json=self.config.receipt_config_json(),
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

    def start_game(self, env: Env, seat: int) -> None:
        validate_agent_setup(self.teacher.agent, env.match.to_rust())
        started = time.perf_counter()
        self.last_receipt = None
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
        )
        self.last_receipt = replace(
            self.last_receipt,
            belief_seconds=spent,
            seconds=time.perf_counter() - search_started,
        )
        self._belief_seconds = -update_seconds
        self.last_scores = np.asarray(self.last_receipt.values, dtype=np.float64)
        self.stats.search.decisions += 1
        self.stats.search.seconds += self.last_receipt.seconds + spent
        self.stats.search.simulations += len(self.last_receipt.rollouts)
        self.stats.search.decision_seconds.append(self.last_receipt.seconds + spent)
        rng = np.random.default_rng(self.last_receipt.seed ^ 0x345678)
        return int(
            rng.choice(len(self.last_receipt.target), p=self.last_receipt.target)
        )
