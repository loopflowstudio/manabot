"""Uniform physical-deal search over admitted, current-view policy sessions.

The live runner owns its session. This search evaluates isolated forks only;
native branches own sampling, legal Commands and game outcomes. Both rollout
seats use the base evaluator on their own live viewer observations. The shared
simplex optimizer changes the decision distribution, never model weights.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as validated_dataclass

import managym
from managym.bots import BeliefSummary, BotDecision
from managym.decision import DecisionFrame

if TYPE_CHECKING:
    from managym.policy import PolicyDescriptor, PolicyRunner, SearchPolicySession

FloatArray = NDArray[np.float64]


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
class SearchSummary:
    algorithm: str
    worlds: int
    depth: int
    alpha: float
    beta: float
    rollouts: int
    decision_seconds: float
    seconds: float
    seed: str  # Hexadecimal so browser JSON preserves all 64 bits.
    prior: list[float]
    receipt_sha256: str


@dataclass(frozen=True)
class SearchDecision(BotDecision):
    mirror: SearchSummary | None = None


NATIVE_PHYSICAL_BELIEF = "managym.determinize/physical-deal/v2"
MIRROR_METHOD = "regularized-local-update"


@validated_dataclass(
    frozen=True,
    config=ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True),
)
class SearchConfig:
    """One bot’s explicit Search configuration, shared by all launch paths."""

    algorithm: Literal["local_update"] = "local_update"
    belief: Literal["physical_deal"] = "physical_deal"
    worlds: int = Field(default=1, ge=1, le=64, strict=True)
    depth: int = Field(default=1, ge=1, le=40, strict=True)
    decision_seconds: float = Field(default=30, gt=0, le=30, strict=True)
    alpha: float = Field(default=0.02, ge=0, strict=True)
    beta: float = Field(default=1, gt=0, strict=True)


class SearchError(RuntimeError):
    """A named failure that commits no action and never selects a fallback."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def search_unavailable(runner: PolicyRunner) -> str | None:
    reason = descriptor_search_unavailable(runner.base_descriptor)
    if reason:
        return reason
    if not all(
        callable(getattr(runner.session, name, None)) for name in ("evaluate", "fork")
    ):
        return "Search requires full probabilities, signed values and isolated forks."
    return None


def descriptor_search_unavailable(descriptor: PolicyDescriptor) -> str | None:
    if not descriptor.supports_search:
        reason = descriptor.search_reason
        if isinstance(reason, str) and reason:
            return reason
        return "This policy does not support Search."
    if descriptor.recurrent:
        return "Search requires evaluation from the current view alone."
    return None


def search_descriptor(
    base: PolicyDescriptor, spec: SearchConfig, *, source: str | None = None
) -> PolicyDescriptor:
    """One composition identity for admission, tables, history and Replay."""
    reason = descriptor_search_unavailable(base)
    if reason:
        raise SearchError("mirror_unavailable", reason)
    return replace(
        base,
        name=f"{base.name} + search {spec.worlds}×{spec.depth}",
        runner=f"search/{source or search_source_identity()}",
        config={"base_policy": base.id, "mirror": asdict(spec)},
        base_policy=base,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def search_source_identity() -> str:
    """Bind both execution and numerical implementation to the composition."""
    directory = Path(__file__).parent
    files = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("search.py", "policy.py")
    }
    with Path(managym._managym.__file__).open("rb") as extension:
        files["native_extension"] = hashlib.file_digest(extension, "sha256").hexdigest()
    return _digest(files)


@dataclass(frozen=True)
class _RolloutEvidence:
    world_seed: int
    rollout_seed: int
    action: int
    commands: int
    value: float
    terminal: bool


def search_decision(
    runner: PolicyRunner,
    env: managym.Env,
    obs: managym.Observation,
    spec: SearchConfig,
    *,
    seed: int,
) -> BotDecision:
    """Return one searched decision without changing the live engine or session.

    Depth counts the root action. Each sampled world is shared by every action
    on the base support; each world/action branch has independent policy and
    action-sampling RNG. Deadline failures discard all estimates. The runner
    commits the returned choice through its ordinary bound Command authority.
    Deadline checks bracket provider/native calls; an in-process call must
    return before its overdue result can be rejected.
    """
    unavailable = search_unavailable(runner)
    if unavailable:
        raise SearchError("mirror_unavailable", unavailable)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Search seed must be an unsigned 64-bit integer.")
    viewer = runner.seat
    if env.current_agent_index() != viewer or int(obs.agent.player_index) != viewer:
        raise SearchError(
            "mirror_invalid_view", "Search requires the chooser's current view."
        )
    started = time.perf_counter()
    deadline = started + spec.decision_seconds

    def check() -> None:
        if time.perf_counter() >= deadline:
            raise SearchError(
                "mirror_deadline_exceeded",
                "Search reached its time limit. Retry the pending decision explicitly.",
            )

    # Native constraints enforce open decklists before any model work. The
    # retained digest contains public constraints, never the actual hidden deal.
    try:
        constraints = json.loads(env.hidden_hand_constraints_json(viewer))
    except managym.AgentError as exc:
        raise SearchError("mirror_world_unavailable", str(exc)) from exc
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    viewer_hash = env.semantic_viewer_state_hash(viewer)
    source = cast("SearchPolicySession", runner.session)
    known_sessions: list[SearchPolicySession] = [source]
    rng = np.random.default_rng(seed ^ 0x345ABC)
    evidence: list[_RolloutEvidence] = []

    with ExitStack() as resources:

        def fork(branch_seed: int, owner: ExitStack) -> SearchPolicySession:
            check()
            branch = source.fork(seed=branch_seed)
            if any(branch is existing for existing in known_sessions):
                raise SearchError(
                    "mirror_branch_isolation",
                    "Policy fork reused live or branch state.",
                )
            known_sessions.append(branch)

            def release() -> None:
                known_sessions.pop()
                branch.close()

            owner.callback(release)
            if not callable(getattr(branch, "evaluate", None)):
                raise SearchError("mirror_unavailable", "Policy fork has no evaluator.")
            check()
            return branch

        def evaluate(
            session: SearchPolicySession,
            engine: managym.Env,
            observation: managym.Observation,
        ) -> tuple[NDArray[np.float64], float]:
            check()
            actor = int(engine.current_agent_index())
            # Use the live reset/step projection: for_player omits recent events
            # and a canonical JSON round trip also loses selector focus.
            if int(observation.agent.player_index) != actor:
                raise SearchError(
                    "mirror_invalid_view", "Rollout observation changed its chooser."
                )
            view = runner.view(observation)
            # Search v1 sessions evaluate either seat from current view alone.
            view = replace(
                view,
                seat=actor,
                capture_json=engine.capture_decision(actor, 0).to_json(),
            )
            result = session.evaluate(view)
            check()
            try:
                probabilities = np.array(
                    result.probabilities, dtype=np.float64, copy=True
                )
                value = result.value
                valid = (
                    probabilities.shape == (len(view.choices),)
                    and bool(probabilities.size)
                    and bool(np.isfinite(probabilities).all())
                    and not bool(np.any(probabilities < 0))
                    and math.isclose(float(probabilities.sum()), 1, abs_tol=1e-5)
                    and value.seat == actor
                    and value.semantic == "signed_outcome"
                    and math.isfinite(value.value)
                    and -1 <= value.value <= 1
                )
            except (AttributeError, TypeError, ValueError):
                raise SearchError(
                    "mirror_invalid_evaluation",
                    "Policy returned no valid search evaluation.",
                ) from None
            if not valid:
                raise SearchError(
                    "mirror_invalid_evaluation",
                    "Search requires full legal probabilities and signed outcome values from the acting seat.",
                )
            return probabilities, float(value.value)

        base, root_value = evaluate(fork(seed ^ 0xABCDEF, resources), env, obs)
        supported = [int(action) for action in np.flatnonzero(base > 0)]
        if len(frame.offers) != len(base):
            raise SearchError(
                "mirror_invalid_evaluation", "Policy rows differ from native offers."
            )
        # Every supported action visits every world, or the whole search fails.
        # Unsupported actions keep missing values rather than synthetic zeros.
        sums = np.where(base > 0, 0.0, np.nan)
        for _ in range(spec.worlds):
            check()
            world_seed = int(rng.integers(0, 2**63))
            world = env.clone_env()
            # The native physical-deal sampler reserves viewer-known cards,
            # samples the residual pool with multiplicities and rebinds hidden
            # commitments. It also shuffles unknown library orders. No Python
            # hand ontology, enumeration or read of the true hand is needed.
            try:
                world.determinize(seed=world_seed, perspective=viewer)
            except managym.AgentError as exc:
                raise SearchError("mirror_world_unavailable", str(exc)) from exc
            if (
                world.semantic_viewer_state_hash(viewer) != viewer_hash
                or DecisionFrame.from_json(world.semantic_decision_frame_json()).offers
                != frame.offers
            ):
                raise SearchError(
                    "mirror_world_mismatch",
                    "Sampled world changed the viewer or legal offers.",
                )
            for action in supported:
                check()
                rollout_seed = int(rng.integers(0, 2**63))
                with ExitStack() as branch_resources:
                    session = fork(rollout_seed, branch_resources)
                    action_rng = np.random.default_rng(rollout_seed)
                    runtime = managym.SelectedBranchRuntime(
                        f"search-{seed}-{len(evidence)}", False
                    )
                    branch = runtime.fork_exact(world, "world")
                    choice = action
                    value = 0.0
                    actor = viewer
                    for ply in range(spec.depth):
                        check()
                        next_observation, _, _, truncated, _, _ = (
                            runtime.apply_policy_choice(branch, "child", choice)
                        )
                        if truncated:
                            raise SearchError(
                                "mirror_rollout_truncated",
                                "Rollout ended without an authoritative outcome.",
                            )
                        if branch.is_game_over():
                            break
                        actor = int(branch.current_agent_index())
                        probabilities, value = evaluate(
                            session, branch, next_observation
                        )
                        if ply + 1 < spec.depth:
                            # Normalize floating-point summation error only. Exact
                            # base zeros remain zero; no probability floor is added.
                            choice = int(
                                action_rng.choice(
                                    len(probabilities),
                                    p=probabilities / probabilities.sum(),
                                )
                            )
                    terminal = bool(branch.is_game_over())
                    if terminal:
                        winner = branch.winner_index()
                        signed = (
                            0.0
                            if winner is None
                            else (1.0 if winner == viewer else -1.0)
                        )
                    else:
                        signed = value if actor == viewer else -value
                    sums[action] += signed
                    evidence.append(
                        _RolloutEvidence(
                            world_seed, rollout_seed, action, ply + 1, signed, terminal
                        )
                    )
                    check()
        values = sums / spec.worlds
        reference = np.full(len(base), 1 / len(base))
        try:
            target = regularized_update(
                base, values, reference, alpha=spec.alpha, beta=spec.beta
            )
        except ValueError as exc:
            raise SearchError("mirror_numerical_failure", str(exc)) from exc
        check()

    # Cleanup is inside the budget and must succeed before a proposal exists.
    check()
    scores = [float(value) if base[i] > 0 else None for i, value in enumerate(values)]
    receipt = {
        "schema": "managym.mirror-search/v1",
        "policy": runner.base_descriptor.id,
        "source": runner.search_runtime or search_source_identity(),
        "spec": asdict(spec),
        "seed": seed,
        "viewer": viewer,
        "viewer_state_hash": viewer_hash,
        "belief": NATIVE_PHYSICAL_BELIEF,
        "constraints": constraints,
        "base": base.tolist(),
        "values": scores,
        "reference": reference.tolist(),
        "target": target.tolist(),
        "rollouts": [asdict(row) for row in evidence],
    }
    action = int(rng.choice(len(target), p=target))
    receipt_sha256 = _digest(receipt)
    check()
    return SearchDecision(
        action=action,
        probabilities=target.tolist(),
        value=root_value,
        value_semantic="signed_outcome",
        search_scores=scores,
        belief=BeliefSummary(
            NATIVE_PHYSICAL_BELIEF, _digest(constraints), 1.0, spec.worlds
        ),
        mirror=SearchSummary(
            algorithm=MIRROR_METHOD,
            worlds=spec.worlds,
            depth=spec.depth,
            alpha=spec.alpha,
            beta=spec.beta,
            rollouts=len(evidence),
            decision_seconds=spec.decision_seconds,
            seconds=time.perf_counter() - started,
            seed=f"{seed:016x}",
            prior=base.tolist(),
            receipt_sha256=receipt_sha256,
        ),
    )
