"""Determinized PUCT over conditional world priors.

INT-13 integration step 3 from the INT-10 search-learning architecture.
Extends the existing determinized PUCT teacher to accept an immutable
ConditionalWorldPrior and paired query plan, then returns one aligned
ConditionalStrategyResult for True, Has, Lacks, Q, and Not(Q) at the same
semantic root.

The planner is determinized PUCT — separate trees per sampled world, uniform
priors, random leaf playouts. This is NOT information-set Monte Carlo tree
search (ISMCTS), NOT public-belief search, and NOT equilibrium solving.
No belief inference or information-set-consistency claim is made.

RUL-8 (the Rules wave task owning the viewer-relative PossibleWorldSpace and
WorldQuery seam) is active but not yet publishable. This module defines the
narrow public Protocols and a local fixture adapter (ScenarioWorldSpace) that
implements them via the existing engine scenario API. When RUL-8 lands, its
real implementation replaces the fixture adapter; the Protocols stay
unchanged.

Sampled worlds and hidden truth are audit-private. The ConditionalStrategyResult
exposes action distributions, Q-values, uncertainty, and comparison deltas —
never opponent hands, world labels, or hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from manabot.sim.flat_mc import DEFAULT_MAX_PLAYOUT_STEPS
from manabot.sim.mcts import (
    LeafEvaluator,
    UniformRandomLeafEvaluator,
    _Node,
    _mix_seed,
    _select_action,
    _simulate,
)
from manabot.sim.search_branch import (
    REFERENCE_BRANCH_DRIVER_ID,
    SearchBranchBackend,
    branch_backend as resolve_branch_backend,
)
from manabot.sim.teacher1_evidence import (
    canonical_sha256,
    engine_source_paths,
    source_bundle_sha256,
)

PLANNER_NAME = "determinized_puct"
CONDITION_TRUE = "true"


class ConditionalSearchError(ValueError):
    """Typed failure for the conditional search vertical slice."""


# ---------------------------------------------------------------------------
# RUL-8 narrow public shape (Protocols)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldSpec:
    """One exact compatible world, identified by its hidden-state definition.

    ``opponent_hand`` is audit-private: it never appears in the viewer-safe
    layer of a ConditionalStrategyResult. The WorldQuery evaluates against
    this definition, not against live engine state — the engine deliberately
    does not expose opponent hand cards to Python.
    """

    world_index: int
    label: str
    weight: float
    opponent_hand: tuple[str, ...]


class WorldQuery(Protocol):
    """Predicate over a possible world's hidden-state definition."""

    query_id: str

    def matches(self, world: WorldSpec) -> bool: ...


class PossibleWorldSpace(Protocol):
    """Viewer-relative set of exact compatible worlds (RUL-8 seam).

    RUL-8 owns the world-space enumeration and configuration. This Protocol
    is the narrow public shape that conditional search consumes. The local
    fixture adapter (ScenarioWorldSpace) implements it via the engine
    scenario API; when RUL-8 lands, its real implementation replaces the
    adapter.
    """

    space_id: str
    viewer: int
    worlds: tuple[WorldSpec, ...]

    def configure(self, env: Any, *, world_index: int) -> None:
        """Configure a forked env to the hidden state of ``world_index``."""
        ...


# ---------------------------------------------------------------------------
# Concrete WorldQuery implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrueQuery:
    """The unconditional baseline: all worlds satisfy it."""

    @property
    def query_id(self) -> str:
        return CONDITION_TRUE

    def matches(self, world: WorldSpec) -> bool:
        return True


@dataclass(frozen=True)
class HasCard:
    """Condition: the opponent's hand contains ``card_name``."""

    card_name: str

    @property
    def query_id(self) -> str:
        return f"has:{self.card_name}"

    def matches(self, world: WorldSpec) -> bool:
        return self.card_name in world.opponent_hand


@dataclass(frozen=True)
class NotQuery:
    """Logical complement of an inner query."""

    inner: WorldQuery

    @property
    def query_id(self) -> str:
        return f"not({self.inner.query_id})"

    def matches(self, world: WorldSpec) -> bool:
        return not self.inner.matches(world)


# ---------------------------------------------------------------------------
# Conditional prior and query plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionalWorldPrior:
    """Immutable prior over exact compatible worlds.

    ``weights`` is a normalized probability vector over
    ``world_space.worlds``. For the fixture, weights are uniform. When the
    belief tracker is built, weights reflect public-history likelihood
    updates.
    """

    world_space: PossibleWorldSpace
    weights: np.ndarray
    viewer: int
    prior_sha256: str

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 1:
            raise ConditionalSearchError("prior weights must be 1-D")
        if len(weights) != len(self.world_space.worlds):
            raise ConditionalSearchError(
                "prior weights length does not match world space support"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ConditionalSearchError(
                "prior weights must be finite and non-negative"
            )
        total = float(weights.sum())
        if total <= 0.0:
            raise ConditionalSearchError("prior weights must have positive mass")
        object.__setattr__(self, "weights", weights / total)


@dataclass(frozen=True)
class ConditionalQueryPlan:
    """Five aligned conditions at one semantic root.

    - True: the unconditional baseline (all worlds).
    - Has: ``has`` query (e.g., opponent has Counterspell).
    - Lacks: Not(has) — opponent lacks the card.
    - Q: the general query.
    - Not(Q): Not(q) — complement of the general query.
    """

    has: WorldQuery
    q: WorldQuery
    plan_sha256: str

    @property
    def conditions(self) -> tuple[tuple[str, WorldQuery], ...]:
        return (
            (CONDITION_TRUE, TrueQuery()),
            (self.has.query_id, self.has),
            (NotQuery(self.has).query_id, NotQuery(self.has)),
            (self.q.query_id, self.q),
            (NotQuery(self.q).query_id, NotQuery(self.q)),
        )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionResult:
    """Search result for one condition at the shared semantic root."""

    condition_id: str
    condition_mass: float
    support: int
    sampled_worlds: int
    visit_counts: np.ndarray
    q_values: np.ndarray
    root_value: float
    world_q_values: np.ndarray
    world_root_values: np.ndarray
    uncertainty: float
    simulations: int
    cap_hits: int
    tree_nodes: int
    max_depth: int
    branch_driver_id: str
    branch_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class ConditionalStrategyResult:
    """Aligned conditional strategy for five conditions at one root.

    All conditions share the same action identities (``action_count`` and
    ``action_labels``), planner, evaluator, budget, and paired seeds.
    ``comparison_deltas`` quantifies how each condition's strategy differs
    from the True baseline.
    """

    conditions: tuple[ConditionResult, ...]
    action_count: int
    action_labels: tuple[str, ...]
    root_state_digest: str
    planner: str
    search_params: Mapping[str, Any]
    prior_sha256: str
    plan_sha256: str
    identities: Mapping[str, Any]
    realized_compute: Mapping[str, Any]
    comparison_deltas: Mapping[str, Mapping[str, float]]

    @property
    def condition_by_id(self) -> dict[str, ConditionResult]:
        return {c.condition_id: c for c in self.conditions}


# ---------------------------------------------------------------------------
# Local fixture adapter: ScenarioWorldSpace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioWorldSpace:
    """Local fixture adapter implementing PossibleWorldSpace.

    Uses the engine scenario API (scenario_clear_hand +
    scenario_force_card_in_hand + scenario_refresh) to configure a forked
    env to a specific opponent hand. This is the stand-in for RUL-8 until
    its real world-space implementation lands.

    The forked env must be an unguarded managym.Env (from clone_env, not
    from SelectedBranchRuntime.fork_exact). The scenario methods do not
    check selected_guard, but using them on a guarded env is outside the
    intended API.
    """

    space_id: str
    viewer: int
    worlds: tuple[WorldSpec, ...]
    opponent_seat: int

    def configure(self, env: Any, *, world_index: int) -> None:
        if world_index < 0 or world_index >= len(self.worlds):
            raise ConditionalSearchError(
                f"world_index {world_index} out of range for {len(self.worlds)} worlds"
            )
        world = self.worlds[world_index]
        env.scenario_clear_hand(self.opponent_seat)
        for card_name in world.opponent_hand:
            env.scenario_force_card_in_hand(self.opponent_seat, card_name)
        env.scenario_refresh()


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def _world_specs_sha256(worlds: Sequence[WorldSpec]) -> str:
    payload = [
        {
            "world_index": w.world_index,
            "label": w.label,
            "weight": float(w.weight),
            "opponent_hand": list(w.opponent_hand),
        }
        for w in worlds
    ]
    return canonical_sha256(payload)


def make_prior(
    world_space: PossibleWorldSpace,
    *,
    viewer: int,
    weights: np.ndarray | None = None,
    space_id: str | None = None,
) -> ConditionalWorldPrior:
    """Construct a ConditionalWorldPrior with deterministic identity."""

    n = len(world_space.worlds)
    w = (
        np.full(n, 1.0 / n, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    prior_sha256 = _world_specs_sha256(world_space.worlds)
    sid = space_id or getattr(world_space, "space_id", "world-space")
    return ConditionalWorldPrior(
        world_space=world_space,
        weights=w,
        viewer=viewer,
        prior_sha256=canonical_sha256(
            {"space_id": sid, "world_specs_sha256": prior_sha256, "viewer": viewer}
        ),
    )


def make_query_plan(
    has: WorldQuery,
    q: WorldQuery,
) -> ConditionalQueryPlan:
    """Construct a ConditionalQueryPlan with deterministic identity."""

    plan_sha256 = canonical_sha256({"has": has.query_id, "q": q.query_id})
    return ConditionalQueryPlan(has=has, q=q, plan_sha256=plan_sha256)


def _runtime_identities(root_env: Any) -> dict[str, Any]:
    """Collect world tag, engine source hash, content digest, and ABI hashes."""

    try:
        content = root_env.content_pack_manifest()
    except Exception:
        content = {}
    engine_paths = engine_source_paths()
    return {
        "world": "w2",
        "engine_source_sha256": source_bundle_sha256(engine_paths),
        "content_digest": content.get("content_digest"),
        "content_manifest_sha256": canonical_sha256(content) if content else None,
    }


def _action_labels(root_env: Any, action_count: int) -> tuple[str, ...]:
    """Extract human-readable action labels from the root env's offers."""

    try:
        context = json.loads(root_env.search_context_json(False))
        offers = context.get("offers", {}).get("offers", [])
        labels = []
        for i in range(action_count):
            if i < len(offers):
                labels.append(
                    str(
                        offers[i].get("label")
                        or offers[i].get("description")
                        or f"action-{i}"
                    )
                )
            else:
                labels.append(f"action-{i}")
        return tuple(labels)
    except Exception:
        return tuple(f"action-{i}" for i in range(action_count))


# ---------------------------------------------------------------------------
# Core search: conditional_determinized_puct
# ---------------------------------------------------------------------------


def _filter_worlds(
    prior: ConditionalWorldPrior,
    query: WorldQuery,
) -> tuple[list[int], np.ndarray, float, int]:
    """Filter worlds by a query. Returns (indices, normalized_weights, mass, support)."""

    worlds = prior.world_space.worlds
    indices = [i for i, w in enumerate(worlds) if query.matches(w)]
    if not indices:
        raise ConditionalSearchError(
            f"condition '{query.query_id}' has empty support — no world satisfies it"
        )
    raw_weights = np.asarray([prior.weights[i] for i in indices], dtype=np.float64)
    mass = float(raw_weights.sum())
    normalized = raw_weights / mass
    return indices, normalized, mass, len(indices)


def _select_world_indices(
    world_indices: list[int],
    weights: np.ndarray,
    max_worlds: int,
    seed: int,
) -> list[int]:
    """Deterministically select up to ``max_worlds`` worlds, weighted.

    Uses a seed-derived stable ordering: sort by (weight descending, index
    ascending) and take the top ``max_worlds``. For uniform weights this is
    just the first ``max_worlds`` indices in order. This is deterministic
    and paired across conditions (same seed → same selection order).
    """

    n = len(world_indices)
    if n <= max_worlds:
        return list(world_indices)
    order = sorted(
        range(n),
        key=lambda k: (-float(weights[k]), world_indices[k], seed),
    )
    return [world_indices[k] for k in order[:max_worlds]]


def _search_one_condition(
    root_env: Any,
    *,
    prior: ConditionalWorldPrior,
    query: WorldQuery,
    simulations: int,
    max_worlds: int,
    seed: int,
    c_puct: float,
    max_steps: int,
    evaluator: LeafEvaluator,
    root_priors: np.ndarray,
    action_count: int,
    hero: int,
    branch_session: Any,
    condition_id: str,
) -> ConditionResult:
    """Run determinized PUCT over the conditioned subset of worlds."""

    all_indices, normalized_weights, mass, support = _filter_worlds(prior, query)
    selected = _select_world_indices(all_indices, normalized_weights, max_worlds, seed)
    actual_worlds = len(selected)
    per_world = [simulations // actual_worlds] * actual_worlds
    for i in range(simulations % actual_worlds):
        per_world[i] += 1

    visits = np.zeros(action_count, dtype=np.int64)
    value_sums = np.zeros(action_count, dtype=np.float64)
    total_value = 0.0
    cap_hits = 0
    tree_nodes = 0
    max_depth = 0
    world_q_rows: list[np.ndarray] = []
    world_root_values: list[float] = []

    for slot, world_idx in enumerate(selected):
        world_seed = _mix_seed(seed, slot)
        world = branch_session.fork_exact(root_env, "world")
        prior.world_space.configure(world, world_index=world_idx)
        if int(world.action_count()) != action_count:
            raise ConditionalSearchError(
                f"condition '{condition_id}' world {world_idx} changed root "
                f"action count from {action_count} to {int(world.action_count())}"
            )
        root = _Node.from_env(world, priors=root_priors.copy())
        tree_nodes += 1
        world_value = 0.0
        for sim_idx in range(per_world[slot]):
            rollout_seed = _mix_seed(world_seed, sim_idx + 1)
            value, hit_cap, depth, added = _simulate(
                root,
                hero=hero,
                c_puct=c_puct,
                rollout_seed=rollout_seed,
                max_steps=max_steps,
                evaluator=evaluator,
                branch_session=branch_session,
            )
            total_value += value
            world_value += value
            cap_hits += int(hit_cap)
            tree_nodes += added
            max_depth = max(max_depth, depth)
        visits += root.visits
        value_sums += root.value_sums
        world_q_rows.append(
            np.divide(
                root.value_sums,
                root.visits,
                out=np.full(action_count, 0.5, dtype=np.float64),
                where=root.visits > 0,
            ).astype(np.float32)
        )
        world_root_values.append(world_value / per_world[slot])

    q_values = np.divide(
        value_sums,
        visits,
        out=np.full(action_count, 0.5, dtype=np.float64),
        where=visits > 0,
    )
    world_q_stack = (
        np.stack(world_q_rows)
        if world_q_rows
        else np.zeros((0, action_count), dtype=np.float32)
    )
    world_root_arr = np.asarray(world_root_values, dtype=np.float32)

    if world_q_stack.shape[0] > 1:
        visited_mask = visits > 0
        per_action_se = np.zeros(action_count, dtype=np.float64)
        for a in range(action_count):
            if visited_mask[a]:
                covered = world_q_stack[:, a]
                if len(covered) > 1:
                    per_action_se[a] = float(
                        np.std(covered, ddof=1) / math.sqrt(len(covered))
                    )
        uncertainty = (
            float(np.mean(per_action_se[visited_mask])) if np.any(visited_mask) else 0.0
        )
    else:
        uncertainty = 0.0

    return ConditionResult(
        condition_id=condition_id,
        condition_mass=mass,
        support=support,
        sampled_worlds=actual_worlds,
        visit_counts=visits,
        q_values=q_values.astype(np.float32),
        root_value=total_value / simulations,
        world_q_values=world_q_stack.astype(np.float32),
        world_root_values=world_root_arr,
        uncertainty=uncertainty,
        simulations=simulations,
        cap_hits=cap_hits,
        tree_nodes=tree_nodes,
        max_depth=max_depth,
        branch_driver_id=branch_session.driver_id,
        branch_receipt=branch_session.snapshot(),
    )


def conditional_determinized_puct(
    root_env: Any,
    *,
    prior: ConditionalWorldPrior,
    query_plan: ConditionalQueryPlan,
    simulations: int,
    worlds: int,
    seed: int,
    c_puct: float = 1.5,
    max_steps: int = DEFAULT_MAX_PLAYOUT_STEPS,
    evaluator: LeafEvaluator | None = None,
    root_observation: Any | None = None,
    branch_driver_id: str = REFERENCE_BRANCH_DRIVER_ID,
    branch_backend: SearchBranchBackend | None = None,
    branch_audit: bool = True,
    branch_match_id: str | None = None,
) -> ConditionalStrategyResult:
    """Search one root under five conditions and return aligned results.

    Parameters
    ----------
    root_env : managym.Env
        The authoritative root at a non-terminal Priority decision.
    prior : ConditionalWorldPrior
        Immutable prior over exact compatible worlds.
    query_plan : ConditionalQueryPlan
        Five conditions: True, Has, Lacks, Q, Not(Q).
    simulations : int
        Total PUCT traversal budget per condition (held fixed across conditions).
    worlds : int
        Maximum number of worlds to search per condition.
    seed : int
        Base seed. Paired across conditions via ``_mix_seed``.

    Returns
    -------
    ConditionalStrategyResult
        Aligned results for all five conditions with comparison deltas.

    Raises
    ------
    ConditionalSearchError
        On empty support, action misalignment, or identity mismatch.
    """

    if simulations < 1:
        raise ConditionalSearchError("simulations must be >= 1")
    if worlds < 1:
        raise ConditionalSearchError("worlds must be >= 1")
    if c_puct <= 0:
        raise ConditionalSearchError("c_puct must be positive")
    if max_steps < 1:
        raise ConditionalSearchError("max_steps must be >= 1")
    if root_env.is_game_over():
        raise ConditionalSearchError("cannot search a terminal environment")

    hero = root_env.current_agent_index()
    if hero is None:
        raise ConditionalSearchError("PUCT root has no acting player")
    hero = int(hero)
    action_count = int(root_env.action_count())
    if action_count < 1:
        raise ConditionalSearchError("root has no legal actions")

    backend = branch_backend or resolve_branch_backend(branch_driver_id)
    if backend.driver_id != branch_driver_id:
        raise ConditionalSearchError(
            f"branch backend {backend.driver_id!r} does not match requested "
            f"driver {branch_driver_id!r}"
        )
    branch_session = backend.open_session(
        match_id=branch_match_id or f"cond-puct-{seed}", audit=branch_audit
    )
    ev = evaluator or UniformRandomLeafEvaluator()
    root_priors = ev.root_priors(root_observation, action_count=action_count)
    root_digest = root_env.state_digest()
    action_labels = _action_labels(root_env, action_count)

    conditions: list[ConditionResult] = []
    for condition_id, query in query_plan.conditions:
        cr = _search_one_condition(
            root_env,
            prior=prior,
            query=query,
            simulations=simulations,
            max_worlds=worlds,
            seed=seed,
            c_puct=c_puct,
            max_steps=max_steps,
            evaluator=ev,
            root_priors=root_priors,
            action_count=action_count,
            hero=hero,
            branch_session=branch_session,
            condition_id=condition_id,
        )
        conditions.append(cr)

    if root_env.state_digest() != root_digest:
        raise ConditionalSearchError(
            "conditional search mutated the authoritative root"
        )

    true_result = conditions[0]
    comparison_deltas: dict[str, dict[str, float]] = {}
    for cr in conditions[1:]:
        deltas: dict[str, float] = {
            "root_value_delta": float(cr.root_value - true_result.root_value),
            "uncertainty_delta": float(cr.uncertainty - true_result.uncertainty),
        }
        if action_count > 0:
            true_visits = true_result.visit_counts.astype(np.float64)
            cr_visits = cr.visit_counts.astype(np.float64)
            true_total = float(true_visits.sum()) or 1.0
            cr_total = float(cr_visits.sum()) or 1.0
            true_dist = true_visits / true_total
            cr_dist = cr_visits / cr_total
            deltas["visit_dist_l1"] = float(np.sum(np.abs(cr_dist - true_dist)))
            q_delta = cr.q_values.astype(np.float64) - true_result.q_values.astype(
                np.float64
            )
            deltas["q_max_abs_delta"] = float(np.max(np.abs(q_delta)))
            top_true = int(np.argmax(true_result.visit_counts))
            top_cr = int(np.argmax(cr.visit_counts))
            deltas["top_action_changed"] = float(top_true != top_cr)
        comparison_deltas[cr.condition_id] = deltas

    total_sims = sum(c.simulations for c in conditions)
    total_tree_nodes = sum(c.tree_nodes for c in conditions)
    identities = _runtime_identities(root_env)

    realized_compute = {
        "total_simulations": total_sims,
        "total_tree_nodes": total_tree_nodes,
        "conditions_searched": len(conditions),
    }

    search_params = {
        "simulations": simulations,
        "worlds": worlds,
        "c_puct": c_puct,
        "max_steps": max_steps,
        "seed": seed,
        "branch_driver_id": branch_driver_id,
    }

    return ConditionalStrategyResult(
        conditions=tuple(conditions),
        action_count=action_count,
        action_labels=action_labels,
        root_state_digest=root_digest,
        planner=PLANNER_NAME,
        search_params=search_params,
        prior_sha256=prior.prior_sha256,
        plan_sha256=query_plan.plan_sha256,
        identities=identities,
        realized_compute=realized_compute,
        comparison_deltas=comparison_deltas,
    )


# ---------------------------------------------------------------------------
# Serialization (authority-private + viewer-safe layers)
# ---------------------------------------------------------------------------


def _condition_result_to_dict(cr: ConditionResult) -> dict[str, Any]:
    receipt = dict(cr.branch_receipt) if cr.branch_receipt else {}
    receipt.pop("tapes", None)
    receipt.pop("tape_lengths", None)
    return {
        "condition_id": cr.condition_id,
        "condition_mass": float(cr.condition_mass),
        "support": int(cr.support),
        "sampled_worlds": int(cr.sampled_worlds),
        "visit_counts": cr.visit_counts.astype(int).tolist(),
        "q_values": [float(x) for x in cr.q_values],
        "root_value": float(cr.root_value),
        "world_q_values": [[float(x) for x in row] for row in cr.world_q_values],
        "world_root_values": [float(x) for x in cr.world_root_values],
        "uncertainty": float(cr.uncertainty),
        "simulations": int(cr.simulations),
        "cap_hits": int(cr.cap_hits),
        "tree_nodes": int(cr.tree_nodes),
        "max_depth": int(cr.max_depth),
        "branch_driver_id": cr.branch_driver_id,
        "branch_receipt": receipt,
    }


def serialize_result(result: ConditionalStrategyResult) -> dict[str, Any]:
    """Serialize to a two-layer JSON payload.

    ``authority_private`` — full per-condition results, identities, and
    branch receipts. Consumed by INT-12 (the lower-level contract).

    ``viewer_safe`` — action distributions, Q-values, uncertainty, and
    comparison deltas. Consumed by GAM-6 (the Etude Study advisor).
    No opponent hands, world labels, or hidden truth appear in this layer.
    """

    conditions_priv = [_condition_result_to_dict(c) for c in result.conditions]

    viewer_conditions = []
    for c in result.conditions:
        total_visits = int(c.visit_counts.sum()) or 1
        dist = (c.visit_counts.astype(np.float64) / total_visits).tolist()
        viewer_conditions.append(
            {
                "condition_id": c.condition_id,
                "condition_mass": float(c.condition_mass),
                "support": int(c.support),
                "action_distribution": [float(x) for x in dist],
                "q_values": [float(x) for x in c.q_values],
                "root_value": float(c.root_value),
                "uncertainty": float(c.uncertainty),
                "simulations": int(c.simulations),
            }
        )

    return {
        "schema_version": 1,
        "planner": result.planner,
        "action_count": int(result.action_count),
        "action_labels": list(result.action_labels),
        "root_state_digest": result.root_state_digest,
        "search_params": dict(result.search_params),
        "prior_sha256": result.prior_sha256,
        "plan_sha256": result.plan_sha256,
        "identities": dict(result.identities),
        "realized_compute": dict(result.realized_compute),
        "comparison_deltas": {k: dict(v) for k, v in result.comparison_deltas.items()},
        "authority_private": {
            "conditions": conditions_priv,
        },
        "viewer_safe": {
            "conditions": viewer_conditions,
        },
    }


def canonical_result_json(result: ConditionalStrategyResult) -> str:
    """Canonical JSON serialization for deterministic fixture comparison."""

    return json.dumps(serialize_result(result), sort_keys=True, separators=(",", ":"))


def result_sha256(result: ConditionalStrategyResult) -> str:
    """SHA-256 of the canonical serialization."""

    return hashlib.sha256(canonical_result_json(result).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Validation (fail-closed)
# ---------------------------------------------------------------------------


def validate_result(result: ConditionalStrategyResult) -> None:
    """Fail closed on identity mismatch, action misalignment, or missing data."""

    if result.planner != PLANNER_NAME:
        raise ConditionalSearchError(
            f"planner is {result.planner!r}, expected {PLANNER_NAME!r}"
        )
    if len(result.conditions) != 5:
        raise ConditionalSearchError(
            f"expected 5 conditions, got {len(result.conditions)}"
        )
    for cr in result.conditions:
        if int(cr.visit_counts.sum()) != cr.simulations:
            raise ConditionalSearchError(
                f"condition {cr.condition_id}: visit sum {int(cr.visit_counts.sum())} "
                f"!= simulations {cr.simulations}"
            )
        if len(cr.visit_counts) != result.action_count:
            raise ConditionalSearchError(
                f"condition {cr.condition_id}: visit count length "
                f"{len(cr.visit_counts)} != action_count {result.action_count}"
            )
        if len(cr.q_values) != result.action_count:
            raise ConditionalSearchError(
                f"condition {cr.condition_id}: q_values length "
                f"{len(cr.q_values)} != action_count {result.action_count}"
            )
        if not (0.0 <= cr.root_value <= 1.0):
            raise ConditionalSearchError(
                f"condition {cr.condition_id}: root_value {cr.root_value} out of [0, 1]"
            )
    true_cr = result.conditions[0]
    if true_cr.condition_id != CONDITION_TRUE:
        raise ConditionalSearchError(
            f"first condition is {true_cr.condition_id!r}, expected {CONDITION_TRUE!r}"
        )
    for cr in result.conditions[1:]:
        if cr.condition_id not in result.comparison_deltas:
            raise ConditionalSearchError(
                f"condition {cr.condition_id} missing from comparison_deltas"
            )
