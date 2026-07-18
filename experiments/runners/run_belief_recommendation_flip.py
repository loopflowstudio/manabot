"""Freeze the INT-15 belief-conditioned recommendation flip.

The retained result is deliberately narrow: three already-probed positions,
three already-probed search seeds, one selected fixture, and the existing
uniform-random determinized-PUCT evaluator. Runtime measurements are recorded
outside the deterministic evidence envelope.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Mapping

from fastapi.testclient import TestClient

from etude.advice import (
    ADVISOR_SOURCE_PATHS,
    CHECKPOINT_VERSIONED_FIXTURE_PATH,
    FLIP_VERSIONED_FIXTURE_PATH,
    VERSIONED_FIXTURE_PATH,
    AdviceProvider,
    AdviceRequest,
    BeliefDistributionPayload,
    LiveAdvisorDecisionResolver,
    RegisteredAdvisor,
    StaticBeliefDistributionResolver,
    StudyAdvisorDecisionResolver,
    VersionedAdviceFixture,
    advice_request_sha256,
    belief_distribution_sha256,
    load_flip_versioned_advice_fixture,
    parse_advice_response_bytes,
    request_versioned_fixture_advice,
    runtime_advisor_abis,
    serialize_advice_response,
    source_bundle_sha256,
)
from etude.advice_identity import (
    AdvisorComputeIdentity,
    AdvisorIdentity,
    AdvisorSeedIdentity,
    CodeSourceArtifact,
)
from etude.replay_index import (
    ReplayDecisionAddress,
    canonical_projection_sha256,
    project_replay,
)
from etude.server import GameSession, app
from etude.testing_house_protocol import (
    BeliefScenario,
    BeliefSource,
    PersonalAudience,
    PlayerAuthoredBeliefProvenance,
    ViewerIdentity,
)
from manabot.belief.range import BeliefState
from manabot.sim.conditional_search import (
    ConditionalStrategyResult,
    canonical_result_json,
    conditional_determinized_puct_beliefs,
    result_sha256,
    validate_result,
)
from manabot.sim.search_branch import SELECTED_BRANCH_DRIVER_ID
from managym.possible_worlds import SupportReceipt, WorldQuery

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = (
    REPO_ROOT / "experiments" / "data" / "int-15-belief-recommendation-flip-v1.json"
)

INT12_FIXTURE_SHA256 = (
    "4a3fbeaa8461e00a785e961b9819508d2c1065ae98f058cc50a3783db0945e8d"
)
CHECKPOINT_FIXTURE_SHA256 = (
    "c26cc1dd2a4104baf033e1792d210013295ebcf063fff184d5483e1b49c93d7a"
)

SEARCH_SEEDS = (197, 198, 199)
SELECTED_SEED = 197
POSITION_RESET_SEED = 197
SIMULATIONS = 512
SAMPLED_WORLDS = 16
C_PUCT = 1.5
MAX_STEPS = 80
HAS_SCENARIO_ID = "typed-has-counterspell"
LACKS_SCENARIO_ID = "typed-lacks-counterspell"
EXPECTED_CONDITION_IDS = (HAS_SCENARIO_ID, LACKS_SCENARIO_ID)
COMPUTE_ID = "puct-512-traversal-16-world-counterspell-flip-v1"
SELECTED_CANDIDATE_ID = "countered-wipe-four-wide-v1"
MIN_VISIT_MARGIN = 32
MAX_CAP_RATE = 0.35


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    hero_life: int
    villain_life: int
    villain_battlefield: tuple[str, ...]


CANDIDATES = (
    CandidateSpec(
        candidate_id=SELECTED_CANDIDATE_ID,
        hero_life=8,
        villain_life=12,
        villain_battlefield=(
            "Gray Ogre",
            "Gray Ogre",
            "Wind Drake",
            "Raging Goblin",
        ),
    ),
    CandidateSpec(
        candidate_id="countered-wipe-three-wide-v1",
        hero_life=8,
        villain_life=12,
        villain_battlefield=("Gray Ogre", "Gray Ogre", "Wind Drake"),
    ),
    CandidateSpec(
        candidate_id="countered-wipe-buffered-v1",
        hero_life=12,
        villain_life=12,
        villain_battlefield=("Gray Ogre", "Gray Ogre", "Wind Drake"),
    ),
)


@dataclass
class PositionRuntime:
    temporary_directory: TemporaryDirectory[str]
    spec: CandidateSpec
    session: GameSession
    address: str
    viewer: ViewerIdentity
    source_replay_sha256: str
    live_resolver: LiveAdvisorDecisionResolver
    study_resolver: StudyAdvisorDecisionResolver
    root_state_digest: str

    def close(self) -> None:
        self.temporary_directory.cleanup()


@dataclass(frozen=True)
class ConditionedRange:
    scenario_id: str
    model_id: str
    query: WorldQuery
    receipt: SupportReceipt
    weights: tuple[float, ...]
    belief: BeliefState


@dataclass
class AdviceRuntime:
    position: PositionRuntime
    provider: AdviceProvider
    study_provider: AdviceProvider
    request: AdviceRequest
    conditions: tuple[ConditionedRange, ...]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss * 1024 if sys.platform.startswith("linux") else rss


def _measurement_profile() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }


def _assert_frozen_evidence_unchanged() -> None:
    if _file_sha256(VERSIONED_FIXTURE_PATH) != INT12_FIXTURE_SHA256:
        raise RuntimeError("INT-12 advice fixture changed")
    if _file_sha256(CHECKPOINT_VERSIONED_FIXTURE_PATH) != CHECKPOINT_FIXTURE_SHA256:
        raise RuntimeError("checkpoint advice fixture changed")


def build_position(spec: CandidateSpec) -> PositionRuntime:
    """Build and commit one exact Game-owned Pyroclasm-or-pass decision."""

    temporary_directory = TemporaryDirectory(prefix=f"int-15-{spec.candidate_id}-")
    session = GameSession(
        Path(temporary_directory.name),
        id_factory=lambda kind: f"int15-{spec.candidate_id}-{kind}",
        clock=lambda: "2026-07-18T00:00:00+00:00",
        villain_offer_policy=lambda context: int(context.offers[-1]["id"]),
    )
    session.new_game(
        {
            "villain_type": "passive",
            "seed": POSITION_RESET_SEED,
            "hero_deck": "interactive",
            "villain_deck": "interactive",
            "auto_pass": False,
        }
    )
    if session.env is None:
        raise RuntimeError("position session did not create an environment")
    env = session.env
    env.scenario_clear_hand(0)
    env.scenario_force_card_in_hand(0, "Pyroclasm")
    for _ in range(2):
        env.scenario_force_battlefield(0, "Mountain", True)
        env.scenario_force_battlefield(1, "Island", True)
    for card_name in spec.villain_battlefield:
        env.scenario_force_battlefield(1, card_name, True)
    env.scenario_set_life(0, spec.hero_life)
    env.scenario_set_life(1, spec.villain_life)
    session.obs = env.scenario_refresh()
    session.published_prompt = None

    frame = session._experience_frame()
    offer_labels = [str(offer["label"]) for offer in frame["offers"]]
    if offer_labels != ["Cast Pyroclasm", "Pass priority"]:
        raise RuntimeError(f"curated root action surface drifted: {offer_labels!r}")
    pass_offer = next(
        offer for offer in frame["offers"] if offer["label"] == "Pass priority"
    )
    ordinal = len(session.canonical_decisions)
    outcome = session.hero_command(
        {
            "command_id": f"int15-{spec.candidate_id}-commit",
            "match_id": frame["match_id"],
            "expected_revision": frame["revision"],
            "prompt_id": frame["prompt"]["id"],
            "offer_id": pass_offer["id"],
            "answers": [],
        }
    )
    if outcome["status"] != "accepted":
        raise RuntimeError("curated root command was not accepted")
    row = session.canonical_decisions[ordinal]
    replay = session.canonical_replay()
    address = ReplayDecisionAddress.from_decision(replay, row).serialize()
    source_replay_sha256 = canonical_projection_sha256(project_replay(replay, 0))
    viewer = ViewerIdentity(
        viewer_id=f"viewer.int15.{spec.candidate_id}",
        table_id=session.match_id,
        rules_viewer=0,
    )
    live_resolver = LiveAdvisorDecisionResolver(session, source_replay_sha256)
    study_resolver = StudyAdvisorDecisionResolver(
        session.study_fork_provider(), source_replay_sha256
    )
    resolved = live_resolver.resolve(ReplayDecisionAddress.parse(address), viewer)
    return PositionRuntime(
        temporary_directory=temporary_directory,
        spec=spec,
        session=session,
        address=address,
        viewer=viewer,
        source_replay_sha256=source_replay_sha256,
        live_resolver=live_resolver,
        study_resolver=study_resolver,
        root_state_digest=resolved.root.state_digest(),
    )


def build_conditioned_ranges(position: PositionRuntime) -> tuple[ConditionedRange, ...]:
    """Condition the compatible prior through authoritative typed receipts."""

    resolved = position.live_resolver.resolve(
        ReplayDecisionAddress.parse(position.address), position.viewer
    )
    space = resolved.world_space
    specs = (
        (
            HAS_SCENARIO_ID,
            "typed-has-counterspell-compatible-prior/v1",
            WorldQuery.has("Counterspell"),
            lambda count: count >= 1,
        ),
        (
            LACKS_SCENARIO_ID,
            "typed-lacks-counterspell-compatible-prior/v1",
            WorldQuery.lacks("Counterspell"),
            lambda count: count < 1,
        ),
    )
    conditioned: list[ConditionedRange] = []
    for scenario_id, model_id, query, includes in specs:
        receipt = space.support(query)
        weights = tuple(
            float(world.weight) if includes(world.count("Counterspell")) else 0.0
            for world in space.worlds
        )
        selected_weights = [
            world.weight
            for world in space.worlds
            if includes(world.count("Counterspell"))
        ]
        if len(selected_weights) != receipt.support_size:
            raise RuntimeError("typed query support size differs from canonical rows")
        if sum(selected_weights) != receipt.total_weight:
            raise RuntimeError("typed query weight differs from canonical receipt")
        belief = BeliefState.from_probabilities(space, model_id, weights)
        if belief.positive_support_size != receipt.support_size:
            raise RuntimeError("conditioned belief support differs from receipt")
        conditioned.append(
            ConditionedRange(
                scenario_id=scenario_id,
                model_id=model_id,
                query=query,
                receipt=receipt,
                weights=weights,
                belief=belief,
            )
        )
    return tuple(conditioned)


def run_search(
    position: PositionRuntime,
    conditions: tuple[ConditionedRange, ...],
    *,
    seed: int,
    branch_audit: bool,
) -> ConditionalStrategyResult:
    resolved = position.live_resolver.resolve(
        ReplayDecisionAddress.parse(position.address), position.viewer
    )
    result = conditional_determinized_puct_beliefs(
        resolved.root,
        beliefs={row.scenario_id: row.belief for row in conditions},
        simulations=SIMULATIONS,
        worlds=SAMPLED_WORLDS,
        seed=seed,
        c_puct=C_PUCT,
        max_steps=MAX_STEPS,
        branch_driver_id=SELECTED_BRANCH_DRIVER_ID,
        branch_audit=branch_audit,
        branch_match_id=f"int15-{position.spec.candidate_id}-{seed}",
    )
    validate_result(result, expected_condition_ids=EXPECTED_CONDITION_IDS)
    if resolved.root.state_digest() != position.root_state_digest:
        raise RuntimeError("conditional search changed the retained root")
    return result


def _winner(values: list[int] | list[float]) -> tuple[int, float]:
    ordered = sorted(enumerate(values), key=lambda row: row[1], reverse=True)
    if len(ordered) < 2 or ordered[0][1] == ordered[1][1]:
        raise RuntimeError("fixture action distribution has no unique winner")
    return int(ordered[0][0]), float(ordered[0][1] - ordered[1][1])


def _top_summary(values: list[int] | list[float]) -> tuple[int, float, bool]:
    ordered = sorted(enumerate(values), key=lambda row: row[1], reverse=True)
    if len(ordered) < 2:
        raise RuntimeError("fixture action distribution has fewer than two actions")
    return (
        int(ordered[0][0]),
        float(ordered[0][1] - ordered[1][1]),
        bool(ordered[0][1] != ordered[1][1]),
    )


def summarize_result(result: ConditionalStrategyResult) -> dict[str, Any]:
    conditions: list[dict[str, Any]] = []
    for condition in result.conditions:
        visits = [int(value) for value in condition.visit_counts]
        top_index, visit_margin, unique_top = _top_summary(visits)
        q_values = [float(value) for value in condition.q_values]
        q_top_index, q_margin, unique_q_top = _top_summary(q_values)
        conditions.append(
            {
                "condition_id": condition.condition_id,
                "visit_counts": visits,
                "q_values": q_values,
                "top_action_index": top_index,
                "unique_top_action": unique_top,
                "visit_margin": int(visit_margin),
                "q_top_action_index": q_top_index,
                "unique_q_top_action": unique_q_top,
                "q_margin": q_margin,
                "root_value": float(condition.root_value),
                "uncertainty": float(condition.uncertainty),
                "support": int(condition.support),
                "sampled_worlds": int(condition.sampled_worlds),
                "simulations": int(condition.simulations),
                "cap_hits": int(condition.cap_hits),
                "tree_nodes": int(condition.tree_nodes),
                "max_depth": int(condition.max_depth),
            }
        )
    return {
        "result_sha256": result_sha256(result),
        "root_state_digest": result.root_state_digest,
        "action_labels": list(result.action_labels),
        "conditions": conditions,
        "top_action_changed": float(
            conditions[0]["top_action_index"] != conditions[1]["top_action_index"]
        ),
        "comparison_deltas": {
            key: dict(value) for key, value in result.comparison_deltas.items()
        },
        "realized_compute": dict(result.realized_compute),
    }


def condition_receipts(
    position: PositionRuntime,
    conditions: tuple[ConditionedRange, ...],
) -> list[dict[str, Any]]:
    resolved = position.live_resolver.resolve(
        ReplayDecisionAddress.parse(position.address), position.viewer
    )
    total_weight = resolved.world_space.total_weight
    return [
        {
            "scenario_id": row.scenario_id,
            "query": row.query.to_dict(),
            "support_receipt": asdict(row.receipt),
            "condition_mass": row.receipt.total_weight / total_weight,
            "condition_mass_ratio": {
                "numerator": row.receipt.total_weight,
                "denominator": total_weight,
            },
            "belief_model_id": row.model_id,
            "normalized_belief_sha256": row.belief.digest,
            "normalization_error": row.belief.normalization_error,
        }
        for row in conditions
    ]


def _scenario(
    position: PositionRuntime,
    identity: AdvisorIdentity,
    scenario_id: str,
    revision: int,
) -> BeliefScenario:
    return BeliefScenario(
        id=scenario_id,
        author_viewer_id=position.viewer.viewer_id,
        source=BeliefSource(
            decision_address=position.address,
            gam6_scenario_id=scenario_id,
            advice_identity=identity,
        ),
        audience=PersonalAudience(kind="personal"),
        provenance=PlayerAuthoredBeliefProvenance(created_at_table_revision=revision),
    )


def build_advice_runtime(
    position: PositionRuntime,
    conditions: tuple[ConditionedRange, ...],
    *,
    source_sha256: str,
) -> AdviceRuntime:
    resolved = position.live_resolver.resolve(
        ReplayDecisionAddress.parse(position.address), position.viewer
    )
    replay = position.session.canonical_replay()
    observation_abi, action_abi, possible_world_abi = runtime_advisor_abis()
    identity = AdvisorIdentity(
        source_replay_id=replay.replay_id,
        source_replay_sha256=position.source_replay_sha256,
        match_id=replay.match_id,
        world_id="w2",
        content_sha256=resolved.content_sha256,
        observation_abi=observation_abi,
        action_abi=action_abi,
        possible_world_abi=possible_world_abi,
        information_boundary="historical_viewer",
        planner_id="determinized_puct",
        evaluator_id="uniform_random_terminal",
        artifact=CodeSourceArtifact(
            kind="code_source", source_bundle_sha256=source_sha256
        ),
        compute=AdvisorComputeIdentity(
            id=COMPUTE_ID,
            simulations_per_scenario=SIMULATIONS,
            sampled_worlds=SAMPLED_WORLDS,
            c_puct=C_PUCT,
            max_steps=MAX_STEPS,
            branch_driver_id=SELECTED_BRANCH_DRIVER_ID,
        ),
        seed=AdvisorSeedIdentity(
            plan_id="paired-belief-inverse-cdf-v1",
            root_seed=SELECTED_SEED,
            derivation_id="same-uniform-draws-per-scenario-v1",
        ),
    )
    scenarios = (
        _scenario(position, identity, conditions[0].scenario_id, 1),
        _scenario(position, identity, conditions[1].scenario_id, 2),
    )
    payloads: dict[str, BeliefDistributionPayload] = {}
    for scenario, condition in zip(scenarios, conditions, strict=True):
        provenance_identity = (
            f"player_authored:{scenario.author_viewer_id}:"
            f"{scenario.provenance.created_at_table_revision}"
        )
        payloads[scenario.id] = BeliefDistributionPayload(
            space_identity=resolved.world_space.identity,
            belief_model_id=condition.model_id,
            weights=list(condition.weights),
            distribution_sha256=belief_distribution_sha256(
                resolved.world_space.identity,
                condition.model_id,
                condition.weights,
                provenance_identity,
            ),
            provenance_kind="player_authored",
            provenance_identity=provenance_identity,
        )
    registered = RegisteredAdvisor(identity, ADVISOR_SOURCE_PATHS)
    belief_resolver = StaticBeliefDistributionResolver(payloads)
    request = AdviceRequest(
        address=position.address,
        contract="advice-v1",
        viewer=position.viewer,
        scenario=scenarios[0],
        comparison_scenario=scenarios[1],
        advisor_identity=identity,
    )
    return AdviceRuntime(
        position=position,
        provider=AdviceProvider(
            registered=registered,
            decision_resolver=position.live_resolver,
            belief_resolver=belief_resolver,
        ),
        study_provider=AdviceProvider(
            registered=registered,
            decision_resolver=position.study_resolver,
            belief_resolver=belief_resolver,
        ),
        request=request,
        conditions=conditions,
    )


def _assert_branch_receipts(result: ConditionalStrategyResult) -> None:
    for condition in result.conditions:
        receipt = condition.branch_receipt
        if receipt.get("counters", {}).get("indexed_fallbacks") != 0:
            raise RuntimeError("selected branch used an indexed fallback")
        if receipt.get("reconciliation") != {
            "per_site_and_total": True,
            "zero_unmeasured_fallback": True,
        }:
            raise RuntimeError("selected branch receipt did not reconcile")


def _assert_positive_response(payload: bytes) -> dict[str, Any]:
    response = parse_advice_response_bytes(payload)
    if response.status != "ok" or response.reason is not None:
        raise RuntimeError(f"positive advice unavailable: {response.reason}")
    if response.strategy is None or response.strategy.comparison is None:
        raise RuntimeError("positive advice has no complete comparison")
    if len(response.strategy.scenarios) != 2:
        raise RuntimeError("positive advice must contain two scenarios")
    labels = {offer.id: offer.label for offer in response.offers}
    top_labels: list[str] = []
    visit_margins: list[int] = []
    for scenario in response.strategy.scenarios:
        visits = [action.visits for action in scenario.actions]
        top_index, margin = _winner(visits)
        top = scenario.actions[top_index]
        top_labels.append(labels[top.offer_id])
        visit_margins.append(int(margin))
        q_values: list[float] = []
        for action in scenario.actions:
            if action.visits <= 0:
                raise RuntimeError("fixture contains an unvisited semantic offer")
            if (
                action.q.status != "available"
                or action.robustness.status != "available"
                or action.uncertainty.status != "available"
            ):
                raise RuntimeError("fixture offer lacks complete viewer-safe evidence")
            q_values.append(float(action.q.value))
        q_top_index, _ = _winner(q_values)
        if q_top_index != top_index:
            raise RuntimeError("fixture Q winner does not agree with visit winner")
        if scenario.root_value.status != "available":
            raise RuntimeError("fixture root value is unavailable")
        if scenario.root_uncertainty.status != "available":
            raise RuntimeError("fixture root uncertainty is unavailable")
    if top_labels != ["Pass priority", "Cast Pyroclasm"]:
        raise RuntimeError(f"fixture direction drifted: {top_labels!r}")
    if any(margin < MIN_VISIT_MARGIN for margin in visit_margins):
        raise RuntimeError(f"fixture visit margin is too small: {visit_margins!r}")
    cap_hits = sum(row.cap_hits for row in response.strategy.scenarios)
    total_simulations = sum(row.simulations for row in response.strategy.scenarios)
    cap_rate = cap_hits / total_simulations
    if cap_rate > MAX_CAP_RATE:
        raise RuntimeError(f"fixture cap rate {cap_rate:.6f} exceeds {MAX_CAP_RATE}")
    return {
        "top_action_changed": 1.0,
        "top_labels": top_labels,
        "visit_margins": visit_margins,
        "cap_hits": cap_hits,
        "cap_rate": cap_rate,
        "response_sha256": response.response_sha256,
    }


def _run_measured_search(
    position: PositionRuntime,
    conditions: tuple[ConditionedRange, ...],
    *,
    seed: int,
    branch_audit: bool,
) -> tuple[ConditionalStrategyResult, dict[str, Any]]:
    started = time.perf_counter_ns()
    result = run_search(
        position,
        conditions,
        seed=seed,
        branch_audit=branch_audit,
    )
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    return result, {
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def generate() -> tuple[VersionedAdviceFixture, dict[str, Any]]:
    """Regenerate the full curated suite and the selected positive fixture."""

    _assert_frozen_evidence_unchanged()
    suite: list[dict[str, Any]] = []
    suite_measurements: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        position = build_position(spec)
        try:
            conditions = build_conditioned_ranges(position)
            receipts = condition_receipts(position, conditions)
            for seed in SEARCH_SEEDS:
                result, measurement = _run_measured_search(
                    position,
                    conditions,
                    seed=seed,
                    branch_audit=False,
                )
                suite.append(
                    {
                        "candidate_id": spec.candidate_id,
                        "seed": seed,
                        "position": asdict(spec),
                        "decision_address": position.address,
                        "source_replay_sha256": position.source_replay_sha256,
                        "condition_receipts": receipts,
                        "result": summarize_result(result),
                    }
                )
                suite_measurements.append(
                    {
                        "candidate_id": spec.candidate_id,
                        "seed": seed,
                        **measurement,
                    }
                )
        finally:
            position.close()

    selected_spec = next(
        spec for spec in CANDIDATES if spec.candidate_id == SELECTED_CANDIDATE_ID
    )
    position = build_position(selected_spec)
    try:
        conditions = build_conditioned_ranges(position)
        first, first_measurement = _run_measured_search(
            position,
            conditions,
            seed=SELECTED_SEED,
            branch_audit=True,
        )
        second, second_measurement = _run_measured_search(
            position,
            conditions,
            seed=SELECTED_SEED,
            branch_audit=True,
        )
        if canonical_result_json(first) != canonical_result_json(second):
            raise RuntimeError("audited ConditionalStrategyResult bytes differ")
        _assert_branch_receipts(first)
        source_sha256 = source_bundle_sha256(ADVISOR_SOURCE_PATHS)
        runtime = build_advice_runtime(
            position,
            conditions,
            source_sha256=source_sha256,
        )
        live_started = time.perf_counter_ns()
        live = runtime.provider.advise(runtime.request, recompute=True)
        live_measurement = {
            "elapsed_seconds": (time.perf_counter_ns() - live_started) / 1_000_000_000,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        study_started = time.perf_counter_ns()
        study = runtime.study_provider.advise(runtime.request, recompute=True)
        study_measurement = {
            "elapsed_seconds": (time.perf_counter_ns() - study_started) / 1_000_000_000,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        if live != study:
            raise RuntimeError("live and Study public AdviceResponse bytes differ")
        positive = _assert_positive_response(live)
        fixture = VersionedAdviceFixture(
            request=runtime.request,
            response=parse_advice_response_bytes(live),
        )
        fixture_bytes = (
            json.dumps(fixture.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        ).encode()
        deterministic = {
            "schema_version": 1,
            "claim": (
                "one_post_hoc_curated_viewer_safe_flip_no_method_strength_or_"
                "stability_claim"
            ),
            "selection": {
                "kind": "post_hoc_fixture_curation",
                "candidate_id": SELECTED_CANDIDATE_ID,
                "seed": SELECTED_SEED,
                "all_observed_cells_retained": True,
                "prospective_or_stability_evidence": False,
            },
            "compute": {
                "id": COMPUTE_ID,
                "simulations_per_scenario": SIMULATIONS,
                "sampled_worlds_per_scenario": SAMPLED_WORLDS,
                "c_puct": C_PUCT,
                "max_steps": MAX_STEPS,
                "planner": "determinized_puct",
                "evaluator": "uniform_random_terminal",
                "branch_driver_id": SELECTED_BRANCH_DRIVER_ID,
                "seeds": list(SEARCH_SEEDS),
            },
            "suite": suite,
            "selected_audit": {
                "first_result_sha256": result_sha256(first),
                "second_result_sha256": result_sha256(second),
                "canonical_results_identical": True,
                "root_state_digest": position.root_state_digest,
                "branch_receipts_reconciled": True,
                "condition_receipts": condition_receipts(position, conditions),
            },
            "serving": {
                "request_sha256": advice_request_sha256(runtime.request),
                "response_sha256": fixture.response.response_sha256,
                "source_bundle_sha256": source_sha256,
                "live_study_public_bytes_identical": True,
                "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
                **positive,
            },
            "frozen_evidence": {
                "int12_fixture_sha256": INT12_FIXTURE_SHA256,
                "checkpoint_fixture_sha256": CHECKPOINT_FIXTURE_SHA256,
                "unchanged": True,
            },
        }
        artifact = {
            "schema_version": 1,
            "deterministic_sha256": hashlib.sha256(
                _canonical_bytes(deterministic)
            ).hexdigest(),
            "deterministic": deterministic,
            "measurement": {
                "excluded_from_deterministic_hashes_and_equality": True,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "profile": _measurement_profile(),
                "suite": suite_measurements,
                "selected_audited_results": [
                    first_measurement,
                    second_measurement,
                ],
                "live_provider": live_measurement,
                "study_provider": study_measurement,
            },
        }
        return fixture, artifact
    finally:
        position.close()


def _assert_checked_endpoint(fixture: VersionedAdviceFixture) -> None:
    load_flip_versioned_advice_fixture.cache_clear()
    checked = load_flip_versioned_advice_fixture()
    if checked != fixture:
        raise RuntimeError("checked INT-15 fixture differs from regeneration")
    expected = serialize_advice_response(fixture.response)
    adapter = request_versioned_fixture_advice(fixture.request)
    adapter_response = parse_advice_response_bytes(adapter)
    if adapter_response.status != "ok" or adapter_response.reason is not None:
        raise RuntimeError(
            f"checked fixture adapter is unavailable: {adapter_response.reason}"
        )
    if adapter != expected:
        raise RuntimeError("checked fixture adapter bytes differ")
    with TestClient(app) as client:
        endpoint = client.post(
            "/api/advice", json=fixture.request.model_dump(mode="json")
        )
    if endpoint.status_code != 200:
        raise RuntimeError(f"checked endpoint returned HTTP {endpoint.status_code}")
    response = parse_advice_response_bytes(endpoint.content)
    if response.status != "ok" or response.strategy is None:
        raise RuntimeError(f"checked endpoint is unavailable: {response.reason}")
    if response.strategy.comparison is None:
        raise RuntimeError("checked endpoint omitted the comparison")
    if endpoint.content != expected:
        raise RuntimeError("checked endpoint bytes differ from fixture")


def verify_checked_evidence() -> tuple[VersionedAdviceFixture, dict[str, Any]]:
    """Verify the retained result without rerunning post-hoc fixture curation."""

    if not FLIP_VERSIONED_FIXTURE_PATH.is_file() or not RESULT_PATH.is_file():
        raise RuntimeError("checked INT-15 evidence has not been generated")
    load_flip_versioned_advice_fixture.cache_clear()
    fixture = load_flip_versioned_advice_fixture()
    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    deterministic = artifact.get("deterministic")
    if not isinstance(deterministic, dict):
        raise RuntimeError("checked INT-15 deterministic result is missing")
    if (
        artifact.get("deterministic_sha256")
        != hashlib.sha256(_canonical_bytes(deterministic)).hexdigest()
    ):
        raise RuntimeError("checked INT-15 deterministic result hash differs")

    expected_cells = {
        (spec.candidate_id, seed) for spec in CANDIDATES for seed in SEARCH_SEEDS
    }
    suite = deterministic.get("suite")
    if (
        not isinstance(suite, list)
        or {(str(row.get("candidate_id")), int(row.get("seed", -1))) for row in suite}
        != expected_cells
    ):
        raise RuntimeError("checked INT-15 curated suite is incomplete")
    selection = deterministic.get("selection")
    if selection != {
        "kind": "post_hoc_fixture_curation",
        "candidate_id": SELECTED_CANDIDATE_ID,
        "seed": SELECTED_SEED,
        "all_observed_cells_retained": True,
        "prospective_or_stability_evidence": False,
    }:
        raise RuntimeError("checked INT-15 curation boundary differs")

    selected_audit = deterministic.get("selected_audit", {})
    if (
        selected_audit.get("canonical_results_identical") is not True
        or selected_audit.get("first_result_sha256")
        != selected_audit.get("second_result_sha256")
        or selected_audit.get("branch_receipts_reconciled") is not True
    ):
        raise RuntimeError("checked INT-15 selected result identity differs")
    measurement = artifact.get("measurement", {})
    if measurement.get("excluded_from_deterministic_hashes_and_equality") is not True:
        raise RuntimeError("checked INT-15 measurement boundary differs")
    deterministic_json = json.dumps(deterministic, sort_keys=True)
    if (
        "elapsed_seconds" in deterministic_json
        or "peak_rss_bytes" in deterministic_json
    ):
        raise RuntimeError("checked INT-15 deterministic result contains measurements")

    serving = deterministic.get("serving", {})
    fixture_bytes = FLIP_VERSIONED_FIXTURE_PATH.read_bytes()
    if serving.get("fixture_sha256") != hashlib.sha256(fixture_bytes).hexdigest():
        raise RuntimeError("checked INT-15 fixture hash differs")
    if serving.get("request_sha256") != advice_request_sha256(fixture.request):
        raise RuntimeError("checked INT-15 request hash differs")
    if serving.get("response_sha256") != fixture.response.response_sha256:
        raise RuntimeError("checked INT-15 response hash differs")
    if serving.get("source_bundle_sha256") != source_bundle_sha256(
        ADVISOR_SOURCE_PATHS
    ):
        raise RuntimeError("checked INT-15 source bundle differs")
    if serving.get("live_study_public_bytes_identical") is not True:
        raise RuntimeError("checked INT-15 live/Study identity differs")
    retained_positive = _assert_positive_response(
        serialize_advice_response(fixture.response)
    )
    for key, value in retained_positive.items():
        if serving.get(key) != value:
            raise RuntimeError(f"checked INT-15 serving field differs: {key}")
    _assert_checked_endpoint(fixture)
    _assert_frozen_evidence_unchanged()
    return fixture, artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-fixture", action="store_true")
    mode.add_argument("--update-fixture", action="store_true")
    args = parser.parse_args()

    if args.update_fixture:
        fixture, artifact = generate()
        fixture_text = (
            json.dumps(fixture.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        artifact_text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
        FLIP_VERSIONED_FIXTURE_PATH.write_text(fixture_text, encoding="utf-8")
        RESULT_PATH.write_text(artifact_text, encoding="utf-8")
        _assert_checked_endpoint(fixture)
        _assert_frozen_evidence_unchanged()
    else:
        fixture, artifact = verify_checked_evidence()
    print(
        json.dumps(
            {
                "status": "updated" if args.update_fixture else "verified",
                "selection": "post_hoc_fixture_curation",
                "top_action_changed": 1.0,
                "top_labels": artifact["deterministic"]["serving"]["top_labels"],
                "visit_margins": artifact["deterministic"]["serving"]["visit_margins"],
                "response_sha256": fixture.response.response_sha256,
                "deterministic_result_sha256": artifact["deterministic_sha256"],
                "endpoint_status": "ok",
                "live_study_public_bytes_identical": True,
                "measurement_excluded_from_deterministic_identity": True,
                "int12_fixture_unchanged": True,
                "checkpoint_fixture_unchanged": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
