"""Generate and measure the checked belief-conditioned advice-v1 slice.

The fixture contains only one canonical request and its viewer-safe response.
Exact distribution weights, materialized worlds, branch receipts, and sampled
indexes remain process-private.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from etude.advice import (
    ADVISOR_SOURCE_PATHS,
    VERSIONED_FIXTURE_PATH,
    AdviceProvider,
    AdviceRequest,
    BeliefDistributionPayload,
    LiveAdvisorDecisionResolver,
    RegisteredAdvisor,
    StaticBeliefDistributionResolver,
    StudyAdvisorDecisionResolver,
    VersionedAdviceFixture,
    advice_schema,
    belief_distribution_sha256,
    parse_advice_response_bytes,
    runtime_advisor_abis,
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
from etude.server import GameSession
from etude.testing_house_protocol import (
    BeliefScenario,
    BeliefSource,
    PersonalAudience,
    PlayerAuthoredBeliefProvenance,
    ViewerIdentity,
)
from manabot.sim.search_branch import SELECTED_BRANCH_DRIVER_ID

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "protocol" / "advice-v1.schema.json"
MEASUREMENT_PATH = (
    REPO_ROOT
    / "experiments"
    / "data"
    / "int-12-belief-strategy-advisor-measurement-v1.json"
)
ROOT_SEED = 197
TARGET_OFFER_LABEL = "Cast Pyroclasm"
TARGET_OCCURRENCE = 1


@dataclass
class AdvisorRuntime:
    temporary_directory: TemporaryDirectory[str]
    session: GameSession
    provider: AdviceProvider
    study_provider: AdviceProvider
    request: AdviceRequest


def _commit_advisor_root(session: GameSession) -> object:
    occurrences = 0
    for command_index in range(64):
        if session.obs is None or session.obs.game_over:
            break
        frame = session._experience_frame()
        offers = list(frame["offers"])
        if any(offer["label"] == TARGET_OFFER_LABEL for offer in offers):
            occurrences += 1
        target = occurrences == TARGET_OCCURRENCE
        preferred = [
            offer
            for offer in offers
            if not str(offer["label"]).lower().startswith(("pass", "do not", "decline"))
        ]
        selected = preferred[0] if preferred else offers[0]
        committed_ordinal = len(session.canonical_decisions)
        outcome = session.hero_command(
            {
                "command_id": f"int12-command-{command_index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": selected["id"],
                "answers": [],
            }
        )
        if outcome["status"] != "accepted":
            raise RuntimeError("fixture command was not accepted")
        if target:
            return session.canonical_decisions[committed_ordinal]
    raise RuntimeError("fixture match did not reach the advisor combat decision")


def _authored_payload(
    *,
    scenario: BeliefScenario,
    space_identity: str,
    belief_model_id: str,
    weights: list[float],
) -> BeliefDistributionPayload:
    provenance = scenario.provenance
    if provenance.kind != "player_authored":
        raise RuntimeError("fixture payload must be player-authored")
    provenance_identity = (
        f"player_authored:{scenario.author_viewer_id}:"
        f"{provenance.created_at_table_revision}"
    )
    return BeliefDistributionPayload(
        space_identity=space_identity,
        belief_model_id=belief_model_id,
        weights=weights,
        distribution_sha256=belief_distribution_sha256(
            space_identity,
            belief_model_id,
            weights,
            provenance_identity,
        ),
        provenance_kind="player_authored",
        provenance_identity=provenance_identity,
    )


def build_runtime() -> AdvisorRuntime:
    temporary_directory = TemporaryDirectory(prefix="int-12-advisor-")
    session = GameSession(
        Path(temporary_directory.name),
        id_factory=lambda kind: f"int12-{kind}",
        clock=lambda: "2026-07-18T00:00:00+00:00",
        villain_offer_policy=lambda context: int(context.offers[-1]["id"]),
    )
    session.new_game(
        {
            "villain_type": "passive",
            "seed": ROOT_SEED,
            "hero_deck": "interactive",
            "villain_deck": "interactive",
            "auto_pass": False,
        }
    )
    row = _commit_advisor_root(session)
    replay = session.canonical_replay()
    address = ReplayDecisionAddress.from_decision(replay, row).serialize()
    projection_sha256 = canonical_projection_sha256(project_replay(replay, 0))
    viewer = ViewerIdentity(
        viewer_id="viewer.int12.pilot",
        table_id=session.match_id,
        rules_viewer=0,
    )
    live_decision_resolver = LiveAdvisorDecisionResolver(session, projection_sha256)
    study_decision_resolver = StudyAdvisorDecisionResolver(
        session.study_fork_provider(), projection_sha256
    )
    resolved = live_decision_resolver.resolve(
        ReplayDecisionAddress.parse(address), viewer
    )
    observation_abi, action_abi, possible_world_abi = runtime_advisor_abis()
    identity = AdvisorIdentity(
        source_replay_id=replay.replay_id,
        source_replay_sha256=projection_sha256,
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
            kind="code_source",
            source_bundle_sha256=source_bundle_sha256(ADVISOR_SOURCE_PATHS),
        ),
        compute=AdvisorComputeIdentity(
            id="puct-8-traversal-2-world-v1",
            simulations_per_scenario=8,
            sampled_worlds=2,
            c_puct=1.5,
            max_steps=80,
            branch_driver_id=SELECTED_BRANCH_DRIVER_ID,
        ),
        seed=AdvisorSeedIdentity(
            plan_id="paired-belief-inverse-cdf-v1",
            root_seed=ROOT_SEED,
            derivation_id="same-uniform-draws-per-scenario-v1",
        ),
    )

    def scenario(scenario_id: str, revision: int) -> BeliefScenario:
        return BeliefScenario(
            id=scenario_id,
            author_viewer_id=viewer.viewer_id,
            source=BeliefSource(
                decision_address=address,
                gam6_scenario_id=scenario_id,
                advice_identity=identity,
            ),
            audience=PersonalAudience(kind="personal"),
            provenance=PlayerAuthoredBeliefProvenance(
                created_at_table_revision=revision
            ),
        )

    prior_scenario = scenario("authored-compatible-deal-prior", 1)
    bolt_scenario = scenario("authored-bolt-heavy-range", 2)
    prior_weights = [float(world.weight) for world in resolved.world_space.worlds]
    bolt_weights = [
        float(world.weight) if world.count("Lightning Bolt") > 0 else 0.0
        for world in resolved.world_space.worlds
    ]
    payloads = {
        prior_scenario.id: _authored_payload(
            scenario=prior_scenario,
            space_identity=resolved.world_space.identity,
            belief_model_id="compatible-deal-prior/v1",
            weights=prior_weights,
        ),
        bolt_scenario.id: _authored_payload(
            scenario=bolt_scenario,
            space_identity=resolved.world_space.identity,
            belief_model_id="authored-bolt-heavy-range/v1",
            weights=bolt_weights,
        ),
    }
    provider = AdviceProvider(
        registered=RegisteredAdvisor(identity, ADVISOR_SOURCE_PATHS),
        decision_resolver=live_decision_resolver,
        belief_resolver=StaticBeliefDistributionResolver(payloads),
    )
    study_provider = AdviceProvider(
        registered=RegisteredAdvisor(identity, ADVISOR_SOURCE_PATHS),
        decision_resolver=study_decision_resolver,
        belief_resolver=StaticBeliefDistributionResolver(payloads),
    )
    request = AdviceRequest(
        address=address,
        contract="advice-v1",
        viewer=viewer,
        scenario=prior_scenario,
        comparison_scenario=bolt_scenario,
        advisor_identity=identity,
    )
    return AdvisorRuntime(
        temporary_directory,
        session,
        provider,
        study_provider,
        request,
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def _measure(
    runtime: AdvisorRuntime,
    *,
    warmups: int,
    calls: int,
) -> dict[str, Any]:
    provider = runtime.provider
    provider.advise(runtime.request)
    for _ in range(warmups):
        provider.advise(runtime.request)
    cached_ms: list[float] = []
    for _ in range(calls):
        started = time.perf_counter_ns()
        provider.advise(runtime.request)
        cached_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    for _ in range(warmups):
        provider.advise(runtime.request, recompute=True)
    recompute_ms: list[float] = []
    component_ms: dict[str, list[float]] = {}
    last_payload = b""
    for _ in range(calls):
        started = time.perf_counter_ns()
        last_payload = provider.advise(runtime.request, recompute=True)
        recompute_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        for name, value in provider.last_timings_ms.items():
            component_ms.setdefault(name, []).append(float(value))

    response = parse_advice_response_bytes(last_payload)
    assert response.strategy is not None
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim": "engineering_profile_only_no_strength_or_service_slo_claim",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
        },
        "profile": {
            "warmups_per_path": warmups,
            "measured_calls_per_path": calls,
            "cached_p50_ms": statistics.median(cached_ms),
            "cached_p95_ms": _percentile(cached_ms, 0.95),
            "recomputed_p50_ms": statistics.median(recompute_ms),
            "recomputed_p95_ms": _percentile(recompute_ms, 0.95),
            "recomputed_to_cached_p50_ratio": (
                statistics.median(recompute_ms) / statistics.median(cached_ms)
            ),
            "response_bytes": len(last_payload),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "component_p50_ms": {
            name: statistics.median(values)
            for name, values in sorted(component_ms.items())
        },
        "realized_compute": response.strategy.realized_compute,
        "method": {
            "policy": response.strategy.policy_semantic,
            "planner": runtime.request.advisor_identity.planner_id,
            "evaluator": runtime.request.advisor_identity.evaluator_id,
            "compute": runtime.request.advisor_identity.compute.model_dump(mode="json"),
            "seed": runtime.request.advisor_identity.seed.model_dump(mode="json"),
        },
    }


def _fixture(runtime: AdvisorRuntime) -> tuple[VersionedAdviceFixture, bytes]:
    live = runtime.provider.advise(runtime.request, recompute=True)
    study = runtime.study_provider.advise(runtime.request, recompute=True)
    if live != study:
        raise RuntimeError("live and Study provider bytes differ")
    response = parse_advice_response_bytes(live)
    if response.strategy is None or response.strategy.comparison is None:
        raise RuntimeError("fixture did not produce a complete comparison")
    deltas = [
        abs(action.policy_probability)
        for action in response.strategy.comparison.actions
    ]
    if not any(delta > 0.0 for delta in deltas):
        raise RuntimeError("fixture scenarios produced no strategy delta")
    return (
        VersionedAdviceFixture(request=runtime.request, response=response),
        live,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-fixture", action="store_true")
    mode.add_argument("--update-fixture", action="store_true")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--measure-calls", type=int, default=128)
    parser.add_argument("--write-measurement", action="store_true")
    parser.add_argument("--skip-measurement", action="store_true")
    args = parser.parse_args()
    if args.verify_fixture and args.skip_measurement:
        raise SystemExit("fixture verification always measures both provider paths")
    if not args.skip_measurement and (args.warmups < 20 or args.measure_calls < 128):
        raise SystemExit(
            "the registered profile requires at least 20 warmups and 128 measured calls"
        )

    runtime = build_runtime()
    fixture, response_bytes = _fixture(runtime)
    encoded_fixture = (
        json.dumps(fixture.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    encoded_schema = json.dumps(advice_schema(), indent=2, sort_keys=True) + "\n"
    if args.update_fixture:
        VERSIONED_FIXTURE_PATH.write_text(encoded_fixture, encoding="utf-8")
        SCHEMA_PATH.write_text(encoded_schema, encoding="utf-8")
    else:
        if VERSIONED_FIXTURE_PATH.read_text(encoding="utf-8") != encoded_fixture:
            raise RuntimeError("checked advice fixture differs from regeneration")
        if SCHEMA_PATH.read_text(encoding="utf-8") != encoded_schema:
            raise RuntimeError("checked advice schema differs from regeneration")

    measurement = None
    if not args.skip_measurement:
        measurement = _measure(
            runtime,
            warmups=args.warmups,
            calls=args.measure_calls,
        )
    if args.write_measurement and measurement is not None:
        MEASUREMENT_PATH.write_text(
            json.dumps(measurement, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    response = parse_advice_response_bytes(response_bytes)
    assert response.strategy is not None
    assert response.strategy.comparison is not None
    print(
        json.dumps(
            {
                "status": "verified" if args.verify_fixture else "updated",
                "left_belief_sha256": response.strategy.comparison.left_belief_sha256,
                "right_belief_sha256": response.strategy.comparison.right_belief_sha256,
                "response_sha256": response.response_sha256,
                "nonzero_policy_delta": max(
                    abs(action.policy_probability)
                    for action in response.strategy.comparison.actions
                ),
                "live_study_bytes_identical": True,
                "measurement": measurement,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
