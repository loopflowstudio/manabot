"""Generate and verify one retained-checkpoint advice-v1 fixture.

The fixture starts from Etude's checked UR Lessons versus GW Allies authored
match. Public bytes contain one canonical viewer-safe decision and aggregate
policy/search evidence. Retained roots, materialized worlds, branch receipts,
and derived RNG state remain authority-private.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from etude.advice import (
    ADVISOR_SOURCE_PATHS,
    CHECKPOINT_VERSIONED_FIXTURE_PATH,
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
)
from etude.advice_identity import (
    AbiIdentity,
    AdvisorComputeIdentity,
    AdvisorIdentity,
    AdvisorSeedIdentity,
    CheckpointArtifact,
)
from etude.authored_match_parity import RECEIPT_PATH as PARITY_RECEIPT_PATH
from etude.authored_match_receipt import (
    DEFAULT_RECEIPT_PATH as AUTHORITY_RECEIPT_PATH,
    play_fixed_authored_match,
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
from manabot.sim.search_runtime import (
    RetainedInt7EvidenceSnapshot,
    retained_int7_evidence_snapshot,
    retained_int7_policy_only_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "protocol" / "advice-v1.schema.json"
FIXTURE_SEED = 197
SELECTED_DECISION_ORDINAL = 1


@dataclass
class CheckpointAdvisorRuntime:
    session: GameSession
    provider: AdviceProvider
    study_provider: AdviceProvider
    request: AdviceRequest
    authority_receipt_sha256: str
    parity_receipt_sha256: str


@dataclass(frozen=True)
class RetainedEvidenceVerificationReceipt:
    """Before/after proof that advice did not change retained INT-7 evidence."""

    before: RetainedInt7EvidenceSnapshot
    after: RetainedInt7EvidenceSnapshot

    @property
    def no_write_or_generation_verified(self) -> bool:
        return self.before == self.after


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_identity(
    *,
    session: GameSession,
    source_replay_sha256: str,
    content_sha256: str,
) -> AdvisorIdentity:
    registration = retained_int7_policy_only_checkpoint(FIXTURE_SEED)
    observation_abi, action_abi, possible_world_abi = runtime_advisor_abis()
    return AdvisorIdentity(
        source_replay_id=session.canonical_replay().replay_id,
        source_replay_sha256=source_replay_sha256,
        match_id=session.match_id,
        world_id=registration.world_id,
        content_sha256=content_sha256,
        observation_abi=observation_abi,
        action_abi=action_abi,
        possible_world_abi=possible_world_abi,
        information_boundary="historical_viewer",
        planner_id="determinized_puct",
        evaluator_id="checkpoint_policy_neutral_value",
        artifact=CheckpointArtifact(
            kind="checkpoint",
            checkpoint_id=registration.checkpoint_id,
            checkpoint_sha256=registration.checkpoint_sha256,
            checkpoint_bytes=registration.checkpoint_bytes,
            manifest_sha256=registration.manifest_sha256,
            training_seed=registration.training_seed,
            observation_abi=AbiIdentity(
                name="manabot_observation",
                version=registration.world_id,
                sha256=registration.observation_abi_sha256,
            ),
            action_abi=AbiIdentity(
                name="manabot_action",
                version=registration.world_id,
                sha256=registration.action_abi_sha256,
            ),
            value_mode=registration.value_mode,
        ),
        compute=AdvisorComputeIdentity(
            id="int7-policy-only-puct-s32-w4-v1",
            simulations_per_scenario=registration.simulations,
            sampled_worlds=registration.sampled_worlds,
            c_puct=registration.c_puct,
            max_steps=registration.max_steps,
            branch_driver_id=registration.branch_driver_id,
        ),
        seed=AdvisorSeedIdentity(
            plan_id="historical-compatible-world-belief-v1",
            root_seed=FIXTURE_SEED,
            derivation_id="conditional-search-paired-inverse-cdf-v1",
        ),
    )


def build_checkpoint_runtime() -> CheckpointAdvisorRuntime:
    """Bind one exact authored-match decision to the retained INT-7 checkpoint."""

    authority_receipt_sha256 = _file_sha256(AUTHORITY_RECEIPT_PATH)
    parity_receipt_sha256 = _file_sha256(PARITY_RECEIPT_PATH)
    session, _ = play_fixed_authored_match()
    replay = session.canonical_replay()
    rows = [
        row for row in replay.decisions if int(row.ordinal) == SELECTED_DECISION_ORDINAL
    ]
    if len(rows) != 1:
        raise RuntimeError("selected authored-match decision is unavailable")
    row = rows[0]
    if row.viewer != 0 or len(row.frame.offers) != 2:
        raise RuntimeError("selected authored-match decision identity drifted")
    address = ReplayDecisionAddress.from_decision(replay, row).serialize()
    projection_sha256 = canonical_projection_sha256(project_replay(replay, row.viewer))
    viewer = ViewerIdentity(
        viewer_id="viewer.authored-match.pilot",
        table_id=session.match_id,
        rules_viewer=0,
    )
    live_resolver = LiveAdvisorDecisionResolver(session, projection_sha256)
    study_resolver = StudyAdvisorDecisionResolver(
        session.study_fork_provider(), projection_sha256
    )
    resolved = live_resolver.resolve(ReplayDecisionAddress.parse(address), viewer)
    identity = _checkpoint_identity(
        session=session,
        source_replay_sha256=projection_sha256,
        content_sha256=resolved.content_sha256,
    )
    scenario = BeliefScenario(
        id="authored-compatible-world-prior",
        author_viewer_id=viewer.viewer_id,
        source=BeliefSource(
            decision_address=address,
            gam6_scenario_id="authored-compatible-world-prior",
            advice_identity=identity,
        ),
        audience=PersonalAudience(kind="personal"),
        provenance=PlayerAuthoredBeliefProvenance(created_at_table_revision=1),
    )
    weights = [float(world.weight) for world in resolved.world_space.worlds]
    provenance_identity = f"player_authored:{viewer.viewer_id}:1"
    belief = BeliefDistributionPayload(
        space_identity=resolved.world_space.identity,
        belief_model_id="compatible-world-prior/v1",
        weights=weights,
        distribution_sha256=belief_distribution_sha256(
            resolved.world_space.identity,
            "compatible-world-prior/v1",
            weights,
            provenance_identity,
        ),
        provenance_kind="player_authored",
        provenance_identity=provenance_identity,
    )
    belief_resolver = StaticBeliefDistributionResolver({scenario.id: belief})
    registered = RegisteredAdvisor(identity, ADVISOR_SOURCE_PATHS)
    request = AdviceRequest(
        address=address,
        contract="advice-v1",
        viewer=viewer,
        scenario=scenario,
        advisor_identity=identity,
    )
    return CheckpointAdvisorRuntime(
        session=session,
        provider=AdviceProvider(
            registered=registered,
            decision_resolver=live_resolver,
            belief_resolver=belief_resolver,
        ),
        study_provider=AdviceProvider(
            registered=registered,
            decision_resolver=study_resolver,
            belief_resolver=belief_resolver,
        ),
        request=request,
        authority_receipt_sha256=authority_receipt_sha256,
        parity_receipt_sha256=parity_receipt_sha256,
    )


def _assert_complete_and_private(payload: bytes) -> None:
    response = parse_advice_response_bytes(payload)
    if response.status != "ok" or response.strategy is None:
        raise RuntimeError(f"checkpoint advice unavailable: {response.reason}")
    if len(response.strategy.scenarios) != 1:
        raise RuntimeError("checkpoint fixture must contain one belief scenario")
    scenario = response.strategy.scenarios[0]
    offer_ids = [offer.offer_id for offer in response.strategy.offers]
    if [action.offer_id for action in scenario.actions] != offer_ids:
        raise RuntimeError("checkpoint evidence does not cover semantic offers")
    for action in scenario.actions:
        if any(
            quantity.status != "available"
            for quantity in (action.q, action.robustness, action.uncertainty)
        ):
            raise RuntimeError(
                f"offer {action.offer_id} lacks complete checkpoint evidence"
            )
    if (
        scenario.root_value.status != "available"
        or scenario.root_uncertainty.status != "available"
    ):
        raise RuntimeError("checkpoint root evidence is incomplete")

    public = response.model_dump(mode="json")
    forbidden_keys = {
        "authority_private",
        "root_state_digest",
        "world_index",
        "world_q_values",
        "world_root_values",
        "world_visit_counts",
        "sampled_indexes",
        "opponent_hand",
        "library",
        "actual_query_truth",
        "derived_seed",
        "rng_tapes",
        "branch_receipt",
        "checkpoint_path",
    }

    def contains_forbidden(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                key in forbidden_keys or contains_forbidden(child)
                for key, child in value.items()
            )
        if isinstance(value, list):
            return any(contains_forbidden(child) for child in value)
        return False

    if contains_forbidden(public):
        raise RuntimeError("checkpoint advice leaked authority-private evidence")


def checkpoint_fixture(
    runtime: CheckpointAdvisorRuntime,
) -> tuple[VersionedAdviceFixture, bytes, RetainedEvidenceVerificationReceipt]:
    """Prove fresh parity plus Game-root and retained-evidence immutability."""

    retained_before = retained_int7_evidence_snapshot((FIXTURE_SEED,))
    assert runtime.request.viewer is not None
    parsed = ReplayDecisionAddress.parse(runtime.request.address)
    study_forks = runtime.session.study_fork_provider()
    source_before = study_forks.resolve_advisor_decision(
        parsed.serialize(), runtime.request.viewer.rules_viewer
    ).source_digest

    live_first = runtime.provider.advise(runtime.request, recompute=True)
    study = runtime.study_provider.advise(runtime.request, recompute=True)
    live_second = runtime.provider.advise(runtime.request, recompute=True)
    if live_first != study or live_first != live_second:
        raise RuntimeError("fresh live and Study advice bytes differ")

    source_after = study_forks.resolve_advisor_decision(
        parsed.serialize(), runtime.request.viewer.rules_viewer
    ).source_digest
    if source_after != source_before:
        raise RuntimeError("checkpoint advice mutated the retained Game root")
    if (
        _file_sha256(AUTHORITY_RECEIPT_PATH) != runtime.authority_receipt_sha256
        or _file_sha256(PARITY_RECEIPT_PATH) != runtime.parity_receipt_sha256
    ):
        raise RuntimeError("checkpoint advice changed an authored-match receipt")
    retained_after = retained_int7_evidence_snapshot((FIXTURE_SEED,))
    retained_receipt = RetainedEvidenceVerificationReceipt(
        before=retained_before,
        after=retained_after,
    )
    if not retained_receipt.no_write_or_generation_verified:
        raise RuntimeError("checkpoint advice changed retained INT-7 evidence")
    _assert_complete_and_private(live_first)
    return (
        VersionedAdviceFixture(
            request=runtime.request,
            response=parse_advice_response_bytes(live_first),
        ),
        live_first,
        retained_receipt,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-fixture", action="store_true")
    mode.add_argument("--update-fixture", action="store_true")
    args = parser.parse_args()

    runtime = build_checkpoint_runtime()
    fixture, response_bytes, retained_receipt = checkpoint_fixture(runtime)
    encoded_fixture = (
        json.dumps(fixture.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    encoded_schema = json.dumps(advice_schema(), indent=2, sort_keys=True) + "\n"
    if args.update_fixture:
        CHECKPOINT_VERSIONED_FIXTURE_PATH.write_text(encoded_fixture, encoding="utf-8")
        SCHEMA_PATH.write_text(encoded_schema, encoding="utf-8")
    else:
        if (
            CHECKPOINT_VERSIONED_FIXTURE_PATH.read_text(encoding="utf-8")
            != encoded_fixture
        ):
            raise RuntimeError("checked checkpoint advice fixture drifted")
        if SCHEMA_PATH.read_text(encoding="utf-8") != encoded_schema:
            raise RuntimeError("checked advice schema differs from regeneration")

    response = parse_advice_response_bytes(response_bytes)
    assert response.strategy is not None
    artifact = runtime.request.advisor_identity.artifact
    assert isinstance(artifact, CheckpointArtifact)
    print(
        json.dumps(
            {
                "status": "available",
                "match_id": runtime.request.advisor_identity.match_id,
                "decision_address": runtime.request.address,
                "checkpoint_sha256": artifact.checkpoint_sha256,
                "response_bytes": len(response_bytes),
                "response_sha256": response.response_sha256,
                "offers": len(response.strategy.offers),
                "live_study_fresh_bytes_identical": True,
                "retained_root_unchanged": True,
                "authority_receipt_sha256": runtime.authority_receipt_sha256,
                "parity_receipt_sha256": runtime.parity_receipt_sha256,
                "retained_evidence_no_write_or_generation_verified": (
                    retained_receipt.no_write_or_generation_verified
                ),
                "retained_evidence_before_sha256": (
                    retained_receipt.before.receipt_sha256
                ),
                "retained_evidence_after_sha256": (
                    retained_receipt.after.receipt_sha256
                ),
                "retained_manifest_digest": (retained_receipt.before.manifest_digest),
                "retained_manifest_file_sha256": (
                    retained_receipt.before.manifest_file.sha256
                ),
                "retained_manifest_bytes": (
                    retained_receipt.before.manifest_file.bytes
                ),
                "retained_selected_checkpoint_file_sha256": (
                    retained_receipt.before.selected_checkpoints[0].file.sha256
                ),
                "retained_selected_checkpoint_bytes": (
                    retained_receipt.before.selected_checkpoints[0].file.bytes
                ),
                "retained_tree_content_sha256": (
                    retained_receipt.before.retention_tree_content_sha256
                ),
                "retained_tree_files": (retained_receipt.before.retention_tree_files),
                "retained_tree_bytes": (retained_receipt.before.retention_tree_bytes),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
