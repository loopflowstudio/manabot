"""Study export tests for actual replayed Teacher-1 decisions."""

from etude.replay_index import ReplayDecisionAddress
from manabot.sim.study_evidence import build_study_artifact
from manabot.sim.teacher1_evidence import (
    record_teacher_trajectories,
    runtime_fingerprints,
)


def test_replayed_search_exports_viewer_safe_distinct_evidence() -> None:
    runtime = runtime_fingerprints(seed=223)
    audit = record_teacher_trajectories(
        games=1,
        simulations=8,
        worlds=2,
        c_puct=1.5,
        seed=223,
        content_hash=runtime["experience_content_hash"],
        asset_manifest_hash=runtime["asset_manifest_hash"],
    )
    decision = next(
        decision
        for game in audit["games"]
        for decision in game["decisions"]
        if len(decision["frame"]["offers"]) >= 2
        and all(value > 0 for value in decision["search"]["visit_counts"])
    )
    offers = decision["frame"]["offers"]
    uniform = {int(offer["id"]): 1.0 / len(offers) for offer in offers}

    artifact = build_study_artifact(
        audit,
        policy_mass_by_offer=uniform,
        source_replay_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        engine_build_sha256=runtime["engine_extension_sha256"],
        content_pack_id="w2-interactive-deck",
        content_pack_version="1",
        model_id="visit-policy-value-seed-197",
        producer_version="test",
        generated_at="2026-07-17T00:00:00Z",
    )

    landmark = artifact.landmarks[0]
    address = ReplayDecisionAddress.parse(landmark.decision_id)
    assert address.replay_id == artifact.identity.source_replay_id
    assert address.match_id == artifact.identity.match_id
    assert landmark.frame.projection.opponent.hand == []
    assert len(landmark.alternatives) >= 2
    assert sum(row.probability for row in landmark.evidence.policy_mass) == 1.0
    assert {row.alternative for row in landmark.evidence.visits} == {
        row.alternative for row in landmark.evidence.search_value
    }
    assert all(
        row.method == "between-determinized-world-q-standard-error/v1"
        for row in landmark.evidence.uncertainty
    )
    serialized = artifact.model_dump(mode="json")
    assert "deal_seed" not in str(serialized)
    assert "call_seed" not in str(serialized)
