from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_rul9_played_workloads",
    ROOT / "experiments/runners/run_rul9_played_workloads.py",
)
assert SPEC is not None and SPEC.loader is not None
rul9 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rul9)


def contract() -> dict:
    return rul9.load_contract()


def token_sample(*, details: bool = False) -> dict:
    sample = {
        "viewer": 0,
        "visible_object_references": 1,
        "expanded_semantic_tokens": 5,
        "max_definition_tokens": 5,
        "overflow_count": 0,
        "projection_failures": 0,
        "unadmitted_visible_definitions": 0,
    }
    if details:
        sample.update(
            {
                "tokens_by_zone": {"hand": 5},
                "tokens_by_definition": {"Mountain": 5},
            }
        )
    return sample


def release_game(surface: str, fallback_names: list[str]) -> dict:
    common = {
        "commands": 132,
        "token_samples": [token_sample()],
        "game_seconds": 0.1,
        "terminal_state_sha256": rul9.parity.TERMINAL_STATE,
        "logical_trace_sha256": "logical",
        "fallback_counters": {name: 0 for name in fallback_names},
    }
    if surface == "live":
        common["protocol_commands"] = [{"duration_ms": 2.0}]
        common["inner_commands"] = [{"duration_ms": 1.0}]
    else:
        common["command_samples"] = [{"duration_ms": 1.0}]
    return common


def training_game(seed: int, ur_seat: int, fallback_names: list[str]) -> dict:
    return {
        "deal_seed": seed,
        "ur_seat": ur_seat,
        "winner": 0,
        "terminal": True,
        "terminal_state_sha256": f"terminal-{seed}",
        "decisions": 1,
        "traversals": 128,
        "game_seconds": 1.0,
        "cpu_seconds": 0.5,
        "rows": [
            {
                "puct_ms": 100.0,
                "command_ms": 1.0,
                "native_actions": 1,
                "token_sample": token_sample(details=True),
            }
        ],
        "branch_counters": {},
        "fallback_counters": {name: 0 for name in fallback_names},
        "logical_trace_sha256": f"logical-{seed}",
    }


def fake_raw(registered: dict) -> dict:
    static = {
        "semantic_pack_sha256": "semantic",
        "learning_schema_sha256": "schema",
        "semantic_ir_sha256": "ir",
        "catalog_active_tokens": 10,
        "definition_tokens": [1, 2],
        "program_tokens": [3, 4],
        "tokens_per_definition": [4, 5],
    }
    repetitions = int(registered["release"]["repetitions"])
    return {
        "release": {
            "correctness": {"parity_summary": {"commands_per_surface": 132}},
            "semantic_static": static,
            "surfaces": {
                surface: [
                    release_game(surface, registered["authority_fallback_counters"])
                    for _ in range(repetitions)
                ]
                for surface in ("live", "headless", "replay")
            },
            "rss": {
                "peak_bytes": 100,
                "sample_count": 2,
                "wall_seconds": 1.0,
                "samples": [
                    {"offset_seconds": 0.0, "rss_bytes": 50},
                    {"offset_seconds": 0.5, "rss_bytes": 100},
                ],
            },
        },
        "training": {
            "semantic_static": static,
            "games": [
                training_game(seed, index % 2, registered["training_fallback_counters"])
                for index, seed in enumerate(registered["training"]["deal_seeds"])
            ],
            "rss": {
                "peak_bytes": 100,
                "sample_count": 2,
                "wall_seconds": 1.0,
                "samples": [
                    {"offset_seconds": 0.0, "rss_bytes": 50},
                    {"offset_seconds": 0.5, "rss_bytes": 100},
                ],
            },
        },
    }


def fake_identity(registered: dict) -> dict:
    return {
        "source": {"algorithm": "test", "files": [], "sha256": "source"},
        "binary": {
            "profile": "release",
            "extension_name": "_managym.so",
            "extension_sha256": "binary",
            "python": "3.12",
            "platform": "test",
            "processor": "test",
            "logical_cpus": 1,
            "cargo": "cargo test",
            "rustc": "rustc test",
            "uv": "uv test",
        },
        "workload": {
            "authority_receipt_sha256": registered["release"][
                "authority_receipt_sha256"
            ],
            "parity_receipt_sha256": registered["release"]["parity_receipt_sha256"],
            "semantic_ir_file_sha256": registered["expected_inputs"][
                "semantic_ir_file_sha256"
            ],
            "semantic_source_file_sha256": registered["expected_inputs"][
                "semantic_source_file_sha256"
            ],
            "learning_schema_file_sha256": registered["expected_inputs"][
                "learning_schema_file_sha256"
            ],
            "content_pack_manifest": {},
            "content_pack_manifest_sha256": "pack",
            "semantic_static": {},
            "ur_deck_sha256": "ur",
            "gw_deck_sha256": "gw",
            "selected_branch_driver": "full_clone/current_game_v1",
        },
    }


def fake_receipt(registered: dict, raw: dict | None = None) -> dict:
    evidence = raw or fake_raw(registered)
    summary = rul9.derive_summary(evidence, registered)
    payload = {
        "schema_version": 1,
        "experiment": rul9.EXPERIMENT_ID,
        "run": {
            "completed_at": "2026-07-18T00:00:00Z",
            "contract_sha256": rul9.sha256_bytes(rul9.canonical_json(registered)),
        },
        "identity": fake_identity(registered),
        "raw": evidence,
        "summary": summary,
        "verdict": rul9.evaluate_verdict(summary, registered),
    }
    payload["artifact_sha256"] = rul9.artifact_hash(payload)
    return payload


def fake_derived_receipt(tmp_path: Path, registered: dict) -> tuple[dict, dict]:
    measurement = fake_receipt(registered)
    measurement_path = tmp_path / "measurement.json"
    measurement_path.write_text(
        json.dumps(measurement, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    derivation_identity = deepcopy(measurement["identity"])
    derivation_identity["source"]["sha256"] = "patched-derivation-source"
    derived = rul9.migrate_receipt(
        registered,
        measurement_path,
        derivation_identity=derivation_identity,
    )
    return measurement, derived


def test_percentile_and_artifact_hash_are_recomputable() -> None:
    assert rul9.percentile([4, 1, 3, 2], 0.50) == 2.5
    payload = {"value": 1, "artifact_sha256": "old"}
    original = rul9.artifact_hash(payload)
    payload["artifact_sha256"] = "new"
    assert rul9.artifact_hash(payload) == original
    payload["value"] = 2
    assert rul9.artifact_hash(payload) != original


def test_checked_parity_provenance_is_pinned_without_regeneration() -> None:
    payload = rul9._checked_parity_provenance(contract())
    assert payload["summary"]["commands_per_surface"] == 132
    assert payload["summary"]["checkpoints_per_surface"] == 133
    assert payload["summary"]["first_divergence"] is None


def test_summary_derives_independent_release_and_training_cells() -> None:
    registered = contract()
    summary = rul9.derive_summary(fake_raw(registered), registered)
    assert summary["release"]["surfaces"]["live"]["command_ms"]["p95"] == 2.0
    assert summary["release"]["surfaces"]["headless"]["steps_per_second"] == 1320.0
    assert summary["training"]["steps_per_second"] == 4.0
    assert summary["training"]["traversals_per_second"] == 512.0
    assert summary["training"]["games_per_second"] == 4.0
    assert rul9.evaluate_verdict(summary, registered)["overall"] == "pass"


def test_verifier_rejects_raw_tampering_even_with_rehashed_artifact() -> None:
    registered = contract()
    payload = fake_receipt(registered)
    payload["raw"]["release"]["surfaces"]["live"][0]["protocol_commands"][0][
        "duration_ms"
    ] = 99.0
    payload["artifact_sha256"] = rul9.artifact_hash(payload)
    with pytest.raises(rul9.Rul9Error, match="does not rederive"):
        rul9.verify_receipt(registered, payload, check_current=False)


def test_verifier_rejects_missing_fallback_counter() -> None:
    registered = contract()
    payload = fake_receipt(registered)
    del payload["raw"]["release"]["surfaces"]["live"][0]["fallback_counters"][
        "candidate_cap"
    ]
    payload["artifact_sha256"] = rul9.artifact_hash(payload)
    with pytest.raises(rul9.Rul9Error, match="inventory mismatch"):
        rul9.verify_receipt(registered, payload, check_current=False)


def test_verifier_rejects_positive_fallback_and_overflow() -> None:
    registered = contract()
    for path, message in (
        (("fallback_counters", "indexed_fallbacks"), "fallback counters"),
        (("rows", 0, "token_sample", "overflow_count"), "overflow_count"),
    ):
        raw = fake_raw(registered)
        cursor = raw["training"]["games"][0]
        for component in path[:-1]:
            cursor = cursor[component]
        cursor[path[-1]] = 1
        payload = fake_receipt(registered, raw)
        with pytest.raises(rul9.Rul9Error, match=message):
            rul9.verify_receipt(registered, payload, check_current=False)


def test_verifier_rejects_budget_miss_without_changing_representation() -> None:
    registered = contract()
    registered["budgets"]["training"]["steps_per_second_min"] = 5.0
    payload = fake_receipt(registered)
    assert payload["verdict"]["training"] == {
        "status": "miss",
        "failures": ["training steps/s"],
    }
    assert payload["verdict"]["representation_decision"] == (
        "retain full_clone/current_game_v1"
    )
    with pytest.raises(rul9.Rul9Error, match="product budgets missed"):
        rul9.verify_receipt(registered, payload, check_current=False)


def test_verifier_rejects_current_source_or_binary_drift() -> None:
    registered = contract()
    payload = fake_receipt(registered)
    current = deepcopy(payload["identity"])
    current["binary"]["extension_sha256"] = "different"
    with pytest.raises(rul9.Rul9Error, match="identity is stale"):
        rul9.verify_receipt(
            registered, payload, check_current=True, current_identity=current
        )


def test_verifier_rederives_rss_and_exact_training_coordinates() -> None:
    registered = contract()
    for mutate, message in (
        (
            lambda raw: raw["release"]["rss"].__setitem__("peak_bytes", 99),
            "RSS peak",
        ),
        (
            lambda raw: raw["training"]["games"][0].__setitem__("deal_seed", 999),
            "coordinates",
        ),
    ):
        raw = fake_raw(registered)
        mutate(raw)
        with pytest.raises(rul9.Rul9Error, match=message):
            rul9.derive_summary(raw, registered)


def test_report_only_claims_expanded_frontier_when_observed() -> None:
    registered = contract()
    receipt = fake_receipt(registered)
    report = rul9.render_report(receipt, registered)
    assert "did not reproduce the exploratory expanded-token pressure" in report
    assert "above the 4,096 diagnostic frontier" not in report

    raw = fake_raw(registered)
    raw["training"]["games"][0]["rows"][0]["token_sample"][
        "expanded_semantic_tokens"
    ] = 5000
    receipt = fake_receipt(registered, raw)
    report = rul9.render_report(receipt, registered)
    assert "above the 4,096 diagnostic frontier" in report


def test_report_narrative_derives_the_live_only_miss() -> None:
    registered = contract()
    registered["budgets"]["release"]["live_command_p95_ms_max"] = 1.0
    registered["budgets"]["release"]["live_games_per_second_min"] = 20.0
    receipt = fake_receipt(registered)
    report = rul9.render_report(receipt, registered)
    assert "Release missed only the player-facing live gates" in report
    assert "WebSocket Command p95 was 2.000 ms against the 1 ms maximum" in report
    assert "live complete-game throughput was 10.000/s" in report
    assert "Training passed at 4.000 roots/s" in report
    assert "both stayed within every pre-registered budget" not in report


def test_migration_binds_distinct_sources_and_byte_identical_raw(
    tmp_path: Path,
) -> None:
    registered = contract()
    measurement, derived = fake_derived_receipt(tmp_path, registered)
    origin = derived["measurement_origin"]

    assert derived["schema_version"] == rul9.DERIVED_SCHEMA_VERSION
    assert origin["artifact_sha256"] == measurement["artifact_sha256"]
    assert origin["identity"] == measurement["identity"]
    assert derived["derivation"]["identity"]["source"]["sha256"] == (
        "patched-derivation-source"
    )
    assert derived["derivation"]["role"].endswith("did-not-produce-measurement-samples")
    assert "raw" not in derived
    origin_raw = rul9.canonical_json(measurement["raw"])
    assert origin["raw_lineage"] == {
        "algorithm": rul9.RAW_LINEAGE_ALGORITHM,
        "canonical_bytes": len(origin_raw),
        "sha256": rul9.sha256_bytes(origin_raw),
    }
    verified = rul9.verify_receipt(registered, derived, check_current=False)
    assert verified["verified"] is True


def test_migration_rejects_origin_lineage_or_derived_summary_tampering(
    tmp_path: Path,
) -> None:
    registered = contract()
    measurement, derived = fake_derived_receipt(tmp_path, registered)

    tampered = deepcopy(derived)
    tampered["measurement_origin"]["raw_lineage"]["sha256"] = "different"
    tampered["artifact_sha256"] = rul9.artifact_hash(tampered)
    with pytest.raises(rul9.Rul9Error, match="canonical raw lineage"):
        rul9.verify_receipt(registered, tampered, check_current=False)

    tampered = deepcopy(derived)
    tampered["summary"]["release"]["surfaces"]["live"]["commands"] = 1
    tampered["artifact_sha256"] = rul9.artifact_hash(tampered)
    with pytest.raises(rul9.Rul9Error, match="summary does not rederive"):
        rul9.verify_receipt(registered, tampered, check_current=False)

    measurement_path = Path(derived["measurement_origin"]["path"])
    measurement["run"]["completed_at"] = "changed"
    measurement["artifact_sha256"] = rul9.artifact_hash(measurement)
    measurement_path.write_text(
        json.dumps(measurement, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(rul9.Rul9Error, match="file SHA-256 mismatch"):
        rul9.verify_receipt(registered, derived, check_current=False)


def test_migration_rejects_current_derivation_source_drift(tmp_path: Path) -> None:
    registered = contract()
    _, derived = fake_derived_receipt(tmp_path, registered)
    current = deepcopy(derived["derivation"]["identity"])
    current["source"]["sha256"] = "newer-source"
    with pytest.raises(rul9.Rul9Error, match="derivation source.*stale"):
        rul9.verify_receipt(
            registered, derived, check_current=True, current_identity=current
        )


def test_real_semantic_census_is_ragged_and_fail_closed() -> None:
    env = rul9._fresh_engine(1197, 0)
    static = rul9._semantic_static(env)
    sample = rul9._TokenCensus(env).sample(
        env, int(env.current_agent_index()), details=True
    )
    assert static["catalog_active_tokens"] == 2088
    assert max(static["tokens_per_definition"]) == 148
    assert sample["visible_object_references"] > 0
    assert sample["overflow_count"] == 0
    assert sum(sample["tokens_by_zone"].values()) == sample["expanded_semantic_tokens"]


def test_checked_receipt_is_fail_closed_when_present() -> None:
    if not rul9.DEFAULT_OUT.exists():
        pytest.skip("canonical RUL-9 evidence is generated after preregistration")
    registered = contract()
    payload = json.loads(rul9.DEFAULT_OUT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == rul9.DERIVED_SCHEMA_VERSION
    assert payload["artifact_sha256"] == rul9.artifact_hash(payload)
    origin = json.loads(rul9.DEFAULT_MEASUREMENT_ORIGIN.read_text(encoding="utf-8"))
    assert "raw" not in payload
    assert payload["summary"] == rul9.derive_summary(origin["raw"], registered)
    assert payload["verdict"] == rul9.evaluate_verdict(payload["summary"], registered)
    assert origin["artifact_sha256"].startswith("498df1")
    assert payload["measurement_origin"]["artifact_sha256"] == origin["artifact_sha256"]
    origin_raw = rul9.canonical_json(origin["raw"])
    assert payload["measurement_origin"]["raw_lineage"]["sha256"] == (
        rul9.sha256_bytes(origin_raw)
    )

    if payload["verdict"]["overall"] == "pass":
        verified = rul9.verify_receipt(registered, payload, check_current=False)
        assert verified["verified"] is True
        assert verified["representation_decision"] == (
            "retain full_clone/current_game_v1"
        )
    else:
        with pytest.raises(rul9.Rul9Error, match="product budgets missed"):
            rul9.verify_receipt(registered, payload, check_current=False)


def test_checked_report_matches_checked_receipt() -> None:
    if not rul9.DEFAULT_OUT.exists():
        pytest.skip("canonical RUL-9 evidence is generated after preregistration")
    payload = json.loads(rul9.DEFAULT_OUT.read_text(encoding="utf-8"))
    report = rul9.DEFAULT_REPORT.read_text(encoding="utf-8")
    assert report == rul9.render_report(payload)
