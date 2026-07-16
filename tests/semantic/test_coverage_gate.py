from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from manabot.semantic.compiler import pretty_json
from manabot.semantic.coverage import (
    CoverageEvidenceError,
    default_paths,
    enforce_coverage_gate,
    generate_coverage_artifact,
    generate_paths,
    write_or_check,
)

ROOT, SOURCE_PATH, EVIDENCE_PATH, ARTIFACT_PATH = default_paths()


def _documents() -> tuple[dict, dict]:
    return (
        json.loads(SOURCE_PATH.read_text(encoding="utf-8")),
        json.loads(EVIDENCE_PATH.read_text(encoding="utf-8")),
    )


def _set_window_kernel_count(evidence: dict, count: int) -> None:
    window = evidence["admission_history"][-20:]
    for index, row in enumerate(window):
        if index < 20 - count:
            row["classification"] = "content_only"
            row.pop("kernel_surfaces")
        else:
            row["classification"] = "kernel_changing"


def test_checked_artifact_is_current_complete_and_deterministic():
    first = generate_paths(SOURCE_PATH, EVIDENCE_PATH, root=ROOT)
    second = generate_paths(SOURCE_PATH, EVIDENCE_PATH, root=ROOT)

    assert pretty_json(first) == pretty_json(second)
    assert ARTIFACT_PATH.read_text(encoding="utf-8") == pretty_json(first)
    assert first["coverage_status"] == "complete"
    assert first["summary"] == {
        "definition_count": 31,
        "deck_definition_count": 29,
        "referenced_definition_count": 2,
        "rule_family_count": 15,
        "sanctioned_deviation_count": 7,
        "test_count": 31,
        "unsupported_count": 0,
    }
    assert first["coverage_gaps"] == {
        "sanctioned_deviations": [
            "stx.divide_by_zero",
            "stx.igneous_inspiration",
            "stx.pop_quiz",
            "tla.accumulate_wisdom",
            "tla.dragonfly_swarm",
            "tla.fire_nation_cadets",
            "tla.suki_kyoshi_warrior",
        ],
        "unmapped_semantic_atoms": [],
        "unsupported_cards": [],
        "untested_cards": [],
    }
    enforce_coverage_gate(first)


def test_badgermole_crosses_source_families_tests_and_history():
    artifact = generate_paths(SOURCE_PATH, EVIDENCE_PATH, root=ROOT)
    card = next(
        card
        for card in artifact["cards"]
        if card["semantic_key"] == "tla.badgermole_cub"
    )
    history = next(
        row
        for row in artifact["admission_history"]
        if row["semantic_key"] == "tla.badgermole_cub"
    )

    assert card["support"] == "supported"
    assert card["rule_families"] == [
        "earthbend",
        "mana_and_payment",
        "targets_and_filters",
        "triggered_abilities",
    ]
    assert "opcode:earthbend" in card["semantic_atoms"]
    assert card["tests"] == [
        "badgermole_waterbend",
        "earthbend_animates",
        "earthbend_returns",
    ]
    assert history["classification"] == "kernel_changing"
    assert history["sequence"] == 20


def test_initial_window_has_explicit_acknowledged_historical_breach():
    artifact = generate_paths(SOURCE_PATH, EVIDENCE_PATH, root=ROOT)
    gate = artifact["rolling_kernel_change_gate"]

    assert gate["counts"] == {
        "content_only": 0,
        "denominator": 20,
        "kernel_changing": 20,
    }
    assert gate["ratios"] == {
        "content_only_share": {"denominator": 20, "numerator": 0},
        "content_only_to_kernel_changing": "0:20",
        "kernel_changing_share": {"denominator": 20, "numerator": 20},
    }
    assert gate["threshold_breached"] is True
    assert gate["gate_status"] == "breached_acknowledged"
    assert gate["latest_sequence"] == 25
    assert gate["response"]["kind"] == "ir_redesign"
    assert gate["response"]["acknowledged_through_sequence"] == 25


@pytest.mark.parametrize(
    ("kernel_count", "expected_breach", "expected_status"),
    [
        (4, False, "within_threshold"),
        (5, True, "breached_acknowledged"),
    ],
)
def test_threshold_is_strictly_more_than_one_in_five(
    kernel_count: int, expected_breach: bool, expected_status: str
):
    source, evidence = _documents()
    _set_window_kernel_count(evidence, kernel_count)

    artifact = generate_coverage_artifact(source, evidence, root=ROOT)

    assert artifact["rolling_kernel_change_gate"]["counts"]["denominator"] == 20
    assert (
        artifact["rolling_kernel_change_gate"]["counts"]["kernel_changing"]
        == kernel_count
    )
    assert (
        artifact["rolling_kernel_change_gate"]["threshold_breached"] is expected_breach
    )
    assert artifact["rolling_kernel_change_gate"]["gate_status"] == expected_status
    enforce_coverage_gate(artifact)


@pytest.mark.parametrize("response_mode", ["missing", "stale"])
def test_breach_requires_a_response_for_the_newest_sequence(response_mode: str):
    source, evidence = _documents()
    if response_mode == "missing":
        evidence["policy"]["response"] = None
    else:
        evidence["policy"]["response"]["acknowledged_through_sequence"] = 24

    artifact = generate_coverage_artifact(source, evidence, root=ROOT)

    assert (
        artifact["rolling_kernel_change_gate"]["gate_status"]
        == "breached_unacknowledged"
    )
    with pytest.raises(CoverageEvidenceError, match="coverage gate failed"):
        enforce_coverage_gate(artifact)


def test_history_excludes_basics_and_tokens_and_rejects_missing_cards():
    source, evidence = _documents()
    history_keys = {row["semantic_key"] for row in evidence["admission_history"]}

    assert len(history_keys) == 25
    assert not history_keys & {
        "basic.forest",
        "basic.island",
        "basic.mountain",
        "basic.plains",
        "token.ally",
        "token.clue",
    }

    evidence["admission_history"].pop()
    with pytest.raises(CoverageEvidenceError, match="missing eligible cards"):
        generate_coverage_artifact(source, evidence, root=ROOT)


def test_content_only_history_cannot_hide_kernel_evidence():
    source, evidence = _documents()
    row = evidence["admission_history"][-1]
    row["classification"] = "content_only"

    with pytest.raises(CoverageEvidenceError, match="cannot name kernel_surfaces"):
        generate_coverage_artifact(source, evidence, root=ROOT)


def test_card_and_family_test_references_are_required_and_exact():
    source, evidence = _documents()
    evidence["cards"][0]["tests"] = []
    with pytest.raises(CoverageEvidenceError, match="must not be empty"):
        generate_coverage_artifact(source, evidence, root=ROOT)

    _, evidence = _documents()
    evidence["tests"][0]["identifier"] += "_renamed"
    with pytest.raises(CoverageEvidenceError, match="test is not declared"):
        generate_coverage_artifact(source, evidence, root=ROOT)


def test_unmapped_typed_semantics_fail_instead_of_disappearing():
    source, evidence = _documents()
    family = next(
        family
        for family in evidence["rule_families"]
        if family["key"] == "tap_and_untap"
    )
    family["semantic_atoms"].remove("opcode:tap")

    with pytest.raises(
        CoverageEvidenceError, match="unmapped semantic atoms: opcode:tap"
    ):
        generate_coverage_artifact(source, evidence, root=ROOT)


def test_check_mode_rejects_stale_output(tmp_path: Path):
    artifact = generate_paths(SOURCE_PATH, EVIDENCE_PATH, root=ROOT)
    output = tmp_path / "coverage-gaps.json"
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(CoverageEvidenceError, match="is stale"):
        write_or_check(output, artifact, check=True)

    write_or_check(output, artifact, check=False)
    write_or_check(output, artifact, check=True)


def test_source_and_evidence_inputs_are_not_mutated():
    source, evidence = _documents()
    expected_source = deepcopy(source)
    expected_evidence = deepcopy(evidence)

    generate_coverage_artifact(source, evidence, root=ROOT)

    assert source == expected_source
    assert evidence == expected_evidence
