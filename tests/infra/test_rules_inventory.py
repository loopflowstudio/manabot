"""Completeness and honesty checks for the pinned CR inventory, not engine proof."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts import audit_rules


def _source(body: str) -> bytes:
    return (
        "\ufeffContents\r\n1. Game Concepts\r\n704. State-Based Actions\r\n"
        "Glossary\r\n1. Game Concepts\r\n" + body + "\r\nGlossary\r\nTerm\r\n"
    ).encode()


def test_inventory_retains_parents_multi_letter_suffixes_and_source_spans() -> None:
    rows = audit_rules.inventory(
        _source(
            "704. State-Based Actions\r\n704.5. List\r\n"
            "704.5k One\r\nExample: unnumbered\r\n"
            "704.5m Two\r\n704.5z Three\r\n704.5aa Four"
        )
    )
    assert [row.id for row in rows] == [
        "1",
        "704",
        "704.5",
        "704.5k",
        "704.5m",
        "704.5z",
        "704.5aa",
    ]
    assert rows[3].source_start_line == 8
    assert rows[3].source_end_line == 9
    assert rows[-1].source_start_line == 12
    assert len({row.text_sha256 for row in rows}) == len(rows)


def test_inventory_rejects_duplicate_body_identifiers() -> None:
    with pytest.raises(ValueError, match="duplicate body rule"):
        audit_rules.inventory(_source("704.5a One\r\n704.5a Two"))


def test_inventory_rejects_missing_body_boundary() -> None:
    with pytest.raises(ValueError, match="contents and body"):
        audit_rules.inventory(b"1. Game Concepts\n100. General\nGlossary\n")


def test_committed_inventory_is_complete_unique_and_evidence_is_honest() -> None:
    directory = audit_rules.DIRECTORY
    metadata = json.loads((directory / "source.json").read_text())
    reviews = json.loads((directory / "reviews.json").read_text())
    rows = [
        json.loads(line)
        for line in (directory / "index.jsonl").read_text().splitlines()
    ]
    audit_rules.validate(rows, metadata, reviews)
    assert len(rows) == 3322
    by_id = {row["id"]: row for row in rows}
    assert {"1", "9", "100", "103.5", "704.5aa", "903", "905.6"} <= by_id.keys()
    # Family evidence must never silently certify a child rule.
    row = by_id["103.5b"]
    assert row["status"] == "unknown"
    assert row["tests"] == []
    assert by_id["704.5c"]["status"] == "missing"
    assert by_id["903"]["status"] == "not_applicable"


def test_source_digest_change_is_rejected() -> None:
    with pytest.raises(ValueError, match="source digest"):
        audit_rules.build(b"different bytes", {"sha256": "0" * 64}, {})


def test_write_rejects_stale_evidence_before_replacing_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _source("704. State-Based Actions")
    source = tmp_path / "rules.txt"
    source.write_bytes(raw)
    metadata: audit_rules.Metadata = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "numbered_entries": 2,
        "ordered_ids_sha256": hashlib.sha256(b"1\n704").hexdigest(),
    }
    reviews: audit_rules.Reviews = {
        "profiles": {
            "unresolved": {"scope": "unreviewed", "anchors": ["rules.txt::absent"]}
        },
        "families": {},
        "rules": {},
    }
    (tmp_path / "source.json").write_text(json.dumps(metadata))
    (tmp_path / "reviews.json").write_text(json.dumps(reviews))
    target = tmp_path / "index.jsonl"
    target.write_text("retained inventory\n")
    monkeypatch.setattr(audit_rules, "ROOT", tmp_path)
    monkeypatch.setattr(audit_rules, "DIRECTORY", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["audit_rules", "--source", str(source), "--write"]
    )

    with pytest.raises(ValueError, match="stale evidence anchor"):
        audit_rules.main()

    assert target.read_text() == "retained inventory\n"
