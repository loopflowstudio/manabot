"""Completeness and honesty checks for the pinned CR inventory, not engine proof."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "audit_rules", ROOT / "scripts/audit_rules.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source(body: str) -> bytes:
    return (
        "\ufeffContents\r\n1. Game Concepts\r\n704. State-Based Actions\r\n"
        "Glossary\r\n1. Game Concepts\r\n" + body + "\r\nGlossary\r\nTerm\r\n"
    ).encode()


def test_inventory_retains_parents_multi_letter_suffixes_and_source_spans() -> None:
    tool = _tool()
    rows = tool.inventory(
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
        _tool().inventory(_source("704.5a One\r\n704.5a Two"))


def test_inventory_rejects_missing_body_boundary() -> None:
    with pytest.raises(ValueError, match="contents and body"):
        _tool().inventory(b"1. Game Concepts\n100. General\nGlossary\n")


def test_committed_inventory_is_complete_unique_and_evidence_is_honest() -> None:
    tool = _tool()
    directory = tool.DIRECTORY
    metadata = json.loads((directory / "source.json").read_text())
    reviews = json.loads((directory / "reviews.json").read_text())
    rows = [
        json.loads(line)
        for line in (directory / "index.jsonl").read_text().splitlines()
    ]
    tool.validate(rows, metadata, reviews)
    assert len(rows) == 3322
    identifiers = {row["id"] for row in rows}
    assert {"1", "9", "100", "103.5", "704.5aa", "903", "905.6"} <= identifiers
    # Family evidence must never silently certify a child rule.
    row = next(row for row in rows if row["id"] == "103.5b")
    assert row["status"] == "unknown"
    assert row["tests"] == []
    assert next(row for row in rows if row["id"] == "704.5c")["status"] == "missing"
    assert next(row for row in rows if row["id"] == "903")["status"] == "not_applicable"


def test_source_digest_change_is_rejected() -> None:
    tool = _tool()
    with pytest.raises(ValueError, match="source digest"):
        tool.build(b"different bytes", {"sha256": "0" * 64}, {})
