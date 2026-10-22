"""Inventory a pinned Wizards TXT without treating citations as conformance.

The index records every numbered body heading and subrule, source spans and
content digests. Authored reviews are separate evidence, never inferred from
matching identifiers. No rules text or network access is needed to check the
committed index; --source additionally proves completeness against raw bytes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import NotRequired, TypedDict, cast

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/rules/audit-2026-10"
NUMBER = re.compile(r"^([1-9]|[1-9]\d{2}(?:\.\d+[a-z]*)?)\.?\s+\S")


class Metadata(TypedDict):
    sha256: str
    numbered_entries: int
    ordered_ids_sha256: str


class Profile(TypedDict):
    scope: str
    anchors: list[str]


class Review(TypedDict, total=False):
    status: str
    scope: str
    authority: str
    dispersion: str
    comments: str
    tests: list[str]
    gap: str
    anchors: list[str]


class Reviews(TypedDict):
    profiles: dict[str, Profile]
    families: dict[str, str]
    rules: dict[str, Review]


class Row(TypedDict):
    id: str
    source_start_line: int
    source_end_line: int
    text_sha256: str
    anchors: NotRequired[list[str]]
    review: str
    status: str
    scope: str
    authority: str
    dispersion: str
    comments: str
    tests: list[str]
    gap: str


@dataclass(frozen=True)
class Rule:
    id: str
    source_start_line: int
    source_end_line: int
    text_sha256: str


def inventory(raw: bytes) -> list[Rule]:
    """Skip the TOC; retain numbered sections, parents and lettered children.

    Spans are one-based inclusive raw text lines (UTF-8, optional BOM). Rule
    hashes normalize line endings to LF, unlike the source's raw-byte hash.
    Unexpected duplicate body identifiers fail instead of silently deduping.
    """
    lines = raw.decode("utf-8-sig").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "1. Game Concepts"]
    if len(starts) != 2:
        raise ValueError("expected contents and body Game Concepts headings")
    start = starts[1]
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "Glossary")
    numbered = [
        (i, match[1]) for i in range(start, end) if (match := NUMBER.match(lines[i]))
    ]
    seen: set[str] = set()
    rules: list[Rule] = []
    for position, (line, identifier) in enumerate(numbered):
        if identifier in seen:
            raise ValueError(f"duplicate body rule: {identifier}")
        seen.add(identifier)
        stop = numbered[position + 1][0] if position + 1 < len(numbered) else end
        text = "\n".join(lines[line:stop]).rstrip() + "\n"
        rules.append(
            Rule(identifier, line + 1, stop, hashlib.sha256(text.encode()).hexdigest())
        )
    return rules


def build(raw: bytes, metadata: Metadata, reviews: Reviews) -> list[Row]:
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError("source digest differs from the pinned Wizards bytes")
    profiles = reviews["profiles"]
    overrides = reviews["rules"]
    if not isinstance(profiles, dict) or not isinstance(overrides, dict):
        raise ValueError("invalid authored reviews")
    rows: list[Row] = []
    for rule in inventory(raw):
        family = rule.id.split(".")[0]
        profile_id = reviews["families"].get(family, "unresolved")
        profile = profiles[profile_id]
        row: Row = {
            "id": rule.id,
            "source_start_line": rule.source_start_line,
            "source_end_line": rule.source_end_line,
            "text_sha256": rule.text_sha256,
            "review": profile_id,
            "status": "unknown",
            "scope": profile["scope"],
            "authority": "unknown; see inspected candidate owners in review",
            "dispersion": "unknown",
            "comments": "unknown for this individual entry",
            "tests": [],
            "gap": f"Verify every obligation in CR {rule.id}; family evidence is not subrule proof.",
        }
        if profile_id == "outside_formats":
            row.update(
                status="not_applicable",
                authority="managym/src/flow/setup.rs:Game::new_with_content",
                dispersion="single admission boundary",
                comments="exactly-two-player assertion; no format configuration",
                gap="Unsupported format; do not treat as an admitted-play defect.",
            )
        override = overrides.get(rule.id, {})
        row.update(override)
        rows.append(row)
    extra = set(overrides) - {row["id"] for row in rows}
    if extra:
        raise ValueError(f"review refers to absent rule identifiers: {sorted(extra)}")
    return rows


def validate(rows: list[Row], metadata: Metadata, reviews: Reviews) -> None:
    identifiers = [row["id"] for row in rows]
    if len(set(identifiers)) != len(rows) or len(rows) != metadata["numbered_entries"]:
        raise ValueError("inventory count or uniqueness mismatch")
    if (
        hashlib.sha256("\n".join(identifiers).encode()).hexdigest()
        != metadata["ordered_ids_sha256"]
    ):
        raise ValueError("ordered identifier inventory mismatch")
    statuses = {"implemented", "partial", "missing", "not_applicable", "unknown"}
    for row in rows:
        if row["status"] not in statuses or row["review"] not in reviews["profiles"]:
            raise ValueError(f"invalid row {row['id']}")
        for key in ("scope", "authority", "dispersion", "comments", "gap"):
            if not row.get(key):
                raise ValueError(f"missing {key} for {row['id']}")
        if row["status"] == "implemented" and not row["tests"]:
            raise ValueError(f"implemented without behavioral evidence: {row['id']}")
    # Reviewed anchors must continue to exist, including specific function names.
    for review in [*reviews["profiles"].values(), *reviews["rules"].values()]:
        for ref in review.get("anchors", []):
            path, needle = ref.split("::", 1)
            if needle not in (ROOT / path).read_text():
                raise ValueError(f"stale evidence anchor: {ref}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="exact downloaded Wizards TXT")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    # JSON is untyped at this local artifact boundary; validate structure and
    # evidence below, then compare all generated rows when source is supplied.
    metadata = cast(Metadata, json.loads((DIRECTORY / "source.json").read_text()))
    reviews = cast(Reviews, json.loads((DIRECTORY / "reviews.json").read_text()))
    target = DIRECTORY / "index.jsonl"
    if args.source:
        rows = build(args.source.read_bytes(), metadata, reviews)
        output = "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
            for row in rows
        )
        if args.write:
            target.write_text(output)
        elif target.read_text() != output:
            raise ValueError("index differs from pinned source and authored reviews")
    elif args.write:
        parser.error("--write requires --source")
    else:
        rows = [cast(Row, json.loads(line)) for line in target.read_text().splitlines()]
    validate(rows, metadata, reviews)
    print(
        json.dumps(
            {
                "numbered_entries": len(rows),
                "statuses": dict(Counter(row["status"] for row in rows)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
