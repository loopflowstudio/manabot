"""Build current pack projections from one reviewed definition catalog.

Definitions own rules and text; setup manifests only select deck entries. Frozen
v1/w11 sources are historical evidence, never inputs to current generation.
JSON is validated by the semantic compiler before leaving this offline boundary.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from manabot.semantic.compiler import (
    SemanticCompileError,
    canonical_json,
    compile_source,
    write_or_check,
)

CONTENT_ROOT = Path(__file__).resolve().parents[2] / "content/semantic/current"


# These mappings exist only at the authored JSON/compiler boundary. Executable
# consumers receive checked IR, never the source dictionaries.
@dataclass(frozen=True)
class ContentSource:
    definitions: dict[str, dict[str, Any]]
    setups: dict[str, dict[str, Any]]
    native_roots: set[str]
    token_templates: set[str]

    @classmethod
    def load(cls, root: Path = CONTENT_ROOT) -> ContentSource:
        rows = json.loads((root / "definitions.json").read_text())
        definitions = {row["key"]: row for row in rows}
        if len(definitions) != len(rows):
            raise SemanticCompileError("duplicate content definition key")
        if len({row["registry_name"] for row in rows}) != len(rows):
            raise SemanticCompileError("duplicate content card name")
        return cls(
            definitions,
            json.loads((root / "setups.json").read_text()),
            set(json.loads((root / "native_roots.json").read_text())),
            set(json.loads((root / "token_templates.json").read_text())),
        )

    def closure(self, roots: set[str]) -> list[dict[str, Any]]:
        """Resolve all token/definition obligations without a name registry."""
        pending = list(roots)
        resolved: set[str] = set()
        while pending:
            key = pending.pop()
            if key in resolved:
                continue
            if key not in self.definitions:
                raise SemanticCompileError(f"missing content definition {key!r}")
            resolved.add(key)
            pending.extend(_references(self.definitions[key]))
        return [self.definitions[key] for key in sorted(resolved)]

    def project(self, preset: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if preset in {"pool", "tla"}:
            if preset == "tla":
                roots = self.native_roots
            else:
                roots = {
                    key
                    for key, row in self.definitions.items()
                    if not row["characteristics"].get("token")
                    and not key.startswith("research.")
                }
            source = {
                "schema_version": 1,
                "pack_key": f"tla-{preset}-v1",
                "decks": [],
                "pool": sorted(roots),
                "token_templates": sorted(self.token_templates),
            }
            roots = roots | self.token_templates
        else:
            setup = self.setups[preset]
            roots = {
                entry["definition"]
                for deck in setup["decks"]
                for field in ("cards", "sideboard")
                for entry in deck.get(field, [])
            }
            source = {"schema_version": 1, **setup}
        source["definitions"] = self.closure(roots)
        return compile_source(source)


def _references(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "definition_ref":
                if not isinstance(child, str):
                    raise SemanticCompileError("definition reference must be a string")
                yield child
            else:
                yield from _references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _references(child)


def generate(*, check: bool = False, root: Path = CONTENT_ROOT) -> None:
    catalog = ContentSource.load(root)
    inventory = json.loads((root / "inventory.json").read_text())
    qualifications = json.loads((root / "qualifications.json").read_text())
    multi_face_cards = {
        row["registry_name"] for row in inventory if len(row["faces"]) != 1
    }
    for key, qualification in qualifications.items():
        digest = hashlib.sha256(
            canonical_json(catalog.closure({key})).encode()
        ).hexdigest()
        definition = catalog.definitions[key]
        if digest != qualification["closure_hash"]:
            raise SemanticCompileError(
                f"stale qualification for {key}: review changed clauses"
            )
        if definition["characteristics"].get("token", False):
            raise SemanticCompileError(f"token {key} cannot qualify for deck entry")
        if definition["registry_name"] in multi_face_cards:
            raise SemanticCompileError(
                f"qualification for {key} requires unsupported face lifecycle"
            )
        if not qualification["witnesses"]:
            raise SemanticCompileError(
                f"qualification for {key} has no independent witnesses"
            )
    # IR already owns names, keys and presence in each projection. Retain only
    # qualification decisions here, bound to the exact programs they reviewed.
    packs: dict[str, dict[str, Any]] = {}
    for preset in [*catalog.setups, "pool", "tla"]:
        ir, coverage = catalog.project(preset)
        write_or_check(root / "generated" / f"{preset}.ir.json", ir, check=check)
        write_or_check(
            root / "generated" / f"{preset}.coverage.json", coverage, check=check
        )
        packs[ir["pack_key"]] = {
            "ir_hash": ir["ir_hash"],
            "eligible": sorted(
                d["semantic_key"]
                for d in ir["definitions"]
                if d["semantic_key"] in qualifications
            ),
        }
    eligibility = {
        "inventory": sorted({row["registry_name"] for row in inventory}),
        "packs": packs,
    }
    write_or_check(root / "generated" / "eligibility.json", eligibility, check=check)
