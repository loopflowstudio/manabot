"""Writer catalogs preserve the rows a particular checkpoint learned."""

import hashlib
import json
from pathlib import Path

from manabot.model.writer_catalog import writer_catalog
import managym


def test_export_follows_policy_definition_and_naming_rows() -> None:
    catalog = writer_catalog()
    tokens = managym.catalog_tokens()
    assert catalog["world_digest"] == tokens["content_digest"]
    assert catalog["definition_keys"] == [
        key or name
        for key, name in zip(tokens["keys"], tokens["registry_names"], strict=True)
    ]
    assert catalog["naming_catalog"] == [
        {"name": candidate.name, "definition": candidate.definition}
        for candidate in managym.naming_catalog()
    ]


def test_frontier_writer_catalog_matches_u1952_checkpoint() -> None:
    # Taken from the original file, sha256 c95c61aa…bbaf8, at ab2b757d9.
    digest = "0798b2c7c7f54b7329043b13cb1742d74fe21b35373820580b76f979e6b7eb5f"
    directory = Path(__file__).resolve().parents[2] / "manabot/model/writer_catalogs"
    catalog = json.loads((directory / f"{digest}.json").read_bytes())
    assert catalog["world_digest"] == digest
    keys = catalog["definition_keys"]
    assert len(keys) == len(set(keys)) == 360
    assert keys[2:4] == ["tla.aang_swift_savior", "Aang and La, Ocean's Fury"]
    assert keys[277] == "token.spirit"
    names = json.dumps(catalog["naming_catalog"], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(names.encode()).hexdigest() == (
        "f3510fb8e53c7128b629c1ba6ad280ad901d4d5774521c608bafb59bbef7be5d"
    )
