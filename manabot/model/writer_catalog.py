"""Export the engine's row identities for checkpoints that predate named rows.

Run this file with the writer's engine on the import path. It uses only the
native public pool and naming APIs available at the writing revision.
"""

import json
from typing import TypedDict

import managym


class NamingEntry(TypedDict):
    name: str
    definition: int | None


class WriterCatalog(TypedDict):
    world_digest: str
    definition_keys: list[str]
    naming_catalog: list[NamingEntry]


def naming_entries() -> list[NamingEntry]:
    """Preserve Command order and the optional joins in the checkpoint digest."""
    return [
        NamingEntry(name=candidate.name, definition=candidate.definition)
        for candidate in managym.naming_catalog()
    ]


def writer_catalog() -> WriterCatalog:
    pool = managym.content_pool("world")
    definitions = sorted(pool["definitions"], key=lambda row: row["card_def_id"])
    keys: list[str] = []
    for index, row in enumerate(definitions):
        if row["card_def_id"] != index:
            raise ValueError(f"missing definition row {index} in the writer catalog")
        # Back faces and rules fixtures have only their registry name.
        keys.append(row["key"] or row["registry_name"])
    if len(set(keys)) != len(keys):
        raise ValueError("writer catalog contains duplicate definition identities")
    return WriterCatalog(
        world_digest=pool["content_digest"],
        definition_keys=keys,
        naming_catalog=naming_entries(),
    )


if __name__ == "__main__":
    print(json.dumps(writer_catalog(), indent=2))
