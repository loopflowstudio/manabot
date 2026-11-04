"""V7 fixed buffers for complete effect payloads and explicit CSR memberships.

Native authority supplies exact provenance. This reference encoder never infers
it from old per-object fragments. Offsets address bounded token/member arrays;
padding does not count as a record and no capacity failure is truncated.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

import managym

EFFECT_ROWS, EFFECT_DIM = 128, 18
DETAIL_ROWS, DETAIL_DIM = 512, 3
MEMBER_ROWS, MEMBER_DIM = 2048, 2
SHARED_EFFECT_SHAPES = {
    "effects": (EFFECT_ROWS, EFFECT_DIM),
    "effect_details": (DETAIL_ROWS, DETAIL_DIM),
    "effect_members": (MEMBER_ROWS, MEMBER_DIM),
}


def _token(name: str) -> tuple[float, float]:
    value = managym.learning_token_value(name) & 0xFFFFFFFF
    return float(value >> 16), float(value & 0xFFFF)


def encode_shared_effects(
    observation: managym.Observation,
    objects: Mapping[int, int],
    ranks: Mapping[int, float],
) -> dict[str, NDArray[np.float32]]:
    records = observation.shared_effects
    if records is None:
        raise ValueError("v7 requires native shared-effect provenance")
    counts = {
        "effects": len(records),
        "effect_details": sum(
            len(r.effect.keywords)
            + len(r.effect.subtypes)
            + (r.effect.color is not None)
            for r in records
        ),
        "effect_members": sum(len(r.members) for r in records),
    }
    for name, count in counts.items():
        capacity = SHARED_EFFECT_SHAPES[name][0]
        if count > capacity:
            raise ValueError(
                f"observation capacity exceeded: {name} {count} > {capacity}"
            )

    def join(ident: int | None) -> float:
        if ident is None or ident not in objects:
            raise ValueError("shared effect has an unavailable object join")
        return float(objects[ident] + 1)

    out = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in SHARED_EFFECT_SHAPES.items()
    }
    d = m = 0
    orders: set[int] = set()
    for index, record in enumerate(records):
        effect = record.effect
        row = out["effects"][index]
        row[:2] = _token(f"Modification::{effect.modification}")
        for i, value in enumerate((effect.power, effect.toughness, effect.amount)):
            row[2 + i] = (value or 0) / 10
            row[5 + i] = value is not None
        if effect.duration not in (
            "EndOfTurn",
            "Indefinite",
            "WhileYouControlSource",
        ) or (effect.duration == "WhileYouControlSource" and record.source is None):
            raise ValueError("invalid shared effect duration/source")
        row[8] = {"EndOfTurn": 1, "Indefinite": 2, "WhileYouControlSource": 3}[
            effect.duration
        ]
        row[9] = ranks[effect.timestamp]
        row[10] = join(record.controller_id)
        row[11] = (
            0 if effect.source_registry_key is None else effect.source_registry_key + 1
        )
        row[12] = 0 if record.source is None else join(record.source.object_id)
        row[13] = d
        tokens = [
            *((1, name) for name in effect.keywords),
            *((2, name) for name in effect.subtypes),
        ]
        if effect.color is not None:
            tokens.append((3, f"Color::{effect.color}"))
        for kind, name in tokens:
            out["effect_details"][d] = kind, *_token(name)
            d += 1
        row[14] = d - row[13]
        row[15] = m
        if not record.members:
            raise ValueError("shared effect has no memberships")
        for member in record.members:
            if not 0 <= member.order < MEMBER_ROWS or member.order in orders:
                raise ValueError("invalid shared effect membership order")
            orders.add(member.order)
            out["effect_members"][m] = join(member.target.object_id), member.order + 1
            m += 1
        row[16] = m - row[15]
        row[17] = 1
    return out
