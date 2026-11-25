"""Shared effect buffer shapes: headers with explicit CSR memberships.

Native `agent/shared_effects.rs` writes these arrays from exact provenance.
Header offsets address the bounded detail and member arrays; padding does not
count as a record, and a record that does not fit is a capacity failure,
never truncation.
"""

from typing import cast

import numpy as np

from manabot.env.observation import V11_LAYOUT, column, columns

# These three tables have two fixed extents in every supported input version.
SHARED_EFFECT_SHAPES: dict[str, tuple[int, int]] = {
    key: cast(tuple[int, int], tuple(V11_LAYOUT[key]["shape"]))
    for key in ("effects", "effect_details", "effect_members")
}
EFFECT_ROWS, EFFECT_DIM = SHARED_EFFECT_SHAPES["effects"]
DETAIL_ROWS, DETAIL_DIM = SHARED_EFFECT_SHAPES["effect_details"]
MEMBER_ROWS, MEMBER_DIM = SHARED_EFFECT_SHAPES["effect_members"]

EFFECT_MODIFICATION = columns("effects", "modification_high", "modification_low")
EFFECT_LINKS = columns("effects", "controller", "source")
EFFECT_SOURCE_DEFINITION = column("effects", "source_definition")
EFFECT_VALID = column("effects", "valid")
EFFECT_SPANS = columns(
    "effects", "detail_offset", "detail_count", "member_offset", "member_count"
)
DETAIL_KIND = column("effect_details", "kind")
DETAIL_VALUE = columns("effect_details", "value_high", "value_low")
MEMBER_OBJECT = column("effect_members", "object")

_detail = V11_LAYOUT["effect_details"]
DETAIL_KIND_NAMES = _detail["row_kind_names"]
DETAIL_KINDS = tuple(DETAIL_KIND_NAMES.values())
PROGRAM_TOKEN_KIND = np.zeros(max(DETAIL_KINDS) + 1, dtype=np.int64)
for detail_kind, token_kind in _detail["program_token_kinds"].items():
    PROGRAM_TOKEN_KIND[detail_kind] = token_kind
NUMBER_TOKEN_KINDS = tuple(
    int(PROGRAM_TOKEN_KIND[DETAIL_KIND_NAMES[name]])
    for name in ("program_integer", "program_boolean")
)


def effect_span_rows(
    offsets: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Owning header and batch-local row index for each detail or member, in order."""
    owners = np.repeat(np.arange(len(counts)), counts)
    # Source offsets restart in each batch. Packed offsets run across the group.
    packed_offsets = np.cumsum(counts) - counts
    within_header = np.arange(len(owners)) - packed_offsets[owners]
    return owners, offsets[owners] + within_header
