"""Shared effect buffer shapes: headers with explicit CSR memberships.

Native `agent/shared_effects.rs` writes these arrays from exact provenance.
Header offsets address the bounded detail and member arrays; padding does not
count as a record, and a record that does not fit is a capacity failure,
never truncation.
"""

import numpy as np

import managym

_LAYOUT = managym.observation_layout()

# Fixed extents, from the native layout.
SHARED_EFFECT_SHAPES: dict[str, tuple[int, int]] = {
    key: (array["shape"][0], array["shape"][1])
    for key, array in _LAYOUT.items()
    if key in ("effects", "effect_details", "effect_members")
}
EFFECT_ROWS, EFFECT_DIM = SHARED_EFFECT_SHAPES["effects"]
DETAIL_ROWS, DETAIL_DIM = SHARED_EFFECT_SHAPES["effect_details"]
MEMBER_ROWS, MEMBER_DIM = SHARED_EFFECT_SHAPES["effect_members"]

_detail = _LAYOUT["effect_details"]
DETAIL_KIND_NAMES = _detail["row_kind_names"]
DETAIL_KINDS = tuple(DETAIL_KIND_NAMES.values())
PROGRAM_TOKEN_KIND = np.zeros(max(DETAIL_KINDS) + 1, dtype=np.int64)
for detail_kind, token_kind in _detail["program_token_kinds"].items():
    PROGRAM_TOKEN_KIND[detail_kind] = token_kind
NUMBER_TOKEN_KINDS = tuple(
    int(PROGRAM_TOKEN_KIND[DETAIL_KIND_NAMES[name]])
    for name in ("program_integer", "program_boolean")
)
