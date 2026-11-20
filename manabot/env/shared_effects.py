"""Shared effect buffer shapes: headers with explicit CSR memberships.

Native `agent/shared_effects.rs` writes these arrays from exact provenance.
Header offsets address the bounded detail and member arrays; padding does not
count as a record, and a record that does not fit is a capacity failure,
never truncation.
"""

import managym

# Fixed extents, from the native layout.
SHARED_EFFECT_SHAPES: dict[str, tuple[int, int]] = {
    key: (array["shape"][0], array["shape"][1])
    for key, array in managym.observation_layout().items()
    if key in ("effects", "effect_details", "effect_members")
}
EFFECT_ROWS, EFFECT_DIM = SHARED_EFFECT_SHAPES["effects"]
DETAIL_ROWS, DETAIL_DIM = SHARED_EFFECT_SHAPES["effect_details"]
MEMBER_ROWS, MEMBER_DIM = SHARED_EFFECT_SHAPES["effect_members"]
