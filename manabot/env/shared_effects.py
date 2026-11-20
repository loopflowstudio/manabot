"""Shared effect buffer shapes: headers with explicit CSR memberships.

Native `agent/shared_effects.rs` writes these arrays from exact provenance.
Header offsets address the bounded detail and member arrays; padding does not
count as a record, and a record that does not fit is a capacity failure,
never truncation.
"""

EFFECT_ROWS, EFFECT_DIM = 128, 18
DETAIL_ROWS, DETAIL_DIM = 4096, 3
MEMBER_ROWS, MEMBER_DIM = 2048, 2
SHARED_EFFECT_SHAPES: dict[str, tuple[int, int]] = {
    "effects": (EFFECT_ROWS, EFFECT_DIM),
    "effect_details": (DETAIL_ROWS, DETAIL_DIM),
    "effect_members": (MEMBER_ROWS, MEMBER_DIM),
}
