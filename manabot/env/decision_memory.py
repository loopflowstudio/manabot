"""Decision-memory buffer widths, field order and the blanking ablation.

Native `agent/decision_memory.rs` writes the record, reference, detail and
member tables; canonical JSON stays expanded. Local table joins are transport,
never learning features.
"""

import numpy as np
from numpy.typing import NDArray

RECORD_DIM = 74
REFERENCE_DIM = 14
DETAIL_DIM = 6
MEMBER_DIM = 2
MEMORY_WIDTHS = {
    "decision_records": RECORD_DIM,
    "decision_references": REFERENCE_DIM,
    "decision_reference_details": DETAIL_DIM,
    "decision_members": MEMBER_DIM,
}

MEMORY_FIELDS = (
    "decision_records",
    "decision_reference_details",
    "decision_references",
    "decision_members",
)


def integer_words(value: int) -> tuple[int, int]:
    return (value & 0xFFFFFFFF) >> 16, value & 0xFFFF


def blank_memory(
    observation: dict[str, NDArray[np.float32]],
) -> dict[str, NDArray[np.float32]]:
    """Erase every history, presence and table cue before host preparation."""
    return {
        key: np.zeros_like(value) if key in MEMORY_WIDTHS else value
        for key, value in observation.items()
    }
