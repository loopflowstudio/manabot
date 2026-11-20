"""Decision-memory buffer widths, field order and the blanking ablation.

Native `agent/decision_memory.rs` writes the record, reference, detail and
member tables; canonical JSON stays expanded. Local table joins are transport,
never learning features.
"""

import numpy as np
from numpy.typing import NDArray

from manabot.env.observation import width

MEMORY_FIELDS = (
    "decision_records",
    "decision_reference_details",
    "decision_references",
    "decision_members",
)
MEMORY_WIDTHS = {key: width(key) for key in MEMORY_FIELDS}
RECORD_DIM, DETAIL_DIM, REFERENCE_DIM, MEMBER_DIM = MEMORY_WIDTHS.values()


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
