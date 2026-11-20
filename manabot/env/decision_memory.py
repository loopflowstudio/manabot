"""Decision-memory buffer widths, field order and the blanking ablation.

Native `agent/decision_memory.rs` writes the record, reference, detail and
member tables; canonical JSON stays expanded. Local table joins are transport,
never learning features.
"""

import numpy as np
from numpy.typing import NDArray

import managym

# Table widths, from the native layout.
MEMORY_WIDTHS: dict[str, int] = {
    key: array["shape"][-1]
    for key, array in managym.observation_layout(decision_memory=True).items()
    if key.startswith("decision_")
}
RECORD_DIM = MEMORY_WIDTHS["decision_records"]
REFERENCE_DIM = MEMORY_WIDTHS["decision_references"]
DETAIL_DIM = MEMORY_WIDTHS["decision_reference_details"]
MEMBER_DIM = MEMORY_WIDTHS["decision_members"]

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
