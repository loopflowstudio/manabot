"""Decision-memory table widths and field order.

Native `agent/decision_memory.rs` writes the record, reference, detail and
member tables; canonical JSON stays expanded. Local table joins are transport,
never learning features.
"""

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
