"""Archived rows earn the same structural guarantees as live encoder output."""

import hashlib
import json
import struct

import numpy as np
import pytest

from manabot.env.match import Match
from manabot.verify.representation import shared_effect_root
import managym


@pytest.fixture(scope="module")
def batch() -> managym.ObservationBatch:
    env = managym.Env(seed=111, decision_memory=True, decision_memory_include_own=True)
    env.reset(Match().to_rust())
    opening = env.observation_for_player(env.current_agent_index())
    for _ in range(9):
        env.step(0)
    layout = managym.ObservationLayout(
        max_actions=128,
        policy_history=True,
        max_decision_records=64,
        max_decision_references=128,
        max_decision_reference_details=2048,
        max_decision_members=512,
    )
    return managym.encode_observation_batch(
        [opening, env, shared_effect_root(1)], layout
    )


def test_archive_roundtrip_preserves_live_rows_and_layout(
    batch: managym.ObservationBatch,
) -> None:
    for source in (
        batch,
        batch.blank_decision_memory(),
        batch.gather([(batch, 2), (batch, 0), (batch, 2)]),
        batch.gather([]),
    ):
        loaded = managym.ObservationBatch.from_bytes(source.to_bytes())
        assert loaded.layout.shapes == source.layout.shapes
        assert loaded.layout.csr_fields == source.layout.csr_fields
        assert loaded.tables() == source.tables()
        assert loaded.action_counts == source.action_counts
        assert loaded.batch_size == source.batch_size
        # A restored table can participate in a mixed gather with a live table.
        coords = [
            (owner, i) for i in range(source.batch_size) for owner in (loaded, source)
        ]
        assert source.gather(coords).action_counts == [
            c for c in source.action_counts for _ in range(2)
        ]


def seal(header: bytes, body: bytes) -> bytes:
    content = header + b"\n" + body
    return content + hashlib.sha256(content).digest()


def test_archive_rejects_truncation_or_changed_bytes(
    batch: managym.ObservationBatch,
) -> None:
    raw = batch.to_bytes()
    for damaged in (b"", raw[:-1], raw[:-32] + bytes(32)):
        with pytest.raises(ValueError, match="corrupt|truncated"):
            managym.ObservationBatch.from_bytes(damaged)
    header, payload = raw[:-32].split(b"\n", 1)
    with pytest.raises(ValueError, match="truncated"):
        managym.ObservationBatch.from_bytes(seal(header, payload[:-1]))
    with pytest.raises(ValueError, match="trailing"):
        managym.ObservationBatch.from_bytes(seal(header, payload + b"!"))


def test_archive_checks_layout_before_reading_rows(
    batch: managym.ObservationBatch,
) -> None:
    header, payload = batch.to_bytes()[:-32].split(b"\n", 1)
    for version in (9, 10):
        info = json.loads(header)
        info["config"].update(layout="Legacy", turn_state=version == 10)
        with pytest.raises(ValueError, match="v11.*v0.2.6"):
            managym.ObservationBatch.from_bytes(
                seal(json.dumps(info).encode(), payload)
            )
    info = json.loads(header)
    info["config"]["max_actions"] += 1
    with pytest.raises(ValueError, match="layout"):
        managym.ObservationBatch.from_bytes(seal(json.dumps(info).encode(), payload))


@pytest.mark.parametrize(
    ("table", "field", "value", "message"),
    [
        ("actions", "offset:0", 1, "first row offset"),
        ("actions", "offset:1", -1, "offsets or capacity"),
        ("actions", "offset:1", 129, "offsets or capacity"),
        ("action_focus", "offset:1", 0, "action companion offsets differ"),
        # In capacity, but absent from the opening player's live card prefix.
        ("action_focus", "value", 50, "live row"),
        ("actions", "continuation", 64, "frame"),
        ("effects", "member_count", 0x7FFF_FFFF, "partition"),
        ("decision_records", "member_offset", 1, "member offset"),
        ("decision_members", "reference", 0, "member reference"),
        ("decision_references", "detail_count", 0x7FFF_FFFF, "partition"),
    ],
)
def test_archive_rejects_invalid_structures_with_valid_checksum(
    batch: managym.ObservationBatch, table: str, field: str, value: int, message: str
) -> None:
    header = batch.to_bytes().split(b"\n", 1)[0]
    payload = bytearray()
    for key, (_, offsets, values) in batch.tables().items():
        if key == table:
            if field.startswith("offset:"):
                edited = bytearray(offsets)
                struct.pack_into("=i", edited, 4 * int(field.split(":")[1]), value)
                offsets = bytes(edited)
            else:
                fields = batch.layout.csr_fields[key]
                dtype = np.dtype([(name, kind) for name, kind, _ in fields])
                edited = bytearray(values)
                struct.pack_into("=i", edited, dtype.fields[field][1], value)
                values = bytes(edited)
        payload.extend(offsets)
        payload.extend(values)
    with pytest.raises(ValueError, match=message):
        managym.ObservationBatch.from_bytes(seal(header, payload))
