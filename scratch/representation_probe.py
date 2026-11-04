"""Bounded representation algebra, not a second rules engine or training result.

Run: uv run --no-project --python 3.12 scratch/representation_probe.py
Fixtures mirror native effect fields at c823c9c8f; native reachability and neural
learning remain separate acceptance checks. Only ordinary single-pass costs are
measured. No environment, network, training loop or GPU is used.
"""

from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Literal


@dataclass(frozen=True, order=True)
class ObjectRef:
    entity: int
    incarnation: int


@dataclass(frozen=True)
class Payload:
    timestamp: int
    source_definition: str
    source_object: ObjectRef | None
    controller: int
    modification: str
    details: tuple[str, ...]
    duration: Literal["EndOfTurn", "Indefinite", "WhileYouControlSource"]


@dataclass(frozen=True)
class Record:
    payload: Payload
    affected: ObjectRef


@dataclass(frozen=True)
class Membership:
    payload: int
    affected: ObjectRef
    ordinal: int


@dataclass(frozen=True)
class Shared:
    payloads: tuple[Payload, ...]
    memberships: tuple[Membership, ...]


def share(records: tuple[Record, ...]) -> Shared:
    indexes: dict[Payload, int] = {}
    links: list[Membership] = []
    for ordinal, record in enumerate(records):
        index = indexes.setdefault(record.payload, len(indexes))
        links.append(Membership(index, record.affected, ordinal))
    return Shared(tuple(indexes), tuple(links))


def expand(shared: Shared) -> tuple[Record, ...]:
    return tuple(
        Record(shared.payloads[link.payload], link.affected)
        for link in sorted(shared.memberships, key=lambda item: item.ordinal)
    )


def wire(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def bounded_wire(shared: Shared, max_bytes: int) -> bytes:
    """Reject oversized serialized output; this is not preallocation admission."""
    encoded = wire(asdict(shared))
    if len(encoded) > max_bytes:
        raise ValueError("fixture wire-byte budget exceeded")
    return encoded


def effective_keywords(records: tuple[Record, ...], target: ObjectRef) -> frozenset[str]:
    """Restricted keyword-only algebra used to exhibit information loss."""
    result: set[str] = set()
    for record in sorted(records, key=lambda item: item.payload.timestamp):
        if record.affected != target:
            continue
        if record.payload.modification == "AddKeywords":
            result.update(record.payload.details)
        elif record.payload.modification == "LoseKeywords":
            result.difference_update(record.payload.details)
        else:
            raise ValueError("fixture supports only keyword changes")
    return frozenset(result)


def cleanup(records: tuple[Record, ...]) -> tuple[Record, ...]:
    return tuple(record for record in records if record.payload.duration != "EndOfTurn")


@dataclass(frozen=True)
class Frame:
    handle: int
    kind: Literal["stack", "execution", "waiting"]
    order: int | None
    parent: int | None
    source: ObjectRef
    targets: tuple[ObjectRef, ...]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    native = (root / "managym/src/agent/observation.rs").read_text()
    view = native.split("pub struct ContinuousEffectData {", 1)[1].split("\n}", 1)[0]
    assert "pub source_registry_key:" in view
    assert "pub source_object:" not in view and "pub controller:" not in view

    drill = Payload(1, "The Fire Nation Drill", None, 0, "LoseKeywords",
                    ("hexproof", "indestructible"), "EndOfTurn")
    targets = tuple(ObjectRef(index, 0) for index in range(10))
    records = tuple(Record(replace(drill, timestamp=stamp), target)
                    for stamp in range(1, 8) for target in targets)
    started = perf_counter_ns()
    shared = share(records)
    encoded = bounded_wire(shared, 1_048_576)
    elapsed_us = (perf_counter_ns() - started) / 1000
    assert expand(shared) == records
    assert len(shared.payloads) == 7 and len(shared.memberships) == 70
    assert len(set(records)) == 70
    # Identical records retain multiplicity too: sharing is not deduplication.
    assert expand(share(records + records[:1])) == records + records[:1]
    remap = {target: ObjectRef(target.entity + 100, 4) for target in targets}
    renamed = tuple(replace(record, affected=remap[record.affected]) for record in records)
    assert Counter(expand(share(renamed[::-1]))) == Counter(renamed)
    # A later entrant, including the same physical card, gains no old links.
    entrant = ObjectRef(0, 1)
    assert all(link.affected != entrant for link in shared.memberships)

    target = targets[0]
    grant = replace(drill, modification="AddKeywords", details=("flying",))
    short = (Record(grant, target),)
    long = (Record(replace(grant, duration="Indefinite"), target),)
    assert effective_keywords(short, target) == effective_keywords(long, target)
    assert effective_keywords(cleanup(short), target) != effective_keywords(cleanup(long), target)
    # Once the grant leaves a 32-event suffix, neither current effective values
    # nor that suffix says which of these futures applies.
    history_a = ("grant until cleanup",) + ("public life event",) * 32
    history_b = ("grant indefinitely",) + ("public life event",) * 32
    assert history_a[-32:] == history_b[-32:]

    source_a, source_b = ObjectRef(100, 0), ObjectRef(101, 0)
    tether = replace(grant, source_definition="same definition",
                     duration="WhileYouControlSource", source_object=source_a)
    a = Record(tether, target)
    b = Record(replace(tether, source_object=source_b), target)
    assert replace(a.payload, source_object=None) == replace(b.payload, source_object=None)
    assert (a.payload.source_object == source_a) != (b.payload.source_object == source_a)
    assert len(share((a, b)).payloads) == 2

    loss = replace(grant, modification="LoseKeywords", timestamp=2)
    ordered = (Record(grant, target), Record(loss, target))
    reversed_order = (Record(replace(grant, timestamp=2), target),
                      Record(replace(loss, timestamp=1), target))
    assert effective_keywords(ordered, target) != effective_keywords(reversed_order, target)
    # Dropping timestamps leaves exactly the same multiset of payloads.
    assert Counter(replace(r.payload, timestamp=0) for r in ordered) == Counter(
        replace(r.payload, timestamp=0) for r in reversed_order)

    frames = (Frame(10, "stack", 0, None, source_a, (target,)),
              Frame(20, "execution", None, 10, source_a, (target, entrant)),
              Frame(30, "waiting", None, None, source_b, (target,)))
    by_kind = {kind: tuple(frame for frame in frames if frame.kind == kind)
               for kind in ("stack", "execution", "waiting")}
    restored = {frame.handle: frame for group in by_kind.values() for frame in group}
    assert restored == {frame.handle: frame for frame in frames}
    actions = ((0, target, 20), (1, entrant, 30))
    object_rows = {obj: index for index, obj in enumerate((target, entrant))}
    reversed_rows = {obj: index for index, obj in enumerate((entrant, target))}
    assert [object_rows[obj] for _, obj, _ in actions] == [0, 1]
    assert [reversed_rows[obj] for _, obj, _ in actions] == [1, 0]
    assert all(handle in restored for _, _, handle in actions)
    # The old table duplicates each header for every detail row.
    relation_rows = sum(1 + len(frame.targets) for frame in frames)
    relation_bytes = relation_rows * 49 * 4
    typed_relation_bytes = len(wire({kind: [asdict(f) for f in group]
                                    for kind, group in by_kind.items()}))
    # Explicit rejection is necessary even with variable lengths.
    rejected = False
    try:
        bounded_wire(shared, len(encoded) - 1)
    except ValueError:
        rejected = True
    assert rejected
    assert bounded_wire(shared, len(encoded)) == encoded

    print(json.dumps({
        "checks": "PASS: roundtrip, multiplicity, renaming, fixed membership, expiry, history eviction, source identity, timestamp order, typed frame joins, action focus, budget rejection",
        "scope": "synthetic projection algebra; no native replay or neural training",
        "drill": {"records": len(records), "fact_rows": 2 * len(records),
                  "active_fact_float32_bytes": len(records) * 2 * 12 * 4,
                  "flat_json_bytes": len(wire([asdict(record) for record in records])),
                  "shared_json_bytes": len(encoded), "payloads": len(shared.payloads),
                  "memberships": len(shared.memberships),
                  "single_pass_group_and_json_us": elapsed_us},
        "relations": {"frames": len(frames), "detail_edges": 4,
                      "active_49_float_row_bytes": relation_bytes,
                      "padded_64_by_49_bytes": 64 * 49 * 4,
                      "typed_json_bytes": typed_relation_bytes},
    }, indent=2))


if __name__ == "__main__":
    main()
