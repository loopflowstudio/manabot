"""Join already prepared full-layout rows without repeating host preparation.

Flat object, frame, action and program indexes are rebased into the joined
batch. Packed recurrent schedules are rebuilt from their retained lengths.
Catalog programs are shared by rows of the same checkpoint.
"""

from dataclasses import fields, replace
from typing import Any, cast

import numpy as np

from manabot.policy.prepare import (
    EffectRoutes,
    FactPrograms,
    FactRoutes,
    Prepared,
    pack,
)


def concatenate(rows: list[Prepared[np.ndarray]]) -> Prepared[np.ndarray]:
    if not rows:
        raise ValueError("a prepared batch needs rows")
    first = rows[0]
    objects = first.validity.shape[1]
    frames = first.relation_features.shape[1]
    facts = first.facts.kinds.shape[1]
    shifted: list[Prepared[np.ndarray]] = []
    batch_offset = action_offset = 0
    for row in rows:
        if (
            row.binding != first.binding
            or row.validity.shape[1:] != first.validity.shape[1:]
            or row.relation_features.shape[1:] != first.relation_features.shape[1:]
            or row.facts.kinds.shape[1:] != first.facts.kinds.shape[1:]
            or row.attention_columns is not None
            or row.live_index is not None
            or row.bucket_needs is not None
            or row.guard is not None
        ):
            raise ValueError("batched inference requires one admitted full layout")
        shifted.append(
            replace(
                row,
                link_targets=tuple(
                    a + batch_offset * objects for a in row.link_targets
                ),
                relation_objects=tuple(
                    a + batch_offset * objects for a in row.relation_objects
                ),
                frame_indexes=row.frame_indexes + batch_offset * frames,
                action_frames=row.action_frames + batch_offset * frames,
                legal_batches=row.legal_batches + batch_offset,
                logit_positions=np.where(
                    row.legal, row.logit_positions + action_offset, 0
                ),
                focus=row.focus + batch_offset * objects,
                stack_targets=None
                if row.stack_targets is None
                else row.stack_targets + batch_offset * objects,
                facts=replace(
                    row.facts, targets=row.facts.targets + batch_offset * objects
                ),
                events=None
                if row.events is None
                else replace(
                    row.events, focus=row.events.focus + batch_offset * objects
                ),
            )
        )
        batch_offset += row.batch
        action_offset += len(row.legal_batches)

    def join(values: list[Any]) -> Any:
        # All array leaves are NumPy; this local structural traversal preserves
        # the typed public Prepared tree and never interprets model inputs.
        value = values[0]
        if isinstance(value, np.ndarray):
            return np.concatenate(values, axis=0)
        if isinstance(value, tuple):
            return tuple(join([v[i] for v in values]) for i in range(len(value)))
        if value is None:
            if any(v is not None for v in values):
                raise ValueError("prepared optional layouts differ")
            return None
        return type(value)(
            **{
                f.name: join([getattr(v, f.name) for v in values])
                for f in fields(value)
            }
        )

    def rebuild(kind: type, values: list[Any], **joined: Any) -> Any:
        """Join every field of `kind` across `values`, except those given."""
        return kind(
            **{
                f.name: joined[f.name]
                if f.name in joined
                else join([getattr(v, f.name) for v in values])
                for f in fields(kind)
            }
        )

    program_rows: list[np.ndarray] = []
    numbers: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    lengths: list[int] = []
    batch_offset = token_offset = 0
    for row in rows:
        p = row.facts.programs
        size = row.batch * facts
        if p is not None:
            program_rows.append(p.rows + batch_offset * facts)
            numbers.append(p.numbers)
            positions.append(np.where(p.mask, p.positions + token_offset, 0))
            masks.append(p.mask)
            lengths.extend(p.schedule.lengths)
            token_offset += len(p.rows)
        else:
            positions.append(np.zeros(size, np.int64))
            masks.append(np.zeros(size, bool))
        batch_offset += row.batch
    programs = (
        FactPrograms(
            rows=np.concatenate(program_rows),
            numbers=np.concatenate(numbers),
            positions=np.concatenate(positions),
            mask=np.concatenate(masks),
            schedule=pack(lengths),
        )
        if program_rows
        else None
    )
    facts_joined = rebuild(
        FactRoutes, [row.facts for row in shifted], programs=programs
    )
    live_effects = [row.effects for row in rows if row.effects is not None]
    effects = None
    if live_effects:
        total = sum(len(e.numeric) for e in live_effects)
        effect_rows: list[EffectRoutes[np.ndarray]] = []
        lengths = []
        batch_offset = header_offset = detail_offset = 0
        for row in rows:
            e = row.effects
            if e is not None:
                local_headers = 3 * len(e.numeric)
                effect_rows.append(
                    replace(
                        e,
                        controller=e.controller + batch_offset * objects,
                        source_object=e.source_object + batch_offset * objects,
                        routes=e.routes + batch_offset * objects,
                        member_records=e.member_records + header_offset,
                        token_indexes=np.where(
                            e.token_indexes < local_headers,
                            e.token_indexes + 3 * header_offset,
                            e.token_indexes - local_headers + 3 * total + detail_offset,
                        ),
                    )
                )
                lengths.extend(e.schedule.lengths)
                header_offset += len(e.numeric)
                detail_offset += len(e.detail_rows)
            batch_offset += row.batch
        effects = rebuild(EffectRoutes, effect_rows, schedule=pack(lengths))
    return cast(
        Prepared[np.ndarray],
        rebuild(
            Prepared,
            shifted,
            batch=sum(row.batch for row in rows),
            binding=first.binding,
            legal_support=all(row.legal_support for row in rows),
            programs=first.programs,
            facts=facts_joined,
            effects=effects,
        ),
    )
