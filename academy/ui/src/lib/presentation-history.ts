import type { PresentationEvent } from './presentation';

/** Derived reading context; missing intervals never carry turn authority. */
export function historyContext(events: readonly PresentationEvent[]): {
  turn: number | null;
  gap: { first: number; last: number } | null;
}[] {
  let turn: number | null = null;
  return events.map((event, index) => {
    const previous = events[index - 1];
    const gap = previous && event.seq > previous.seq + 1
      ? { first: previous.seq + 1, last: event.seq - 1 } : null;
    if (gap) turn = null;
    if (event.kind.kind === 'turn_started') turn = event.kind.turn_number;
    return { turn, gap };
  });
}

/** Same-scope seeks choose the nearest preceding retained fact, then the first. */
export function precedingEvent(seqs: readonly number[], anchor: number): number | null {
  return seqs.findLast(seq => seq <= anchor) ?? seqs[0] ?? null;
}
