import { expect, it } from 'vitest';
import { historyContext, precedingEvent } from './presentation-history';
import { COMBAT_TO_TURN_PRESENTATION } from './presentation-fixtures';
import type { PresentationEvent } from './presentation';

it('ends turn authority at a missing interval, including before a page boundary', () => {
  const events: PresentationEvent[] = Array.from({ length: 104 }, (_, index) => ({
    ...COMBAT_TO_TURN_PRESENTATION.events[0], seq: index < 3 ? index : index + 3,
    kind: index === 0 || index === 102
      ? { kind: 'turn_started', player: 0, turn_number: index === 0 ? 1 : 3 }
      : { kind: 'damage', source: null, target: { kind: 'player', id: 0 }, amount: 1 },
  }));
  const rows = historyContext(events);
  expect(rows[2].turn).toBe(1);
  expect(rows[3]).toEqual({ turn: null, gap: { first: 3, last: 5 } });
  expect(rows.slice(4, 102).every(row => row.turn === null)).toBe(true);
  expect(rows[102].turn).toBe(3);
  expect(rows.slice(100)[0].turn).toBe(null);
});

it('repositions a removed anchor to a preceding fact or the first available fact', () => {
  expect(precedingEvent([10, 12, 15], 12)).toBe(12);
  expect(precedingEvent([10, 12, 15], 14)).toBe(12);
  expect(precedingEvent([10, 12, 15], 2)).toBe(10);
  expect(precedingEvent([], 14)).toBe(null);
});
