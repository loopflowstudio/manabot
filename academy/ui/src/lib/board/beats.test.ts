import { describe, expect, it } from 'vitest';

import type { Observation, PresentationEvent } from '$lib/types';
import {
  BEAT_MS,
  PACE_MS,
  computeStopDeltas,
  cursorView,
  hasWalkableTurn,
  paceMs,
  scrubberStops,
  sharedLaterCount,
  unreadWalkStops,
  walkStops,
} from './beats';
import type { StepBoard } from './boards';
import { buildHistory, turnAt } from './history';

const EMPTY = { objects: {}, players: {}, stacks: {} };
const PLAYERS = { youSeat: 0, you: 'You', opponent: 'Heuristic' };
const obs = (label: string): Observation => ({ label }) as unknown as Observation;

function ev(
  seq: number,
  kind: PresentationEvent['kind'],
  extra: Partial<PresentationEvent> = {},
): PresentationEvent {
  return {
    seq,
    from_revision: seq,
    to_revision: seq + 1,
    caused_by: null,
    group: seq,
    importance: 'normal',
    suggested_ms: 0,
    sound: null,
    kind,
    ...extra,
  };
}

describe('walk stops over kept boards', () => {
  it('groups lines sharing one kept board into one stop and separates cast, damage, death and trigger', () => {
    const events: PresentationEvent[] = [
      ev(40, { kind: 'turn_started', player: 1, turn_number: 6 }),
      ev(41, { kind: 'cast', object: { entity: 12, incarnation: 1 }, controller: 1, stack: 12 }, {
        labels: { objects: { '12:1': 'Lightning Strike' }, stacks: { '12': 'Lightning Strike' } },
      }),
      ev(42, { kind: 'targeted', source: { kind: 'stack', id: 12 }, target: { kind: 'object', id: { entity: 19, incarnation: 1 } } }, {
        cause: 41,
        group: 41,
        labels: { objects: { '19:1': 'Earth Kingdom Jailer' } },
      }),
      ev(43, { kind: 'resolved', stack: 12 }, {
        cause: 41,
        group: 41,
        labels: { stacks: { '12': 'Lightning Strike' } },
      }),
      ev(44, { kind: 'damage', source: { kind: 'stack', id: 12 }, target: { kind: 'object', id: { entity: 19, incarnation: 1 } }, amount: 3 }, {
        cause: 43,
        group: 41,
        labels: { objects: { '19:1': 'Earth Kingdom Jailer' }, stacks: { '12': 'Lightning Strike' } },
      }),
      ev(45, { kind: 'died', objects: [{ entity: 19, incarnation: 1 }] }, {
        cause: 43,
        group: 41,
        rule: 'lethal_damage',
        labels: { objects: { '19:1': 'Earth Kingdom Jailer' } },
      }),
      ev(46, { kind: 'triggered', source: { entity: 24, incarnation: 1 }, controller: 1, stack: 24 }, {
        cause: 45,
        group: 41,
        labels: { objects: { '24:1': 'Tiger-Seal' } },
      }),
      ev(47, { kind: 'ability_resolved', source: { entity: 24, incarnation: 1 }, stack: 24 }, {
        cause: 46,
        group: 41,
        labels: { objects: { '24:1': 'Tiger-Seal' } },
      }),
      ev(48, { kind: 'drew', player: 1 }, {
        cause: 47,
        group: 41,
      }),
    ];
    const boards: StepBoard[] = [
      { first: 41, last: 42, projection: obs('cast') },
      { first: 43, last: 44, projection: obs('damage') },
      { first: 45, last: 45, projection: obs('died') },
      { first: 46, last: 46, projection: obs('trigger') },
      { first: 47, last: 48, projection: obs('resolved-and-drew') },
    ];
    const turns = buildHistory(events, () => EMPTY, PLAYERS);
    const stops = walkStops(turns, boards);

    expect(stops.map(s => ({ seq: s.seq, moveStart: s.moveStart, caption: s.caption }))).toEqual([
      { seq: 41, moveStart: true, caption: 'Heuristic cast Lightning Strike, targeting Earth Kingdom Jailer' },
      { seq: 44, moveStart: false, caption: 'Lightning Strike dealt 3 to Earth Kingdom Jailer' },
      { seq: 45, moveStart: false, caption: 'Earth Kingdom Jailer died · lethal damage' },
      { seq: 46, moveStart: false, caption: 'Trigger · Tiger-Seal' },
      // Ability resolved and drew share board 47..48, so they form one stop.
      { seq: 47, moveStart: false, caption: "Tiger-Seal's trigger resolved · Heuristic drew a card" },
    ]);
    expect(stops[2].rules).toEqual(['lethal_damage']);
    expect(unreadWalkStops(stops, 40).map(s => s.seq)).toEqual([41, 44, 45, 46, 47]);
    expect(unreadWalkStops(stops, 48)).toEqual([]);

    // Scrubber click / hold selects `scope: 'stop'`, preserving both lines, seqs, and full caption of the multi-line stop.
    const heldStop = cursorView(turns, stops, { seq: 47, scope: 'stop' });
    expect([...heldStop.lineIds]).toEqual(['seq-47', 'seq-48']);
    expect([...heldStop.seqs]).toEqual([47, 48]);
    expect(heldStop.caption).toBe("Tiger-Seal's trigger resolved · Heuristic drew a card");

    // Clicking an individual history line or pressing ← → selects `scope: 'line'`, focusing only that line.
    const singleLine = cursorView(turns, stops, { seq: 47, scope: 'line' });
    expect([...singleLine.lineIds]).toEqual(['seq-47']);
    expect([...singleLine.seqs]).toEqual([47]);
    expect(singleLine.caption).toBe("Tiger-Seal's trigger resolved");
  });

  it('walks the person\'s own moves just like the opponent\'s, while skipping a bare draw step alone', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 0, turn_number: 1 }),
      ev(2, { kind: 'cast', object: { entity: 10, incarnation: 1 }, controller: 0, stack: 10 }, {
        caused_by: 'cmd-1',
        labels: { objects: { '10:1': 'Lightning Strike' }, stacks: { '10': 'Lightning Strike' } },
      }),
      ev(3, { kind: 'targeted', source: { kind: 'stack', id: 10 }, target: { kind: 'object', id: { entity: 19, incarnation: 1 } } }, {
        cause: 2,
        group: 2,
        labels: { objects: { '19:1': 'Earth Kingdom Jailer' } },
      }),
      ev(4, { kind: 'resolved', stack: 10 }, {
        cause: 2,
        group: 2,
        labels: { stacks: { '10': 'Lightning Strike' } },
      }),
      ev(5, { kind: 'damage', source: { kind: 'stack', id: 10 }, target: { kind: 'object', id: { entity: 19, incarnation: 1 } }, amount: 3 }, {
        cause: 4,
        group: 2,
        labels: { objects: { '19:1': 'Earth Kingdom Jailer' }, stacks: { '10': 'Lightning Strike' } },
      }),
      ev(6, { kind: 'died', objects: [{ entity: 19, incarnation: 1 }] }, {
        cause: 4,
        group: 2,
        rule: 'lethal_damage',
        labels: { objects: { '19:1': 'Earth Kingdom Jailer' } },
      }),
      ev(7, { kind: 'turn_started', player: 1, turn_number: 2 }),
      ev(8, { kind: 'drew', player: 1 }, { rule: 'draw_step' }),
    ];
    const boards: StepBoard[] = [
      { first: 2, last: 3, projection: obs('cast') },
      { first: 4, last: 5, projection: obs('damage') },
      { first: 6, last: 6, projection: obs('died') },
    ];
    const turns = buildHistory(events, () => EMPTY, PLAYERS);
    const stops = walkStops(turns, boards);
    // Your own spell walks cast -> damage -> death.
    expect(unreadWalkStops(stops, 1).map(s => s.seq)).toEqual([2, 5, 6, 8]);
    // A bare draw step alone after everything else was already shown does not walk.
    expect(unreadWalkStops(stops, 6)).toEqual([]);
    expect(hasWalkableTurn(stops)).toBe(true);
    expect(BEAT_MS).toBe(1400);
    expect(PACE_MS).toEqual({ study: 1400, quick: 700, off: null });
    expect(paceMs('quick')).toBe(700);
    expect(paceMs('off')).toBeNull();
  });

  it('selects the cursor turn or the latest turn with a move for the scrubber', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 0, turn_number: 1 }),
      ev(2, { kind: 'land_played', object: { entity: 10, incarnation: 1 }, player: 0 }, {
        caused_by: 'cmd-1',
        labels: { objects: { '10:1': 'Island' } },
      }),
      ev(3, { kind: 'turn_started', player: 1, turn_number: 2 }),
      ev(4, { kind: 'land_played', object: { entity: 11, incarnation: 1 }, player: 1 }, {
        labels: { objects: { '11:1': 'Plains' } },
      }),
      ev(5, { kind: 'turn_started', player: 0, turn_number: 3 }),
      ev(6, { kind: 'drew', player: 0, object: { entity: 12, incarnation: 1 } }, {
        rule: 'draw_step',
        labels: { objects: { '12:1': 'Mountain' } },
      }),
    ];
    const turns = buildHistory(events, () => EMPTY, PLAYERS);
    const stops = walkStops(turns, []);
    // At now (cursor null), scrubber shows Turn 2 (latest turn with a non-draw-step move).
    expect(scrubberStops(stops, null)).toMatchObject({ turn: 2 });
    expect(scrubberStops(stops, null).stops.map(s => s.seq)).toEqual([4]);
    // Looking back at Turn 1 shows Turn 1's stops.
    expect(scrubberStops(stops, 2)).toMatchObject({ turn: 1 });
    expect(sharedLaterCount(turns, 2, 6)).toBe(2);
  });

  it('computes per-stop deltas for P/T pumps and shrinks', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'pt_changed', objects: [{ entity: 10, incarnation: 0 }, { entity: 11, incarnation: 0 }], power: 1, toughness: 0 }),
      ev(2, { kind: 'pt_changed', objects: [{ entity: 12, incarnation: 0 }], power: -2, toughness: -2 }),
    ];
    const proj = { agent: { id: 0, player_index: 0 }, opponent: { id: 1, player_index: 1 } } as unknown as Observation;
    const deltas = computeStopDeltas(events, new Set([1, 2]), proj);
    expect(deltas.objects.get(10)).toEqual([{ kind: 'pump', text: '+1/+0' }]);
    expect(deltas.objects.get(11)).toEqual([{ kind: 'pump', text: '+1/+0' }]);
    expect(deltas.objects.get(12)).toEqual([{ kind: 'damage', text: '−2/−2' }]);
  });
});
describe('the turn an event happened in', () => {
  const events = [
    ev(10, { kind: 'turn_started', player: 1, turn_number: 10 }),
    ev(11, { kind: 'land_played', object: { entity: 1, incarnation: 1 }, player: 1 }),
    ev(12, { kind: 'turn_started', player: 0, turn_number: 11 }),
  ];

  it('is the last turn start at or before it', () => {
    expect(turnAt(events, 11)).toBe(10);
    expect(turnAt(events, 12)).toBe(11);
    expect(turnAt(events, 40)).toBe(11);
  });

  it('is unknown before the first retained turn start', () => {
    expect(turnAt(events, 9)).toBeNull();
  });
});
