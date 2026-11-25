import { describe, expect, it } from 'vitest';

import type { PresentationEvent, PresentationLabels } from '$lib/types';
import {
  buildHistory,
  hasTrackHistory,
  isMoveEvent,
  legacyMoveHistory,
  trackMarks,
  type HistoryNote,
} from './history';
import { segmentText, type TellPlayers } from './tell';

const EMPTY: PresentationLabels = { objects: {}, players: {}, stacks: {} };
const PLAYERS: TellPlayers = { youSeat: 0, you: 'You', opponent: 'Heuristic' };

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
    suggested_ms: 400,
    sound: null,
    kind,
    ...extra,
  };
}

describe('buildHistory', () => {
  it('renders the plan Turn 6 example with folded cast, indented causes, rules, trigger resolution and activation', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 1, turn_number: 6 }),
      ev(
        2,
        { kind: 'cast', object: { entity: 10, incarnation: 1 }, controller: 1, stack: 10 },
        { labels: { objects: { '10:1': 'Lightning Strike' }, stacks: { '10': 'Lightning Strike' } } },
      ),
      ev(
        3,
        { kind: 'targeted', source: { kind: 'stack', id: 10 }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } } },
        { cause: 2, group: 2, labels: { objects: { '20:1': 'Earth Kingdom Jailer' }, stacks: { '10': 'Lightning Strike' } } },
      ),
      ev(4, { kind: 'passed', player: 1 }),
      ev(5, { kind: 'passed', player: 0 }, { caused_by: 'cmd-pass-1' }),
      ev(6, { kind: 'resolved', stack: 10 }, { labels: { stacks: { '10': 'Lightning Strike' } } }),
      ev(
        7,
        { kind: 'damage', source: { kind: 'stack', id: 10 }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } }, amount: 3 },
        { cause: 6, group: 6, labels: { objects: { '20:1': 'Earth Kingdom Jailer' }, stacks: { '10': 'Lightning Strike' } } },
      ),
      ev(
        8,
        { kind: 'died', objects: [{ entity: 20, incarnation: 1 }] },
        { cause: 6, group: 6, rule: 'lethal_damage', labels: { objects: { '20:1': 'Earth Kingdom Jailer' } } },
      ),
      ev(
        9,
        { kind: 'triggered', source: { entity: 30, incarnation: 1 }, controller: 1, stack: 90 },
        { cause: 8, group: 6, labels: { objects: { '30:1': 'Tiger-Seal' }, stacks: { '90': 'Tiger-Seal' } } },
      ),
      ev(10, { kind: 'passed', player: 0 }, { caused_by: 'cmd-pass-2' }),
      ev(
        11,
        { kind: 'ability_resolved', source: { entity: 30, incarnation: 1 }, stack: 90 },
        { labels: { objects: { '30:1': 'Tiger-Seal' }, stacks: { '90': 'Tiger-Seal' } } },
      ),
      ev(12, { kind: 'drew', player: 1 }, { cause: 11, group: 11 }),
      ev(
        13,
        { kind: 'activated', source: { entity: 40, incarnation: 1 }, controller: 1, stack: 91 },
        { labels: { objects: { '40:1': 'Water Tribe Rallier' }, stacks: { '91': 'Water Tribe Rallier' } } },
      ),
      ev(
        14,
        { kind: 'ability_resolved', source: { entity: 40, incarnation: 1 }, stack: 91 },
        { labels: { objects: { '40:1': 'Water Tribe Rallier' }, stacks: { '91': 'Water Tribe Rallier' } } },
      ),
      ev(
        15,
        { kind: 'counters', object: { entity: 50, incarnation: 1 }, counter: 'plus_one_plus_one', count: 1, total: 2 },
        { cause: 14, group: 14, labels: { objects: { '50:1': 'South Pole Voyager' } } },
      ),
    ];

    const turns = buildHistory(events, () => EMPTY, PLAYERS);
    expect(turns).toHaveLength(1);
    const turn = turns[0];
    expect(turn.turn).toBe(6);
    expect(turn.owner).toBe('Heuristic');
    expect(turn.consequences).toBe(5);
    expect(turn.busy).toBe(true);

    const rendered = turn.lines.map(line => ({
      who: line.who,
      text: segmentText(line.segments),
      boardSeq: line.boardSeq,
      children: line.children.map(child => segmentText(child.segments)),
    }));
    expect(rendered).toEqual([
      {
        who: 'Heuristic',
        text: 'Cast Lightning Strike, targeting Earth Kingdom Jailer',
        // Folded cast with consequences shows the cast board (spell on stack).
        boardSeq: 2,
        children: [
          '3 damage to Earth Kingdom Jailer',
          'Earth Kingdom Jailer died · lethal damage',
          'Trigger · Tiger-Seal',
        ],
      },
      {
        who: '',
        text: "Tiger-Seal's trigger resolved",
        boardSeq: 11,
        children: ['Heuristic drew a card'],
      },
      {
        who: 'Heuristic',
        text: 'Activate Water Tribe Rallier',
        boardSeq: 13,
        children: ['+1/+1 counter on South Pole Voyager (now 2)'],
      },
    ]);
  });

  it('keeps a response resolution in order without reordering, and picks resolved board for clean folded casts', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 1, turn_number: 6 }),
      ev(
        2,
        { kind: 'cast', object: { entity: 10, incarnation: 1 }, controller: 1, stack: 10 },
        { labels: { objects: { '10:1': 'Lightning Strike' }, stacks: { '10': 'Lightning Strike' } } },
      ),
      ev(
        3,
        { kind: 'targeted', source: { kind: 'stack', id: 10 }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } } },
        { cause: 2, labels: { objects: { '20:1': 'Earth Kingdom Jailer' }, stacks: { '10': 'Lightning Strike' } } },
      ),
      ev(
        4,
        { kind: 'cast', object: { entity: 11, incarnation: 1 }, controller: 0, stack: 11 },
        { caused_by: 'cmd-resp', labels: { objects: { '11:1': 'Accumulate Wisdom' }, stacks: { '11': 'Accumulate Wisdom' } } },
      ),
      ev(5, { kind: 'resolved', stack: 11 }, { labels: { stacks: { '11': 'Accumulate Wisdom' } } }),
      ev(6, { kind: 'drew', player: 0, object: { entity: 12, incarnation: 1 } }, { cause: 5, labels: { objects: { '12:1': 'Island' } } }),
      ev(7, { kind: 'resolved', stack: 10 }, { labels: { stacks: { '10': 'Lightning Strike' } } }),
      ev(
        8,
        { kind: 'damage', source: { kind: 'stack', id: 10 }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } }, amount: 3 },
        { cause: 7, labels: { objects: { '20:1': 'Earth Kingdom Jailer' }, stacks: { '10': 'Lightning Strike' } } },
      ),
      // A plain creature spell resolving cleanly with no consequences: boardSeq is resolved seq.
      ev(
        9,
        { kind: 'cast', object: { entity: 60, incarnation: 1 }, controller: 1, stack: 60 },
        { labels: { objects: { '60:1': 'Gray Ogre' }, stacks: { '60': 'Gray Ogre' } } },
      ),
      ev(10, { kind: 'resolved', stack: 60 }, { labels: { stacks: { '60': 'Gray Ogre' } } }),
    ];

    const turns = buildHistory(events, () => EMPTY, PLAYERS);
    expect(turns[0].busy).toBe(false);
    expect(turns[0].lines.map(line => [line.who, segmentText(line.segments), line.sourceSeq, line.boardSeq, line.children.map(c => segmentText(c.segments))])).toEqual([
      ['Heuristic', 'Cast Lightning Strike, targeting Earth Kingdom Jailer', null, 2, []],
      ['You', 'Cast Accumulate Wisdom', null, 4, ['You drew Island']],
      ['', 'Lightning Strike resolved', 2, 7, ['3 damage to Earth Kingdom Jailer']],
      ['Heuristic', 'Cast Gray Ogre', null, 10, []],
    ]);
  });

  it('folds opponent and automated passes into quiet runs while keeping your own passes and client-local notes', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 0, turn_number: 2 }),
      ev(2, { kind: 'passed', player: 0 }, { caused_by: 'cmd-1' }),
      ev(3, { kind: 'passed', player: 1 }),
      ev(4, { kind: 'passed', player: 0 }), // auto-pass (caused_by: null)
      ev(5, { kind: 'passed', player: 1 }),
      ev(6, { kind: 'land_played', object: { entity: 70, incarnation: 1 }, player: 0 }, { caused_by: 'cmd-2', labels: { objects: { '70:1': 'Mountain' } } }),
    ];
    const notes: HistoryNote[] = [
      { id: 'n1', actor: 'auto', turn: 2, afterSeq: 5, text: 'Stopped once at First main on your turn; that stop is now off' },
    ];

    const turns = buildHistory(events, () => EMPTY, PLAYERS, { notes });
    expect(turns[0].lines.map(line => [line.actor, line.who, line.passes ?? segmentText(line.segments)])).toEqual([
      ['you', 'You', 'Pass priority'],
      ['note', '', [{ actor: 'opponent', who: 'Heuristic', count: 2 }, { actor: 'auto', who: 'auto', count: 1 }]],
      ['auto', 'auto', 'Stopped once at First main on your turn; that stop is now off'],
      ['you', 'You', 'Play Mountain'],
    ]);
  });
});

describe('isMoveEvent, hasTrackHistory and trackMarks', () => {
  it('classifies exact move kinds and ignores turn headings, draw steps and triggers', () => {
    expect(isMoveEvent(ev(1, { kind: 'cast', object: { entity: 1, incarnation: 0 }, controller: 0, stack: 1 }))).toBe(true);
    expect(isMoveEvent(ev(2, { kind: 'passed', player: 0 }))).toBe(true);
    expect(isMoveEvent(ev(3, { kind: 'turn_started', player: 0, turn_number: 1 }))).toBe(false);
    expect(isMoveEvent(ev(4, { kind: 'drew', player: 0 }, { rule: 'draw_step' }))).toBe(false);
    expect(isMoveEvent(ev(5, { kind: 'triggered', source: { entity: 2, incarnation: 0 }, controller: 0, stack: 2 }))).toBe(false);
  });

  it('builds live decision strip marks from move events on the track', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 0, turn_number: 1 }),
      ev(2, { kind: 'land_played', object: { entity: 5, incarnation: 1 }, player: 0 }, { caused_by: 'c1', labels: { objects: { '5:1': 'Island' } } }),
      ev(3, { kind: 'passed', player: 0 }),
      ev(4, { kind: 'turn_started', player: 1, turn_number: 2 }),
      ev(5, { kind: 'land_played', object: { entity: 6, incarnation: 1 }, player: 1 }, { labels: { objects: { '6:1': 'Plains' } } }),
    ];
    expect(hasTrackHistory(events)).toBe(true);
    expect(trackMarks(events, () => EMPTY, PLAYERS)).toEqual([
      { at: 0, key: 'seq-2', turn: 1, turnStart: true, decision: true, side: 'you', value: null, label: 'Turn 1 · Play Island' },
      { at: 1, key: 'seq-3', turn: 1, turnStart: false, decision: false, side: null, value: null, label: 'Turn 1 · Pass priority' },
      { at: 2, key: 'seq-5', turn: 2, turnStart: true, decision: true, side: 'opponent', value: null, label: 'Turn 2 · Play Plains' },
    ]);
  });

  it('preserves frozen legacy move lists for pre-track recordings', () => {
    const turns = legacyMoveHistory(
      [
        { id: 'c1', actor: 'hero', turn: 1, text: 'Play Island' },
        { id: 'c2', actor: 'auto', turn: 1, text: 'Pass priority' },
        { id: 'c3', actor: 'villain', turn: 1, text: 'Pass priority' },
      ],
      { you: 'You', opponent: 'Heuristic' },
      new Map([[1, 'you']]),
      ['Island'],
    );
    expect(turns[0].lines.map(l => [l.actor, l.passes ?? segmentText(l.segments)])).toEqual([
      ['you', 'Play Island'],
      ['note', [{ actor: 'auto', who: 'auto', count: 1 }, { actor: 'opponent', who: 'Heuristic', count: 1 }]],
    ]);
  });
});
