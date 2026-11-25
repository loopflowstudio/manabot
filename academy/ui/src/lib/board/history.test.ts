import { describe, expect, it } from 'vitest';

import type { PresentationEvent, PresentationLabels } from '$lib/types';
import {
  buildHistory,
  cursorReading,
  flattenHistory,
  hasTrackHistory,
  isMoveEvent,
  legacyMoveHistory,
  recentHistory,
  stepCursor,
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

  it('wires sourceSeq, absorbs passes, and words illegal-target fizzles separately from counterspells', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 0, turn_number: 4 }),
      // Spell fizzle (with preceding priority passes absorbed).
      ev(2, { kind: 'cast', object: { entity: 10, incarnation: 1 }, controller: 0, stack: 10 }, {
        caused_by: 'c1',
        labels: { objects: { '10:1': 'Lightning Bolt' }, stacks: { '10': 'Lightning Bolt' } },
      }),
      ev(3, { kind: 'targeted', source: { kind: 'stack', id: 10 }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } } }, {
        cause: 2,
        labels: { objects: { '20:1': 'Gray Ogre' }, stacks: { '10': 'Lightning Bolt' } },
      }),
      ev(4, { kind: 'passed', player: 1 }),
      ev(5, { kind: 'countered', stack: 10 }, {
        rule: 'illegal_targets',
        labels: { stacks: { '10': 'Lightning Bolt' } },
      }),
      // Triggered ability fizzle.
      ev(6, { kind: 'triggered', source: { entity: 30, incarnation: 1 }, controller: 1, stack: 90 }, {
        labels: { objects: { '30:1': 'Vengeful Bloodwitch' }, stacks: { '90': 'Vengeful Bloodwitch' } },
      }),
      ev(7, { kind: 'passed', player: 1 }),
      ev(8, { kind: 'countered', stack: 90 }, {
        rule: 'illegal_targets',
        labels: { stacks: { '90': 'Vengeful Bloodwitch' } },
      }),
      // Activated ability fizzle.
      ev(9, { kind: 'activated', source: { entity: 40, incarnation: 1 }, controller: 0, stack: 91 }, {
        caused_by: 'c2',
        labels: { objects: { '40:1': 'Prodigal Pyromancer' }, stacks: { '91': 'Prodigal Pyromancer' } },
      }),
      ev(10, { kind: 'passed', player: 1 }),
      ev(11, { kind: 'countered', stack: 91 }, {
        rule: 'illegal_targets',
        labels: { stacks: { '91': 'Prodigal Pyromancer' } },
      }),
      // Real counterspell keeps "was countered" indented under the counterspell cast.
      ev(12, { kind: 'cast', object: { entity: 50, incarnation: 1 }, controller: 0, stack: 50 }, {
        caused_by: 'c3',
        labels: { objects: { '50:1': 'Gray Ogre' }, stacks: { '50': 'Gray Ogre' } },
      }),
      ev(13, { kind: 'cast', object: { entity: 51, incarnation: 1 }, controller: 1, stack: 51 }, {
        labels: { objects: { '51:1': 'Counterspell' }, stacks: { '51': 'Counterspell' } },
      }),
      ev(14, { kind: 'targeted', source: { kind: 'stack', id: 51 }, target: { kind: 'stack', id: 50 } }, {
        cause: 13,
        labels: { stacks: { '50': 'Gray Ogre', '51': 'Counterspell' } },
      }),
      ev(15, { kind: 'resolved', stack: 51 }, { labels: { stacks: { '51': 'Counterspell' } } }),
      ev(16, { kind: 'countered', stack: 50 }, {
        cause: 15,
        labels: { stacks: { '50': 'Gray Ogre' } },
      }),
    ];

    const turns = buildHistory(events, () => EMPTY, PLAYERS);
    expect(turns[0].lines.map(line => ({
      text: segmentText(line.segments),
      sourceSeq: line.sourceSeq,
      rule: line.rule,
      children: line.children.map(child => ({ text: segmentText(child.segments), sourceSeq: child.sourceSeq, rule: child.rule })),
    }))).toEqual([
      { text: 'Cast Lightning Bolt, targeting Gray Ogre', sourceSeq: null, rule: null, children: [] },
      { text: 'Lightning Bolt did not resolve · illegal targets', sourceSeq: 2, rule: 'illegal_targets', children: [] },
      { text: 'Trigger · Vengeful Bloodwitch', sourceSeq: null, rule: null, children: [] },
      { text: "Vengeful Bloodwitch's trigger did not resolve · illegal targets", sourceSeq: 6, rule: 'illegal_targets', children: [] },
      { text: 'Activate Prodigal Pyromancer', sourceSeq: null, rule: null, children: [] },
      { text: "Prodigal Pyromancer's ability did not resolve · illegal targets", sourceSeq: 9, rule: 'illegal_targets', children: [] },
      { text: 'Cast Gray Ogre', sourceSeq: null, rule: null, children: [] },
      {
        text: 'Cast Counterspell, targeting Gray Ogre',
        sourceSeq: null,
        rule: null,
        children: [{ text: 'Gray Ogre was countered', sourceSeq: 12, rule: null }],
      },
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

  it('steps the cursor across selectable lines, formats the top-bar reading, and folds authority.auto passes', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 1, turn_number: 6 }),
      ev(2, { kind: 'passed', player: 0 }, { caused_by: 'authority.auto.match.1' }),
      ev(
        3,
        { kind: 'cast', object: { entity: 10, incarnation: 1 }, controller: 1, stack: 10 },
        { labels: { objects: { '10:1': 'Lightning Strike' }, stacks: { '10': 'Lightning Strike' } } },
      ),
      ev(4, { kind: 'resolved', stack: 10 }, { labels: { stacks: { '10': 'Lightning Strike' } } }),
      ev(
        5,
        { kind: 'damage', source: { kind: 'stack', id: 10 }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } }, amount: 3 },
        { cause: 4, labels: { objects: { '20:1': 'Earth Kingdom Jailer' }, stacks: { '10': 'Lightning Strike' } } },
      ),
      ev(
        6,
        { kind: 'died', objects: [{ entity: 20, incarnation: 1 }] },
        { cause: 4, rule: 'lethal_damage', labels: { objects: { '20:1': 'Earth Kingdom Jailer' } } },
      ),
    ];
    const turns = buildHistory(events, () => EMPTY, PLAYERS);
    expect(turns[0].lines[0].passes).toEqual([{ actor: 'auto', who: 'auto', count: 1 }]);
    // Selectable lines skip passes: cast (3), damage (5), died (6).
    expect(flattenHistory(turns).map(item => item.cursor)).toEqual([3, 5, 6]);
    expect(stepCursor(turns, null, -1)).toBe(6);
    expect(stepCursor(turns, 6, -1)).toBe(5);
    expect(stepCursor(turns, 5, -1)).toBe(3);
    expect(stepCursor(turns, 3, -1)).toBe(3);
    expect(stepCursor(turns, 3, 1)).toBe(5);
    expect(stepCursor(turns, 6, 1)).toBeNull();
    expect(cursorReading(turns, events, 6, { you: 'You', opponent: 'Heuristic' }))
      .toBe("Turn 6 · Heuristic's Lightning Strike · Earth Kingdom Jailer died · lethal damage");
    expect(cursorReading(turns, events, 5, { you: 'You', opponent: 'Heuristic' }))
      .toBe("Turn 6 · Heuristic's Lightning Strike · 3 damage to Earth Kingdom Jailer");
    expect(cursorReading(turns, events, 3, { you: 'You', opponent: 'Heuristic' }))
      .toBe('Turn 6 · Heuristic cast Lightning Strike');
  });

  it('names the active line or multi-line stop directly when its parent is not a player move', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 1, turn_number: 6 }),
      ev(2, { kind: 'died', objects: [{ entity: 30, incarnation: 1 }] }, {
        rule: 'lethal_damage',
        labels: { objects: { '30:1': 'Vengeful Bloodwitch' } },
      }),
      ev(3, { kind: 'triggered', source: { entity: 30, incarnation: 1 }, controller: 1, stack: 90 }, {
        cause: 2,
        labels: { objects: { '30:1': 'Vengeful Bloodwitch' }, stacks: { '90': 'Vengeful Bloodwitch' } },
      }),
      ev(4, { kind: 'targeted', source: { kind: 'stack', id: 90 }, target: { kind: 'player', id: 0 } }, {
        cause: 3,
        labels: { stacks: { '90': 'Vengeful Bloodwitch' } },
      }),
      ev(5, { kind: 'ability_resolved', source: { entity: 30, incarnation: 1 }, stack: 90 }, {
        labels: { objects: { '30:1': 'Vengeful Bloodwitch' }, stacks: { '90': 'Vengeful Bloodwitch' } },
      }),
      ev(6, { kind: 'life', player: 0, amount: -1, total: 11 }, { cause: 5 }),
      ev(7, { kind: 'life', player: 1, amount: 1, total: 20 }, { cause: 5 }),
    ];
    const turns = buildHistory(events, () => EMPTY, { youSeat: 0, you: 'You', opponent: 'Search 8' });
    const names = { you: 'You', opponent: 'Search 8' };

    // Parent is a combat death (not a player move): top bar names the trigger itself, not the death.
    expect(cursorReading(turns, events, 3, names))
      .toBe('Turn 6 · Trigger · Vengeful Bloodwitch, targeting you');
    // Parent is trigger resolution (not a player move): top bar names the life change itself.
    expect(cursorReading(turns, events, 6, names))
      .toBe('Turn 6 · You lost 1 life (now 11)');
    // Stop scope on the 2-line drain stop keeps both phrases.
    expect(cursorReading(turns, events, { seq: 6, scope: 'stop' }, names, {
      turn: 6,
      caption: 'You lost 1 life (now 11) · Search 8 gained 1 life (now 20)',
      lineIds: ['seq-6', 'seq-7'],
    })).toBe('Turn 6 · You lost 1 life (now 11) · Search 8 gained 1 life (now 20)');
  });
});

describe('recentHistory', () => {
  it('derives the unread recap tail from history with hoverable card segments, sides, cast stack ids, and maxSeq bounds', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 0, turn_number: 1 }, { from_revision: 1, to_revision: 2 }),
      ev(2, { kind: 'land_played', object: { entity: 10, incarnation: 1 }, player: 0 }, {
        from_revision: 1, to_revision: 2, caused_by: 'cmd-1', labels: { objects: { '10:1': 'Island' } },
      }),
      ev(3, { kind: 'passed', player: 0 }, { from_revision: 2, to_revision: 3, caused_by: 'cmd-2' }),
      ev(4, { kind: 'cast', object: { entity: 44, incarnation: 1 }, controller: 1, stack: 44 }, {
        from_revision: 3, to_revision: 4, labels: { objects: { '44:1': 'Allies at Last' }, stacks: { '44': 'Allies at Last' } },
      }),
      ev(5, { kind: 'targeted', source: { kind: 'stack', id: 44 }, target: { kind: 'object', id: { entity: 137, incarnation: 3 } } }, {
        from_revision: 3, to_revision: 4, cause: 4, labels: { objects: { '137:3': 'Tiger-Seal' }, stacks: { '44': 'Allies at Last' } },
      }),
      ev(6, { kind: 'resolved', stack: 44 }, {
        from_revision: 3, to_revision: 4, cause: 4, labels: { stacks: { '44': 'Allies at Last' } },
      }),
      ev(7, { kind: 'died', objects: [{ entity: 137, incarnation: 3 }] }, {
        from_revision: 3, to_revision: 4, cause: 6, rule: 'lethal_damage', labels: { objects: { '137:3': 'Tiger-Seal' } },
      }),
      ev(8, { kind: 'discarded', objects: [{ entity: 96, incarnation: 2 }], player: 1 }, {
        from_revision: 4, to_revision: 5, labels: { objects: { '96:2': 'Earth Kingdom Jailer' } },
      }),
      ev(9, { kind: 'revealed', cards: ['Kyoshi Warriors'], player: 1 }, {
        from_revision: 5, to_revision: 6,
      }),
    ];
    const turns = buildHistory(events, () => EMPTY, { youSeat: 0, you: 'You', opponent: 'Search 64' });

    expect(recentHistory(turns, null)).toEqual([]);
    const tail = recentHistory(turns, 3);
    expect(tail.map(item => [item.by, item.text, item.spell, item.line.boardSeq])).toEqual([
      ['opponent', 'Search 64 cast Allies at Last, targeting Tiger-Seal', 44, 4],
      [null, 'Tiger-Seal died · lethal damage', null, 7],
      ['opponent', 'Search 64 discarded Earth Kingdom Jailer', null, 8],
      ['opponent', 'Search 64 revealed Kyoshi Warriors', null, 9],
    ]);
    // Hoverable card segments are preserved directly from `tell`.
    expect(tail[0].segments.filter(s => s.card !== null).map(s => s.card)).toEqual(['Allies at Last', 'Tiger-Seal']);
    // Review prefix bound (`maxSeq`) clips later game lines cleanly.
    expect(recentHistory(turns, 3, 7).map(item => item.seq)).toEqual([4, 7]);
  });
});
