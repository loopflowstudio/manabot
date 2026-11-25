import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type { CardState, ExperienceFrame, PermanentState, PresentationEvent, PresentationLabels } from '$lib/types';
import { recentMarks, boardLines, targetNames, turnOwners } from './recent';

const names = { you: 'You', opponent: 'Heuristic' };

const bolt = JSON.parse(readFileSync(new URL('../../../../../protocol/fixtures/bolt-target.json', import.meta.url), 'utf8')) as {
  recovery: { frame: ExperienceFrame; presentation_tail: PresentationEvent[] };
};
const labels: PresentationLabels = { objects: { '31:0': 'Lightning Bolt', '77:0': 'Grizzly Bears' }, players: {}, stacks: { 4001: 'Lightning Bolt' } };
const emptyLabels: PresentationLabels = { objects: {}, players: {}, stacks: {} };

function withBoard(frame: ExperienceFrame, change: (projection: ExperienceFrame['projection']) => void): ExperienceFrame {
  const copy = structuredClone(frame);
  change(copy.projection);
  return copy;
}

const card = (id: number, name: string): CardState => ({ ...structuredClone(bolt.recovery.frame.projection.agent.hand[0] ?? {}), id, name } as CardState);
const permanent = (id: number, name: string, extra: Partial<PermanentState> = {}): PermanentState =>
  ({ id, name, controller_id: 1, tapped: false, power: 2, toughness: 2, damage: 0, summoning_sick: false, plus1_counters: 0, ...extra } as PermanentState);

describe('recentMarks', () => {
  it('collects resolved spells, touched objects, and damage totals since the last decision', () => {
    const marks = recentMarks(bolt.recovery.frame, 1, bolt.recovery.presentation_tail, () => labels);
    expect(marks.resolved).toEqual([{ id: 4001, name: 'Lightning Bolt' }]);
    expect([...marks.damage.players]).toEqual([[1, 3]]);
  });

  it('ignores events before sinceRevision or when sinceRevision is null', () => {
    const after = recentMarks(bolt.recovery.frame, 2, bolt.recovery.presentation_tail, () => labels);
    expect(after.resolved).toEqual([]);
    expect(after.touched.size).toBe(0);
    expect(after.damage.players.size).toBe(0);
    expect(recentMarks(bolt.recovery.frame, null, bolt.recovery.presentation_tail, () => labels).resolved).toEqual([]);
  });

  it('tracks touched permanents and event.labels spell names without frame fallback', () => {
    const base = bolt.recovery.presentation_tail[0];
    const events: PresentationEvent[] = [
      { ...base, seq: 10, from_revision: 3, to_revision: 4, kind: { kind: 'cast', object: { entity: 44, incarnation: 1 }, controller: 1, stack: 44 },
        labels: { objects: { '44:1': 'Allies at Last' }, stacks: { '44': 'Allies at Last' } } },
      { ...base, seq: 11, from_revision: 3, to_revision: 4, cause: 10,
        kind: { kind: 'targeted', source: { kind: 'stack', id: 44 }, target: { kind: 'object', id: { entity: 137, incarnation: 3 } } } },
      { ...base, seq: 12, from_revision: 3, to_revision: 4,
        kind: { kind: 'land_played', object: { entity: 95, incarnation: 1 }, player: 1 } },
      { ...base, seq: 13, from_revision: 3, to_revision: 4,
        kind: { kind: 'activated', source: { entity: 93, incarnation: 2 }, controller: 1, stack: 91 } },
      { ...base, seq: 14, from_revision: 3, to_revision: 4,
        kind: { kind: 'counters', object: { entity: 98, incarnation: 1 }, counter: '+1/+1', count: 1, total: 2 } },
    ];
    const marks = recentMarks(bolt.recovery.frame, 3, events, () => emptyLabels);
    expect(marks.resolved).toEqual([{ id: 44, name: 'Allies at Last' }]);
    expect([...marks.touched].sort((a, b) => a - b)).toEqual([93, 95, 98, 137]);
  });
});

describe('boardLines', () => {
  it('draws a spell on the stack to its target, and drops the line once it leaves', () => {
    const onStack = withBoard(bolt.recovery.frame, projection => { projection.agent.stack = [card(4001, 'Lightning Bolt')]; });
    expect(boardLines(onStack, bolt.recovery.presentation_tail)).toEqual([
      { from: { kind: 'stack', id: 4001 }, to: { kind: 'player', index: 1 }, kind: 'target' },
    ]);
    expect(boardLines(bolt.recovery.frame, bolt.recovery.presentation_tail)).toEqual([]);
  });

  it('draws target lines for triggered and activated abilities in stack_objects targeting players or permanents', () => {
    const withAbilities = withBoard(bolt.recovery.frame, projection => {
      projection.opponent.battlefield = [permanent(50, 'Earth Kingdom Jailer')];
      projection.stack_objects = [
        { id: 901, kind: 'triggered_ability', controller: 1, label: 'Vengeful Bloodwitch — Drain', source_permanent: 30 },
        { id: 902, kind: 'activated_ability', controller: 0, label: 'Water Tribe Rallier — Pump', source_permanent: 40 },
      ];
    });
    const base = bolt.recovery.presentation_tail[0];
    const events: PresentationEvent[] = [
      { ...base, seq: 20, kind: { kind: 'targeted', source: { kind: 'stack', id: 901 }, target: { kind: 'player', id: 0 } } },
      { ...base, seq: 21, kind: { kind: 'targeted', source: { kind: 'stack', id: 902 }, target: { kind: 'object', id: { entity: 50, incarnation: 0 } } } },
    ];
    expect(boardLines(withAbilities, events)).toEqual([
      { from: { kind: 'stack', id: 901 }, to: { kind: 'player', index: 0 }, kind: 'target' },
      { from: { kind: 'stack', id: 902 }, to: { kind: 'object', id: 50 }, kind: 'target' },
    ]);
  });

  it('keeps the line of a spell that just resolved while it is shown beside the stack', () => {
    const resolved = [{ id: 4001, name: 'Lightning Bolt' }];
    expect(boardLines(bolt.recovery.frame, bolt.recovery.presentation_tail, resolved)).toHaveLength(1);
  });

  it('draws a triggered ability on the stack to its targets, and names them', () => {
    const trigger = { id: 7, kind: 'triggered_ability', controller: 1, label: 'Glider Staff',
      targets: [{ kind: 'object' as const, id: { entity: 60, incarnation: 1 } }, { kind: 'player' as const, id: 0 }] };
    const frame = withBoard(bolt.recovery.frame, projection => {
      projection.agent.battlefield = [permanent(60, 'Hermitic Herbalist', { controller_id: 0 })];
      projection.stack_objects = [trigger];
    });
    expect(boardLines(frame, [])).toEqual([
      { from: { kind: 'entry', id: 7 }, to: { kind: 'object', id: 60 }, kind: 'target' },
      { from: { kind: 'entry', id: 7 }, to: { kind: 'player', index: frame.projection.agent.player_index }, kind: 'target' },
    ]);
    expect(targetNames(trigger, frame, names)).toEqual(['Hermitic Herbalist', 'you']);
  });

  it('joins each blocker to the attacker it blocks', () => {
    const blocked = withBoard(bolt.recovery.frame, projection => {
      projection.opponent.battlefield = [permanent(50, 'Attacker', { attacking: true })];
      projection.agent.battlefield = [permanent(60, 'Blocker', { blocking: 50, controller_id: 0 })];
    });
    expect(boardLines(blocked, [])).toEqual([{ from: { kind: 'object', id: 60 }, to: { kind: 'object', id: 50 }, kind: 'block' }]);
  });

  it('keeps a block drawn after combat from the block events since your last move', () => {
    const after = withBoard(bolt.recovery.frame, projection => {
      projection.opponent.battlefield = [permanent(50, 'Attacker')];
      projection.agent.battlefield = [permanent(60, 'Blocker', { controller_id: 0 })];
    });
    const block: PresentationEvent = {
      ...bolt.recovery.presentation_tail[0], seq: 10, from_revision: 3, to_revision: 4,
      kind: { kind: 'blocked', attacker: { entity: 50, incarnation: 0 }, blockers: [{ entity: 60, incarnation: 0 }] },
    };
    expect(boardLines(after, [block], [], 3)).toEqual([{ from: { kind: 'object', id: 60 }, to: { kind: 'object', id: 50 }, kind: 'block' }]);
    expect(boardLines(after, [block], [], 4)).toEqual([]);
  });
});

describe('turnOwners', () => {
  it('records each turn owner from turn_started events and the current frame', () => {
    const start: PresentationEvent = {
      ...bolt.recovery.presentation_tail[0], seq: 1,
      kind: { kind: 'turn_started', player: 1, turn_number: 2 },
    };
    const owners = turnOwners(bolt.recovery.frame, [start]);
    expect(owners.get(2)).toBe('opponent');
    expect(owners.get(bolt.recovery.frame.projection.turn.turn_number)).toBe('you');
  });
});
