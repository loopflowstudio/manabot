import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type { CardState, ExperienceFrame, PermanentState, PresentationEvent, PresentationLabels } from '$lib/types';
import { recentView, boardLines, turnOwners } from './recent';

const bolt = JSON.parse(readFileSync(new URL('../../../../../protocol/fixtures/bolt-target.json', import.meta.url), 'utf8')) as {
  recovery: { frame: ExperienceFrame; presentation_tail: PresentationEvent[] };
};
const names = { you: 'You', opponent: 'Search 64' };
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

describe('recentView', () => {
  it('says what happened since the last decision, from presentation events only', () => {
    const view = recentView(bolt.recovery.frame, 1, bolt.recovery.presentation_tail, () => labels, names);
    expect(view.facts.map(fact => fact.text)).toEqual([
      'You cast Lightning Bolt, targeting Search 64',
      'An effect dealt 3 to Search 64',
      'Grizzly Bears died',
    ]);
    // The spell left the stack, so it is shown as resolved beside the stack.
    expect(view.resolved).toEqual([{ id: 4001, name: 'Lightning Bolt' }]);
  });

  it('calls the person "you" mid-sentence, and "yourself" in their own move', () => {
    const atYou = bolt.recovery.presentation_tail.map(event =>
      event.kind.kind === 'targeted' || event.kind.kind === 'damage'
        ? { ...event, kind: { ...event.kind, target: { kind: 'player' as const, id: 0 } } } : event);
    const view = recentView(bolt.recovery.frame, 1, atYou, () => labels, names);
    expect(view.facts.slice(0, 2).map(fact => fact.text)).toEqual(['You cast Lightning Bolt, targeting yourself', 'An effect dealt 3 to you']);
  });

  it('adds up the damage each player and object was dealt', () => {
    const view = recentView(bolt.recovery.frame, 1, bolt.recovery.presentation_tail, () => labels, names);
    expect([...view.damage.players]).toEqual([[1, 3]]);
  });

  it('shows nothing that happened before the decision it follows', () => {
    const view = recentView(bolt.recovery.frame, 2, bolt.recovery.presentation_tail, () => labels, names);
    expect(view.facts).toEqual([]);
    expect(recentView(bolt.recovery.frame, null, bolt.recovery.presentation_tail, () => labels, names).facts).toEqual([]);
  });

  it('reads exact names from event.labels for hidden-hand spells, activations, triggers, counters, rules and draws', () => {
    const base = bolt.recovery.presentation_tail[0];
    const events: PresentationEvent[] = [
      { ...base, seq: 10, from_revision: 3, to_revision: 4, kind: { kind: 'cast', object: { entity: 44, incarnation: 1 }, controller: 1, stack: 44 },
        labels: { objects: { '44:1': 'Allies at Last' }, stacks: { '44': 'Allies at Last' } } },
      { ...base, seq: 11, from_revision: 3, to_revision: 4, cause: 10,
        kind: { kind: 'targeted', source: { kind: 'stack', id: 44 }, target: { kind: 'object', id: { entity: 137, incarnation: 3 } } },
        labels: { objects: { '137:3': 'Tiger-Seal' }, stacks: { '44': 'Allies at Last' } } },
      { ...base, seq: 12, from_revision: 3, to_revision: 4, rule: 'lethal_damage',
        kind: { kind: 'died', objects: [{ entity: 137, incarnation: 3 }] },
        labels: { objects: { '137:3': 'Tiger-Seal' } } },
      { ...base, seq: 13, from_revision: 3, to_revision: 4, cause: 12,
        kind: { kind: 'triggered', source: { entity: 137, incarnation: 3 }, controller: 0, stack: 90 },
        labels: { objects: { '137:3': 'Tiger-Seal' }, stacks: { '90': 'Tiger-Seal' } } },
      { ...base, seq: 14, from_revision: 3, to_revision: 4,
        kind: { kind: 'activated', source: { entity: 93, incarnation: 2 }, controller: 1, stack: 91 },
        labels: { objects: { '93:2': 'Water Tribe Rallier' }, stacks: { '91': 'Water Tribe Rallier' } } },
      { ...base, seq: 15, from_revision: 3, to_revision: 4,
        kind: { kind: 'counters', object: { entity: 95, incarnation: 1 }, counter: '+1/+1', count: 1, total: 2 },
        labels: { objects: { '95:1': 'South Pole Voyager' } } },
      { ...base, seq: 16, from_revision: 3, to_revision: 4, kind: { kind: 'drew', player: 1 } },
    ];
    const view = recentView(bolt.recovery.frame, 3, events, () => emptyLabels, names);
    expect(view.facts.map(fact => fact.text)).toEqual([
      'Search 64 cast Allies at Last, targeting Tiger-Seal',
      'Tiger-Seal died · lethal damage',
      'Trigger · Tiger-Seal',
      'Search 64 activated Water Tribe Rallier',
      '+1/+1 counter on South Pole Voyager (now 2)',
      'Search 64 drew a card',
    ]);
    expect(view.resolved).toEqual([{ id: 44, name: 'Allies at Last' }]);
  });
});

describe('lands, discards and reveals', () => {
  it("tells the opponent's land play and discard as facts about the cards they became", () => {
    const base = bolt.recovery.presentation_tail[0];
    const land: PresentationEvent = {
      ...base, seq: 7, from_revision: 3, to_revision: 4,
      kind: { kind: 'land_played', object: { entity: 95, incarnation: 1 }, player: 1 },
      labels: { objects: { '95:1': 'Plains' } },
    };
    const discard: PresentationEvent = {
      ...base, seq: 8, from_revision: 4, to_revision: 5,
      kind: { kind: 'discarded', objects: [{ entity: 96, incarnation: 2 }, { entity: 97, incarnation: 2 }], player: 1 },
      labels: { objects: { '96:2': 'Earth Kingdom Jailer', '97:2': 'Yip Yip!' } },
    };
    const view = recentView(bolt.recovery.frame, 3, [land, discard], () => emptyLabels, names);
    expect(view.facts.map(fact => [fact.by, fact.text])).toEqual([
      ['opponent', 'Search 64 played Plains'],
      ['opponent', 'Search 64 discarded Earth Kingdom Jailer and Yip Yip!'],
    ]);
    expect(view.touched.has(95)).toBe(true);
  });

  it("tells the opponent's reveal by the cards' names", () => {
    const base = bolt.recovery.presentation_tail[0];
    const reveal: PresentationEvent = {
      ...base, seq: 7, from_revision: 3, to_revision: 4,
      kind: { kind: 'revealed', cards: ['Kyoshi Warriors'], player: 1 },
    };
    const view = recentView(bolt.recovery.frame, 3, [reveal], () => emptyLabels, names);
    expect(view.facts.map(fact => [fact.by, fact.text])).toEqual([
      ['opponent', 'Search 64 revealed Kyoshi Warriors'],
    ]);
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

  it('keeps the line of a spell that just resolved while it is shown beside the stack', () => {
    const resolved = [{ id: 4001, name: 'Lightning Bolt' }];
    expect(boardLines(bolt.recovery.frame, bolt.recovery.presentation_tail, resolved)).toHaveLength(1);
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
