import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type { CardState, ExperienceFrame, PermanentState, PresentationEvent, PresentationLabels } from '$lib/types';
import { nameBook, recentView, tableLines, turnOwners } from './recent';

const bolt = JSON.parse(readFileSync(new URL('../../../../../protocol/fixtures/bolt-target.json', import.meta.url), 'utf8')) as {
  recovery: { frame: ExperienceFrame; presentation_tail: PresentationEvent[] };
};
const names = { you: 'You', opponent: 'Search 64' };
const labels: PresentationLabels = { objects: { '31:0': 'Lightning Bolt', '77:0': 'Grizzly Bears' }, players: {}, stacks: { 4001: 'Lightning Bolt' } };

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
    // The spell left the stack, so it is shown as resolved with its target.
    expect(view.resolved).toEqual([{ id: 4001, name: 'Lightning Bolt', by: 'you', targets: ['Search 64'] }]);
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
});

describe('nameBook', () => {
  it("names an opponent's spell from hidden hand by the move's narration, and others by what was on screen", () => {
    const after = withBoard(bolt.recovery.frame, projection => { projection.opponent.battlefield = [permanent(93, 'Water Tribe Rallier')]; });
    const book = nameBook([bolt.recovery.frame, after], [
      { id: 'a', actor: 'villain', turn: 4, text: 'Play Plains' },
      { id: 'b', actor: 'villain', turn: 4, text: 'Cast Water Tribe Rallier' },
    ]);
    const cast: PresentationEvent = { ...bolt.recovery.presentation_tail[0], seq: 7, from_revision: 3, to_revision: 4,
      kind: { kind: 'cast', object: { entity: 79, incarnation: 1 }, controller: 1, stack: 79 } };
    const damage: PresentationEvent = { ...bolt.recovery.presentation_tail[3], seq: 8, from_revision: 3, to_revision: 4,
      kind: { kind: 'damage', source: null, target: { kind: 'object', id: { entity: 93, incarnation: 2 } }, amount: 2 } };
    const view = recentView(after, 3, [cast, damage], () => ({ objects: {}, players: {}, stacks: {} }), names, book);
    expect(view.facts.map(fact => fact.text)).toEqual(['Search 64 cast Water Tribe Rallier', 'An effect dealt 2 to Water Tribe Rallier']);
    expect(view.resolved.map(spell => spell.name)).toEqual(['Water Tribe Rallier']);
  });
});

describe('lands and discards', () => {
  it("tells the opponent's land play and discard as facts about the cards they became", () => {
    const after = withBoard(bolt.recovery.frame, projection => {
      projection.opponent.battlefield = [permanent(95, 'Plains')];
      projection.opponent.graveyard = [card(96, 'Earth Kingdom Jailer'), card(97, 'Yip Yip!')];
    });
    const base = bolt.recovery.presentation_tail[0];
    const land: PresentationEvent = { ...base, seq: 7, from_revision: 3, to_revision: 4,
      kind: { kind: 'land_played', object: { entity: 95, incarnation: 1 }, player: 1 } };
    const discard: PresentationEvent = { ...base, seq: 8, from_revision: 4, to_revision: 5,
      kind: { kind: 'discarded', objects: [{ entity: 96, incarnation: 2 }, { entity: 97, incarnation: 2 }], player: 1 } };
    const view = recentView(after, 3, [land, discard], () => ({ objects: {}, players: {}, stacks: {} }), names, nameBook([after], []));
    expect(view.facts.map(fact => [fact.by, fact.text, fact.subjects])).toEqual([
      ['opponent', 'Search 64 played Plains', [95]],
      ['opponent', 'Search 64 discarded Earth Kingdom Jailer and Yip Yip!', [96, 97]],
    ]);
  });
});

describe('tableLines', () => {
  it('draws a spell on the stack to its target, and drops the line once it leaves', () => {
    const onStack = withBoard(bolt.recovery.frame, projection => { projection.agent.stack = [card(4001, 'Lightning Bolt')]; });
    expect(tableLines(onStack, bolt.recovery.presentation_tail)).toEqual([
      { from: { kind: 'stack', id: 4001 }, to: { kind: 'player', index: 1 }, kind: 'target' },
    ]);
    expect(tableLines(bolt.recovery.frame, bolt.recovery.presentation_tail)).toEqual([]);
  });

  it('keeps the line of a spell that just resolved while it is shown beside the stack', () => {
    const resolved = [{ id: 4001, name: 'Lightning Bolt', by: 'you' as const, targets: [] }];
    expect(tableLines(bolt.recovery.frame, bolt.recovery.presentation_tail, resolved)).toHaveLength(1);
  });

  it('joins each blocker to the attacker it blocks', () => {
    const blocked = withBoard(bolt.recovery.frame, projection => {
      projection.opponent.battlefield = [permanent(50, 'Attacker', { attacking: true })];
      projection.agent.battlefield = [permanent(60, 'Blocker', { blocking: 50, controller_id: 0 })];
    });
    expect(tableLines(blocked, [])).toEqual([{ from: { kind: 'object', id: 60 }, to: { kind: 'object', id: 50 }, kind: 'block' }]);
  });

  it('keeps a block drawn after combat from the block events since your last move', () => {
    const after = withBoard(bolt.recovery.frame, projection => {
      projection.opponent.battlefield = [permanent(50, 'Attacker')];
      projection.agent.battlefield = [permanent(60, 'Blocker', { controller_id: 0 })];
    });
    const block: PresentationEvent = { ...bolt.recovery.presentation_tail[0], seq: 9, from_revision: 5, to_revision: 6,
      kind: { kind: 'blocked', attacker: { entity: 50, incarnation: 1 }, blockers: [{ entity: 60, incarnation: 1 }] } };
    const line = { from: { kind: 'object', id: 60 }, to: { kind: 'object', id: 50 }, kind: 'block' };
    expect(tableLines(after, [block], [], 5)).toEqual([line]);
    // Before your move, or with a creature gone, there is no line.
    expect(tableLines(after, [block], [], 6)).toEqual([]);
    const gone = withBoard(after, projection => { projection.opponent.battlefield = []; });
    expect(tableLines(gone, [block], [], 5)).toEqual([]);
  });
});

describe('turnOwners', () => {
  it('knows whose turn each retained turn start was, and the current one', () => {
    const start = (seq: number, player: number, turn: number): PresentationEvent => ({
      ...bolt.recovery.presentation_tail[0], seq, kind: { kind: 'turn_started', player, turn_number: turn },
    });
    const owners = turnOwners(bolt.recovery.frame, [start(1, 0, 1), start(2, 1, 2)]);
    expect(owners.get(1)).toBe('you');
    expect(owners.get(2)).toBe('opponent');
  });
});
