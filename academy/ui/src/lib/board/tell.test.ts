import { describe, expect, it } from 'vitest';

import type { PresentationEvent, PresentationLabels, PresentationRule } from '$lib/types';
import { segmentText, tell, tellRule, type TellPlayers } from './tell';

const EMPTY: PresentationLabels = { objects: {}, players: {}, stacks: {} };
const PLAYERS: TellPlayers = { youSeat: 0, you: 'You', opponent: 'Heuristic' };

function event(
  seq: number,
  kind: PresentationEvent['kind'],
  extra: Partial<PresentationEvent> = {},
): PresentationEvent {
  return {
    seq,
    from_revision: 1,
    to_revision: 2,
    caused_by: null,
    group: seq,
    importance: 'normal',
    suggested_ms: 400,
    sound: null,
    kind,
    ...extra,
  };
}

describe('tell', () => {
  it('words moves in move form and sentence form, emitting card names as inspectable segments', () => {
    const cast = event(
      1,
      { kind: 'cast', object: { entity: 10, incarnation: 1 }, controller: 1, stack: 10 },
      { labels: { objects: { '10:1': 'Lightning Strike' }, stacks: { '10': 'Lightning Strike' } } },
    );
    const targetRef = { kind: 'object' as const, id: { entity: 20, incarnation: 1 } };
    const target = event(
      2,
      { kind: 'targeted', source: { kind: 'stack', id: 10 }, target: targetRef },
      { cause: 1, labels: { objects: { '20:1': 'Earth Kingdom Jailer' }, stacks: { '10': 'Lightning Strike' } } },
    );
    const targets = [{ target: targetRef, labels: EMPTY, event: target }];

    const moveSegments = tell(cast, EMPTY, PLAYERS, { form: 'move', targets });
    expect(segmentText(moveSegments)).toBe('Cast Lightning Strike, targeting Earth Kingdom Jailer');
    expect(moveSegments.filter(s => s.card !== null).map(s => s.card)).toEqual([
      'Lightning Strike',
      'Earth Kingdom Jailer',
    ]);

    const sentenceSegments = tell(cast, EMPTY, PLAYERS, { form: 'sentence', targets });
    expect(segmentText(sentenceSegments)).toBe('Heuristic cast Lightning Strike, targeting Earth Kingdom Jailer');
  });

  it('words triggers, resolutions, activations, counters, draws and rules in plain words', () => {
    const damage = event(
      4,
      { kind: 'damage', source: { kind: 'stack', id: 10 }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } }, amount: 3 },
      { cause: 3, labels: { objects: { '20:1': 'Earth Kingdom Jailer' }, stacks: { '10': 'Lightning Strike' } } },
    );
    expect(segmentText(tell(damage, EMPTY, PLAYERS, { indented: true }))).toBe('3 damage to Earth Kingdom Jailer');
    expect(segmentText(tell(damage, EMPTY, PLAYERS, { indented: false }))).toBe('Lightning Strike dealt 3 to Earth Kingdom Jailer');

    const died = event(
      5,
      { kind: 'died', objects: [{ entity: 20, incarnation: 1 }] },
      { cause: 3, rule: 'lethal_damage', labels: { objects: { '20:1': 'Earth Kingdom Jailer' } } },
    );
    expect(segmentText(tell(died, EMPTY, PLAYERS))).toBe('Earth Kingdom Jailer died · lethal damage');

    const triggered = event(
      6,
      { kind: 'triggered', source: { entity: 30, incarnation: 1 }, controller: 1, stack: 90 },
      { cause: 5, labels: { objects: { '30:1': 'Tiger-Seal' }, stacks: { '90': 'Tiger-Seal' } } },
    );
    expect(segmentText(tell(triggered, EMPTY, PLAYERS))).toBe('Trigger · Tiger-Seal');

    const resolvedTrigger = event(
      7,
      { kind: 'ability_resolved', source: { entity: 30, incarnation: 1 }, stack: 90 },
      { labels: { objects: { '30:1': 'Tiger-Seal' }, stacks: { '90': 'Tiger-Seal' } } },
    );
    expect(segmentText(tell(resolvedTrigger, EMPTY, PLAYERS, { trigger: true }))).toBe("Tiger-Seal's trigger resolved");
    expect(segmentText(tell(resolvedTrigger, EMPTY, PLAYERS, { trigger: false }))).toBe("Tiger-Seal's ability resolved");

    const hiddenDraw = event(8, { kind: 'drew', player: 1 }, { cause: 7 });
    expect(segmentText(tell(hiddenDraw, EMPTY, PLAYERS))).toBe('Heuristic drew a card');

    const ownDraw = event(
      9,
      { kind: 'drew', player: 0, object: { entity: 40, incarnation: 1 } },
      { rule: 'draw_step', labels: { objects: { '40:1': 'Island' } } },
    );
    expect(segmentText(tell(ownDraw, EMPTY, PLAYERS))).toBe('You drew Island · draw step');

    const counter = event(
      10,
      { kind: 'counters', object: { entity: 50, incarnation: 1 }, counter: 'plus_one_plus_one', count: 1, total: 2 },
      { labels: { objects: { '50:1': 'South Pole Voyager' } } },
    );
    expect(segmentText(tell(counter, EMPTY, PLAYERS))).toBe('+1/+1 counter on South Pole Voyager (now 2)');

    const counters = event(
      11,
      { kind: 'counters', object: { entity: 50, incarnation: 1 }, counter: 'plus_one_plus_one', count: 2, total: 4 },
      { labels: { objects: { '50:1': 'South Pole Voyager' } } },
    );
    expect(segmentText(tell(counters, EMPTY, PLAYERS))).toBe('2 +1/+1 counters on South Pole Voyager (now 4)');

    const stun = event(
      12,
      { kind: 'counters', object: { entity: 50, incarnation: 1 }, counter: 'double_strike', count: 1, total: 1 },
      { labels: { objects: { '50:1': 'South Pole Voyager' } } },
    );
    expect(segmentText(tell(stun, EMPTY, PLAYERS))).toBe('double strike counter on South Pole Voyager (now 1)');
  });

  it('prefers event.labels over fallback labels and names unknown objects by role', () => {
    const fallback: PresentationLabels = {
      objects: { '10:1': 'Old Name' },
      players: {},
      stacks: {},
    };
    const withOwn = event(
      1,
      { kind: 'land_played', object: { entity: 10, incarnation: 1 }, player: 0 },
      { labels: { objects: { '10:1': 'Mountain' } } },
    );
    expect(segmentText(tell(withOwn, fallback, PLAYERS))).toBe('Play Mountain');

    const unnamed = event(2, { kind: 'land_played', object: { entity: 99, incarnation: 1 }, player: 1 });
    expect(segmentText(tell(unnamed, EMPTY, PLAYERS))).toBe('Play a land');
  });

  it('words every presentation rule in plain English', () => {
    const rules: [PresentationRule, string][] = [
      ['draw_step', 'draw step'],
      ['lethal_damage', 'lethal damage'],
      ['zero_toughness', 'toughness 0'],
      ['zero_loyalty', 'loyalty 0'],
      ['zero_defense', 'defense 0'],
      ['legend_rule', 'legend rule'],
      ['unattached_aura', 'Aura had nothing to enchant'],
      ['cleanup', 'cleanup'],
      ['zero_life', '0 life'],
      ['empty_library', 'drew from an empty library'],
      ['poison', '10 poison counters'],
    ];
    for (const [rule, words] of rules) {
      expect(tellRule(rule)).toBe(words);
    }
  });

  it('words searched, looked (scry/surveil), sacrificed and lost with rules', () => {
    expect(segmentText(tell(event(1, { kind: 'searched', player: 0 }), EMPTY, PLAYERS)))
      .toBe('You searched your library');
    expect(segmentText(tell(event(2, { kind: 'searched', player: 1 }), EMPTY, PLAYERS)))
      .toBe('Heuristic searched their library');
    expect(segmentText(tell(event(3, { kind: 'looked', player: 0, action: 'scry' }), EMPTY, PLAYERS)))
      .toBe('You scried');
    expect(segmentText(tell(event(4, { kind: 'looked', player: 1, action: 'surveil' }), EMPTY, PLAYERS)))
      .toBe('Heuristic surveilled');
    const sac = tell(
      event(5, { kind: 'sacrificed', player: 0, objects: [{ entity: 7, incarnation: 0 }] }, {
        labels: { objects: { '7:0': 'Evolving Wilds' } },
      }),
      EMPTY,
      PLAYERS,
    );
    expect(segmentText(sac)).toBe('You sacrificed Evolving Wilds');
    expect(sac.find(s => s.card === 'Evolving Wilds')).toBeDefined();
    expect(segmentText(tell(event(6, { kind: 'lost', player: 1 }, { rule: 'zero_life' }), EMPTY, PLAYERS)))
      .toBe('Heuristic lost the game · 0 life');
    expect(segmentText(tell(event(7, { kind: 'lost', player: 0 }, { rule: 'empty_library' }), EMPTY, PLAYERS)))
      .toBe('You lost the game · drew from an empty library');
  });
});
