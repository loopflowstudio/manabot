import { describe, expect, it } from 'vitest';

import type { Observation, PermanentState, PresentationEvent } from '$lib/types';
import { buildCombatTable } from './combat-table';

const EMPTY = { objects: {}, players: {}, stacks: {} };
const PLAYERS = { youSeat: 0, you: 'You', opponent: 'Manabot' };

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

function perm(id: number, name: string, power: number, toughness: number): PermanentState {
  return {
    id,
    name,
    controller_id: 0,
    tapped: false,
    power,
    toughness,
    damage: 0,
    summoning_sick: false,
    plus1_counters: 0,
  } as PermanentState;
}

function makeProjection(agentBattlefield: PermanentState[], opponentBattlefield: PermanentState[]): Observation {
  return {
    agent: { id: 0, player_index: 0, battlefield: agentBattlefield },
    opponent: { id: 1, player_index: 1, battlefield: opponentBattlefield },
  } as unknown as Observation;
}

describe('buildCombatTable', () => {
  it('returns null when fewer than three creatures are in the fight', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 1, turn_number: 6 }),
      ev(2, { kind: 'attack_group', attackers: [{ entity: 10, incarnation: 1 }], defender: { kind: 'player', id: 0 } }, {
        labels: { objects: { '10:1': 'Earth Kingdom Jailer' } },
      }),
      ev(3, { kind: 'blocked', attacker: { entity: 10, incarnation: 1 }, blockers: [{ entity: 20, incarnation: 1 }] }, {
        labels: { objects: { '10:1': 'Earth Kingdom Jailer', '20:1': 'Beetle-Headed Merchants' } },
      }),
      ev(4, { kind: 'damage', source: { kind: 'object', id: { entity: 10, incarnation: 1 } }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } }, amount: 3 }),
      ev(5, { kind: 'damage', source: { kind: 'object', id: { entity: 20, incarnation: 1 } }, target: { kind: 'object', id: { entity: 10, incarnation: 1 } }, amount: 5 }),
    ];
    const proj = makeProjection([perm(20, 'Beetle-Headed Merchants', 5, 4)], [perm(10, 'Earth Kingdom Jailer', 3, 3)]);
    expect(buildCombatTable(events, new Set([4, 5]), () => EMPTY, PLAYERS, proj)).toBeNull();
  });

  it('builds one row per attacker when three or more creatures participate in combat', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 1, turn_number: 6 }),
      ev(2, {
        kind: 'attack_group',
        attackers: [{ entity: 10, incarnation: 1 }, { entity: 11, incarnation: 1 }],
        defender: { kind: 'player', id: 0 },
      }, {
        labels: { objects: { '10:1': 'Earth Kingdom Jailer', '11:1': 'Wandering Musicians' } },
      }),
      ev(3, { kind: 'blocked', attacker: { entity: 10, incarnation: 1 }, blockers: [{ entity: 20, incarnation: 1 }] }, {
        labels: { objects: { '10:1': 'Earth Kingdom Jailer', '20:1': 'Beetle-Headed Merchants' } },
      }),
      ev(4, { kind: 'damage', source: { kind: 'object', id: { entity: 10, incarnation: 1 } }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } }, amount: 3 }),
      ev(5, { kind: 'damage', source: { kind: 'object', id: { entity: 20, incarnation: 1 } }, target: { kind: 'object', id: { entity: 10, incarnation: 1 } }, amount: 5 }),
      ev(6, { kind: 'damage', source: { kind: 'object', id: { entity: 11, incarnation: 1 } }, target: { kind: 'player', id: 0 }, amount: 2 }),
      ev(7, { kind: 'died', objects: [{ entity: 10, incarnation: 1 }] }, {
        cause: 5,
        rule: 'lethal_damage',
        labels: { objects: { '10:1': 'Earth Kingdom Jailer' } },
      }),
    ];
    const priorProj = makeProjection(
      [perm(20, 'Beetle-Headed Merchants', 5, 4)],
      [perm(10, 'Earth Kingdom Jailer', 3, 3), perm(11, 'Wandering Musicians', 2, 5)],
    );
    const afterProj = makeProjection(
      [perm(20, 'Beetle-Headed Merchants', 5, 4)],
      [perm(11, 'Wandering Musicians', 2, 5)],
    );

    // Non-damage/non-death stop (the attack declaration itself) does not show the damage table.
    expect(buildCombatTable(events, new Set([2]), () => EMPTY, PLAYERS, priorProj)).toBeNull();

    // Combat damage stop shows the full breakdown for all 3 creatures.
    const table = buildCombatTable(events, new Set([4, 5, 6]), () => EMPTY, PLAYERS, afterProj, priorProj);
    expect(table).not.toBeNull();
    expect(table!.creatureCount).toBe(3);
    expect(table!.heading).toBe("Manabot's attack · turn 6 (3 creatures)");
    expect(table!.rows).toEqual([
      {
        attacker: { entity: 10, name: 'Earth Kingdom Jailer', pt: '3/3' },
        blockers: [{ entity: 20, name: 'Beetle-Headed Merchants', pt: '5/4' }],
        dealt: '3 to Beetle-Headed Merchants',
        took: '5',
        result: 'died · lethal damage',
        subjects: [10, 20],
        seq: 4,
      },
      {
        attacker: { entity: 11, name: 'Wandering Musicians', pt: '2/5' },
        blockers: [],
        dealt: '2 to you',
        took: '0',
        result: 'you −2',
        subjects: [11],
        seq: 6,
      },
    ]);
  });
});
