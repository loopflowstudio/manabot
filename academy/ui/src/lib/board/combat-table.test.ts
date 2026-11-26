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
      },
      {
        attacker: { entity: 11, name: 'Wandering Musicians', pt: '2/5' },
        blockers: [],
        dealt: '2 to you',
        took: '0',
        result: 'you −2',
        subjects: [11],
      },
    ]);
  });

  it('excludes Main-2 burn damage and deaths from combat and does not hijack Main-2 stops', () => {
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
      ev(4, { kind: 'damage', source: { kind: 'object', id: { entity: 10, incarnation: 1 } }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } }, amount: 3 }, { sound: 'combat.damage' }),
      ev(5, { kind: 'damage', source: { kind: 'object', id: { entity: 20, incarnation: 1 } }, target: { kind: 'object', id: { entity: 10, incarnation: 1 } }, amount: 5 }, { sound: 'combat.damage' }),
      ev(6, { kind: 'damage', source: { kind: 'object', id: { entity: 11, incarnation: 1 } }, target: { kind: 'player', id: 0 }, amount: 2 }, { sound: 'combat.damage' }),
      ev(7, { kind: 'died', objects: [{ entity: 10, incarnation: 1 }] }, {
        cause: 5,
        rule: 'lethal_damage',
        labels: { objects: { '10:1': 'Earth Kingdom Jailer' } },
      }),
      // Main 2: burn spell finishes off surviving blocker Beetle-Headed Merchants.
      ev(8, { kind: 'cast', object: { entity: 30, incarnation: 1 }, controller: 1, stack: 30 }, {
        labels: { objects: { '30:1': 'Lightning Bolt' }, stacks: { '30': 'Lightning Bolt' } },
      }),
      ev(9, { kind: 'resolved', stack: 30 }, { labels: { stacks: { '30': 'Lightning Bolt' } } }),
      ev(10, { kind: 'damage', source: { kind: 'stack', id: 30 }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } }, amount: 3 }, {
        cause: 9,
        sound: 'damage.fire',
        labels: { objects: { '20:1': 'Beetle-Headed Merchants' }, stacks: { '30': 'Lightning Bolt' } },
      }),
      ev(11, { kind: 'died', objects: [{ entity: 20, incarnation: 1 }] }, {
        cause: 9,
        rule: 'lethal_damage',
        labels: { objects: { '20:1': 'Beetle-Headed Merchants' } },
      }),
      // Legacy uncaused death after Main 2 resolution is also excluded.
      ev(12, { kind: 'died', objects: [{ entity: 11, incarnation: 1 }] }, {
        rule: 'zero_toughness',
        labels: { objects: { '11:1': 'Wandering Musicians' } },
      }),
    ];
    const proj = makeProjection(
      [perm(20, 'Beetle-Headed Merchants', 5, 4)],
      [perm(10, 'Earth Kingdom Jailer', 3, 3), perm(11, 'Wandering Musicians', 2, 5)],
    );

    // Main-2 burn damage and death stops never open the combat table.
    expect(buildCombatTable(events, new Set([10]), () => EMPTY, PLAYERS, proj)).toBeNull();
    expect(buildCombatTable(events, new Set([11]), () => EMPTY, PLAYERS, proj)).toBeNull();
    expect(buildCombatTable(events, new Set([12]), () => EMPTY, PLAYERS, proj)).toBeNull();

    // Combat damage stop keeps only combat deaths, without spoiling Result with Main-2 deaths.
    const table = buildCombatTable(events, new Set([4, 5, 6]), () => EMPTY, PLAYERS, proj);
    expect(table).not.toBeNull();
    expect(table!.rows[0].result).toBe('died · lethal damage');
    expect(table!.rows[1].result).toBe('you −2');
  });

  it('keeps first-strike and regular combat steps across an intervening instant while excluding the instant', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 1, turn_number: 7 }),
      ev(2, {
        kind: 'attack_group',
        attackers: [{ entity: 10, incarnation: 1 }, { entity: 11, incarnation: 1 }],
        defender: { kind: 'player', id: 0 },
      }, {
        labels: { objects: { '10:1': 'First Striker', '11:1': 'Regular Attacker' } },
      }),
      ev(3, { kind: 'blocked', attacker: { entity: 10, incarnation: 1 }, blockers: [{ entity: 20, incarnation: 1 }] }, {
        labels: { objects: { '10:1': 'First Striker', '20:1': 'Blocker A' } },
      }),
      ev(4, { kind: 'blocked', attacker: { entity: 11, incarnation: 1 }, blockers: [{ entity: 21, incarnation: 1 }] }, {
        labels: { objects: { '11:1': 'Regular Attacker', '21:1': 'Blocker B' } },
      }),
      // First-strike combat damage kills Blocker A.
      ev(5, { kind: 'damage', source: { kind: 'object', id: { entity: 10, incarnation: 1 } }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } }, amount: 3 }, { sound: 'combat.damage' }),
      ev(6, { kind: 'died', objects: [{ entity: 20, incarnation: 1 }] }, {
        cause: 5,
        rule: 'lethal_damage',
        labels: { objects: { '20:1': 'Blocker A' } },
      }),
      // Intervening instant kills First Striker before regular damage.
      ev(7, { kind: 'cast', object: { entity: 30, incarnation: 1 }, controller: 0, stack: 30 }, {
        labels: { objects: { '30:1': 'Lightning Bolt' }, stacks: { '30': 'Lightning Bolt' } },
      }),
      ev(8, { kind: 'resolved', stack: 30 }, { labels: { stacks: { '30': 'Lightning Bolt' } } }),
      ev(9, { kind: 'damage', source: { kind: 'stack', id: 30 }, target: { kind: 'object', id: { entity: 10, incarnation: 1 } }, amount: 3 }, {
        cause: 8,
        sound: 'damage.fire',
        labels: { objects: { '10:1': 'First Striker' }, stacks: { '30': 'Lightning Bolt' } },
      }),
      ev(10, { kind: 'died', objects: [{ entity: 10, incarnation: 1 }] }, {
        cause: 8,
        rule: 'lethal_damage',
        labels: { objects: { '10:1': 'First Striker' } },
      }),
      // Regular combat damage trades Regular Attacker and Blocker B.
      ev(11, { kind: 'damage', source: { kind: 'object', id: { entity: 11, incarnation: 1 } }, target: { kind: 'object', id: { entity: 21, incarnation: 1 } }, amount: 2 }, { sound: 'combat.damage' }),
      ev(12, { kind: 'damage', source: { kind: 'object', id: { entity: 21, incarnation: 1 } }, target: { kind: 'object', id: { entity: 11, incarnation: 1 } }, amount: 2 }, { sound: 'combat.damage' }),
      ev(13, { kind: 'died', objects: [{ entity: 11, incarnation: 1 }, { entity: 21, incarnation: 1 }] }, {
        cause: 12,
        rule: 'lethal_damage',
        labels: { objects: { '11:1': 'Regular Attacker', '21:1': 'Blocker B' } },
      }),
    ];
    const proj = makeProjection(
      [perm(20, 'Blocker A', 2, 2), perm(21, 'Blocker B', 2, 2)],
      [perm(10, 'First Striker', 3, 2), perm(11, 'Regular Attacker', 2, 2)],
    );

    // Intervening instant damage and death do not open the combat table.
    expect(buildCombatTable(events, new Set([9]), () => EMPTY, PLAYERS, proj)).toBeNull();
    expect(buildCombatTable(events, new Set([10]), () => EMPTY, PLAYERS, proj)).toBeNull();

    // Both first-strike and regular combat stops open the combat table with both combat steps intact.
    const fsTable = buildCombatTable(events, new Set([5]), () => EMPTY, PLAYERS, proj);
    const regTable = buildCombatTable(events, new Set([11, 12]), () => EMPTY, PLAYERS, proj);
    expect(fsTable).not.toBeNull();
    expect(regTable).toEqual(fsTable);
    // First Striker was killed by the instant (not combat damage), while Blocker A died in combat.
    expect(regTable!.rows[0].dealt).toBe('3 to Blocker A');
    expect(regTable!.rows[0].took).toBe('0');
    expect(regTable!.rows[0].result).toBe('Blocker A died');
    expect(regTable!.rows[1].dealt).toBe('2 to Blocker B');
    expect(regTable!.rows[1].took).toBe('2');
    expect(regTable!.rows[1].result).toBe('died · lethal damage · Blocker B died');
  });

  it('ignores non-combat damage pings sourced from a combatant', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 1, turn_number: 5 }),
      ev(2, {
        kind: 'attack_group',
        attackers: [{ entity: 10, incarnation: 1 }, { entity: 11, incarnation: 1 }],
        defender: { kind: 'player', id: 0 },
      }, {
        labels: { objects: { '10:1': 'Pinger Attacker', '11:1': 'Ally' } },
      }),
      // Attack trigger ping from Pinger Attacker (ability-caused, sound damage.fire).
      ev(3, { kind: 'triggered', source: { entity: 10, incarnation: 1 }, controller: 1, stack: 90 }, {
        cause: 2,
        labels: { objects: { '10:1': 'Pinger Attacker' }, stacks: { '90': 'Pinger Attacker' } },
      }),
      ev(4, { kind: 'ability_resolved', source: { entity: 10, incarnation: 1 }, stack: 90 }, {
        labels: { objects: { '10:1': 'Pinger Attacker' }, stacks: { '90': 'Pinger Attacker' } },
      }),
      ev(5, { kind: 'damage', source: { kind: 'object', id: { entity: 10, incarnation: 1 } }, target: { kind: 'player', id: 0 }, amount: 1 }, {
        cause: 4,
        sound: 'damage.fire',
        labels: { objects: { '10:1': 'Pinger Attacker' } },
      }),
      ev(6, { kind: 'blocked', attacker: { entity: 11, incarnation: 1 }, blockers: [{ entity: 20, incarnation: 1 }] }, {
        labels: { objects: { '11:1': 'Ally', '20:1': 'Guard' } },
      }),
      ev(7, { kind: 'damage', source: { kind: 'object', id: { entity: 10, incarnation: 1 } }, target: { kind: 'player', id: 0 }, amount: 2 }, { sound: 'combat.damage' }),
      ev(8, { kind: 'damage', source: { kind: 'object', id: { entity: 11, incarnation: 1 } }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } }, amount: 1 }, { sound: 'combat.damage' }),
      ev(9, { kind: 'damage', source: { kind: 'object', id: { entity: 20, incarnation: 1 } }, target: { kind: 'object', id: { entity: 11, incarnation: 1 } }, amount: 1 }, { sound: 'combat.damage' }),
    ];
    const proj = makeProjection(
      [perm(20, 'Guard', 1, 3)],
      [perm(10, 'Pinger Attacker', 2, 2), perm(11, 'Ally', 1, 1)],
    );

    // Attack-trigger ping stop never triggers the combat table.
    expect(buildCombatTable(events, new Set([5]), () => EMPTY, PLAYERS, proj)).toBeNull();

    // Combat damage stop counts only the 2 combat damage, not the 1 ping.
    const table = buildCombatTable(events, new Set([7, 8, 9]), () => EMPTY, PLAYERS, proj);
    expect(table).not.toBeNull();
    expect(table!.rows[0].dealt).toBe('2 to you');
    expect(table!.rows[0].result).toBe('you −2');
  });
});
