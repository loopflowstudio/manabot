// Recorded presentation sequences shared by the presentation, socket and stage tests.
import type { ObjectRenderId, SubjectRef } from './types';
import { objectLabelKey, type PresentationSequence } from './presentation';
import curatedCombatToTurn from './fixtures/curated-combat-to-turn-w4.json';

const bolt = { entity: 31, incarnation: 0 } satisfies ObjectRenderId;
const ally = { entity: 77, incarnation: 0 } satisfies ObjectRenderId;
const boltOnStack = { kind: 'stack', id: 4001 } satisfies SubjectRef;
const allyTarget = { kind: 'object', id: ally } satisfies SubjectRef;

/**
 * Recorded vertical-slice fixture: one authoritative update, five semantic
 * beats. Both live and replay players consume this exact array in tests.
 */
export const LIGHTNING_BOLT_PRESENTATION: PresentationSequence = {
  id: 'bolt-kills-ally-v1',
  title: 'An Ally token dies after Lightning Bolt deals lethal damage',
  labels: {
    objects: {
      [objectLabelKey(bolt)]: 'Lightning Bolt',
      [objectLabelKey(ally)]: 'Ally token',
    },
    players: {
      '0': 'Hero',
      '1': 'Opponent',
    },
    stacks: {
      '4001': 'Lightning Bolt',
    },
  },
  events: [
    {
      seq: 900,
      from_revision: 42,
      to_revision: 43,
      caused_by: 'command-bolt-42',
      group: 700,
      importance: 'emphasized',
      suggested_ms: 650,
      sound: 'spell.cast',
      kind: { kind: 'cast', object: bolt, controller: 0, stack: 4001 },
    },
    {
      seq: 901,
      from_revision: 42,
      to_revision: 43,
      caused_by: 'command-bolt-42',
      group: 701,
      importance: 'normal',
      suggested_ms: 500,
      sound: null,
      kind: { kind: 'targeted', source: boltOnStack, target: allyTarget },
    },
    {
      seq: 902,
      from_revision: 42,
      to_revision: 43,
      caused_by: 'command-bolt-42',
      group: 702,
      importance: 'emphasized',
      suggested_ms: 450,
      sound: 'spell.resolve',
      kind: { kind: 'resolved', stack: 4001 },
    },
    {
      seq: 903,
      from_revision: 42,
      to_revision: 43,
      caused_by: 'command-bolt-42',
      group: 703,
      importance: 'critical',
      suggested_ms: 700,
      sound: 'damage.fire',
      kind: { kind: 'damage', source: boltOnStack, target: allyTarget, amount: 3 },
    },
    {
      seq: 904,
      from_revision: 42,
      to_revision: 43,
      caused_by: 'command-bolt-42',
      group: 704,
      importance: 'critical',
      suggested_ms: 650,
      sound: 'creature.dies',
      kind: { kind: 'died', objects: [ally] },
    },
  ],
};

/** Rust/Python-authority recording shared by table, replay, and inspector tests. */
export const COMBAT_TO_TURN_PRESENTATION =
  curatedCombatToTurn as PresentationSequence;
