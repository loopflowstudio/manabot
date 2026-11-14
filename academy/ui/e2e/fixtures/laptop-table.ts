import boltFixtureJson from '../../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type {
  CardState,
  ExperienceFrame,
  InteractionOffer,
  Observation,
  PermanentState,
  RecoveryEnvelope,
} from '../../src/lib/types';

const boltFixture = boltFixtureJson as { recovery: RecoveryEnvelope };

interface PermanentSpec {
  name: string;
  power?: number;
  toughness?: number;
  basePower?: number;
  baseToughness?: number;
  tapped?: boolean;
  sick?: boolean;
  damage?: number;
  counters?: number;
}

let nextId = 500;

function permanent(controllerId: number, spec: PermanentSpec): PermanentState {
  const isCreature = spec.power !== undefined;
  return {
    id: nextId++,
    name: spec.name,
    controller_id: controllerId,
    tapped: spec.tapped ?? false,
    damage: spec.damage ?? 0,
    summoning_sick: spec.sick ?? false,
    power: isCreature ? (spec.power ?? null) : null,
    toughness: isCreature ? (spec.toughness ?? null) : null,
    base_power: isCreature ? (spec.basePower ?? spec.power ?? null) : null,
    base_toughness: isCreature ? (spec.baseToughness ?? spec.toughness ?? null) : null,
    plus1_counters: spec.counters ?? 0,
    types: { is_creature: isCreature, is_land: !isCreature, is_artifact: false,
      is_enchantment: false, is_planeswalker: false, is_spell: false, is_battle: false },
  };
}

function graveCard(ownerId: number, name: string, creature: boolean, pt: number): CardState {
  return {
    id: nextId++,
    registry_key: nextId,
    name,
    zone: 'GRAVEYARD',
    owner_id: ownerId,
    power: creature ? pt : 0,
    toughness: creature ? pt : 0,
    mana_value: creature ? pt : 1,
    types: {
      is_creature: creature,
      is_land: false,
      is_spell: !creature,
      is_artifact: false,
      is_enchantment: false,
      is_planeswalker: false,
      is_battle: false,
    },
  };
}

function denseObservation(): Observation {
  const observation = structuredClone(boltFixture.recovery.frame.projection);
  observation.turn = {
    turn_number: 9,
    phase: 'PRECOMBAT_MAIN',
    step: 'PRECOMBAT_MAIN_STEP',
    active_player_id: 0,
    agent_player_id: 0,
  };

  // Hero: 18 permanents — a mid-game UR board with tapped lands, fresh
  // (summoning-sick) creatures, combat damage, and one buffed creature.
  observation.agent.battlefield = [
    permanent(0, { name: 'Otter-Penguin', power: 2, toughness: 2, tapped: true, damage: 1 }),
    permanent(0, { name: 'Otter-Penguin', power: 2, toughness: 2 }),
    permanent(0, { name: 'Fire Nation Cadets', power: 3, toughness: 2, tapped: true }),
    permanent(0, { name: 'Fire Nation Cadets', power: 3, toughness: 2, sick: true }),
    permanent(0, {
      name: 'Tiger-Seal',
      power: 5,
      toughness: 5,
      basePower: 3,
      baseToughness: 3,
      counters: 2,
      damage: 2,
    }),
    permanent(0, { name: 'First-Time Flyer', power: 1, toughness: 3, sick: true }),
    permanent(0, { name: 'First-Time Flyer', power: 1, toughness: 3, tapped: true, damage: 2 }),
    permanent(0, { name: 'Dragonfly Swarm', power: 4, toughness: 4, sick: true }),
    permanent(0, { name: 'Mountain', tapped: true }),
    permanent(0, { name: 'Mountain', tapped: true }),
    permanent(0, { name: 'Mountain', tapped: true }),
    permanent(0, { name: 'Mountain' }),
    permanent(0, { name: 'Mountain' }),
    permanent(0, { name: 'Island', tapped: true }),
    permanent(0, { name: 'Island', tapped: true }),
    permanent(0, { name: 'Island' }),
    permanent(0, { name: 'Island' }),
    permanent(0, { name: 'Island' }),
  ];
  observation.agent.graveyard = [
    graveCard(0, 'Otter-Penguin', true, 2),
    graveCard(0, 'Tiger-Seal', true, 3),
    graveCard(0, 'Boomerang Toss', false, 0),
    graveCard(0, 'Firebending Lesson', false, 0),
    graveCard(0, 'First-Time Flyer', true, 1),
    graveCard(0, 'Dragonfly Swarm', true, 4),
  ];
  observation.agent.hand = Array.from({ length: 12 }, (_, index) => ({
    ...graveCard(0, index % 2 ? 'Island' : 'Boomerang Toss', false, 0), zone: 'HAND',
  }));
  observation.agent.zone_counts.HAND = observation.agent.hand.length;
  observation.agent.life = 11;

  // Villain: 15 permanents — GW allies with tokens and tapped lands.
  observation.opponent.battlefield = [
    permanent(1, { name: 'Badgermole Cub', power: 1, toughness: 1, tapped: true }),
    permanent(1, { name: 'Badgermole Cub', power: 1, toughness: 1, damage: 1 }),
    permanent(1, { name: 'Suki, Kyoshi Warrior', power: 2, toughness: 3, counters: 1 }),
    permanent(1, { name: 'Kyoshi Warriors', power: 4, toughness: 3, tapped: true, damage: 2 }),
    permanent(1, { name: 'Kyoshi Warriors', power: 4, toughness: 3, sick: true }),
    permanent(1, { name: 'Ally', power: 1, toughness: 1 }),
    permanent(1, { name: 'Ally', power: 1, toughness: 1, tapped: true }),
    permanent(1, { name: 'Ally', power: 1, toughness: 1, sick: true }),
    permanent(1, { name: 'Forest', tapped: true }),
    permanent(1, { name: 'Forest', tapped: true }),
    permanent(1, { name: 'Forest' }),
    permanent(1, { name: 'Forest' }),
    permanent(1, { name: 'Forest' }),
    permanent(1, { name: 'Plains', tapped: true }),
    permanent(1, { name: 'Plains' }),
  ];
  observation.opponent.graveyard = [
    graveCard(1, 'Badgermole Cub', true, 1),
    graveCard(1, 'Ally', true, 1),
    graveCard(1, 'Ally', true, 1),
    graveCard(1, 'Earthbending Lesson', false, 0),
    graveCard(1, 'Kyoshi Warriors', true, 4),
  ];
  observation.opponent.life = 14;

  return observation;
}

function passPriorityOffer(): InteractionOffer {
  return {
    id: 901,
    actor: 0,
    verb: 'pass_priority',
    source: null,
    label: 'Pass priority',
    help: null,
    choices: [],
    confirm_label: 'Pass priority',
    action_type: 'PRIORITY_PASS_PRIORITY',
    focus: [],
  };
}

export function denseFrame(): ExperienceFrame {
  const base = structuredClone(boltFixture.recovery.frame);
  return {
    ...base,
    match_id: 'dense-board-stress',
    revision: 40,
    frame_hash: 'dense-board-frame-40',
    prompt: {
      id: 400,
      actor: 0,
      kind: 'priority',
      title: 'You have priority',
      instruction: 'You have priority',
    },
    projection: denseObservation(),
    offers: [passPriorityOffer()],
    action_space: 'PRIORITY',
    deck_names: { hero: 'UR Lessons', villain: 'GW Allies' },
  };
}
