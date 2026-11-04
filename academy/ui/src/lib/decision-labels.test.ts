import { describe, expect, it } from 'vitest';
import { copyLabel, stageLabel, subjectLabel, visibleCardLabel } from './decision-labels';
import type { Observation } from './types';
import fixture from '../../../../protocol/fixtures/bolt-target.json';

describe('decision labels', () => {
  it('distinguishes identical cards by their visible zone and copy order', () => {
    const observation = structuredClone(fixture.recovery.frame.projection) as Observation;
    const card = { id: 101, registry_key: 1, name: 'Callous Inspector', zone: 'HAND', owner_id: 0, power: 1, toughness: 1, mana_value: 1,
      types: { is_creature: true, is_land: false, is_planeswalker: false, is_enchantment: false, is_artifact: false, is_spell: true, is_battle: false } };
    observation.agent.hand = [{ ...card, id: 101 }, { ...card, id: 202 }];
    expect(copyLabel(observation.agent.hand, 202)).toBe('copy 2 of 2');
    expect(visibleCardLabel(observation, 202)).toBe(`${card.name} · your hand · copy 2 of 2`);
    expect(subjectLabel(observation, { entity_id: 7, display_id: 202, label: card.name,
      subject: { kind: 'object', id: { entity: 202, incarnation: 3 } },
    })).toBe(`${card.name} · your hand · copy 2 of 2`);
    expect(visibleCardLabel(observation, 999)).toBeNull();
  });

  it('keeps absent subjects readable and unknown stages free of internal names', () => {
    const observation = fixture.recovery.frame.projection as Observation;
    expect(subjectLabel(observation, { entity_id: 7, display_id: 999, label: 'Treasure',
      subject: { kind: 'object', id: { entity: 999, incarnation: 1 } },
    })).toBe('Treasure');
    expect(stageLabel('mana_color')).toBe('Choose mana to add');
    expect(stageLabel('private_search')).toBe('Search cards');
    expect(stageLabel('unknown_internal_stage')).toBe('Pending decision');
  });

  it.each([0, 1])('labels inspected cards by native owner and zone from viewer seat %i', (viewer) => {
    const observation = structuredClone(fixture.recovery.frame.projection) as Observation;
    observation.agent.player_index = viewer;
    observation.agent.id = 40 + viewer;
    observation.opponent.player_index = 1 - viewer;
    observation.opponent.id = 41 - viewer;
    const card = { ...observation.agent.hand[0], name: 'Island', zone: 'LIBRARY' };
    observation.agent.inspected = [
      { ...card, id: 900, owner_id: observation.agent.id },
      { ...card, id: 901, owner_id: observation.opponent.id },
      { ...card, id: 902, owner_id: observation.opponent.id },
      { ...card, id: 903, owner_id: observation.opponent.id, zone: 'HAND' },
    ];
    expect(visibleCardLabel(observation, 900)).toBe('Island · your library · private inspection');
    expect(visibleCardLabel(observation, 902)).toBe("Island · opponent's library · private inspection · copy 2 of 2");
    expect(visibleCardLabel(observation, 903)).toBe("Island · opponent's hand · private inspection");
    observation.agent.inspected = [];
    expect(visibleCardLabel(observation, 902)).toBeNull();
  });
});
