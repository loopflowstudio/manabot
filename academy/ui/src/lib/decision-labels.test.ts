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
    expect(stageLabel('unknown_internal_stage')).toBe('Pending decision');
  });
});
