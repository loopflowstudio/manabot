import { describe, expect, it } from 'vitest';

import {
  CURATED_PACK,
  fnv1a32,
  resolveTreatment,
  treatmentBackground,
  validateCuratedPack,
} from './curated-pack';

describe('curated matchup pack', () => {
  it('freezes the current decks and every reachable identity', () => {
    expect(CURATED_PACK.pack).toEqual({
      id: 'tla-ur-lessons-vs-gw-allies',
      version: '1.0.0',
      title: 'UR Lessons vs GW Allies',
    });
    expect(CURATED_PACK.matchup.hero.card_count).toBe(41);
    expect(CURATED_PACK.matchup.villain.card_count).toBe(40);
    expect(Object.keys(CURATED_PACK.identities)).toHaveLength(31);
    expect(CURATED_PACK.matchup.reachable_tokens).toEqual(['Ally', 'Clue']);

    for (const name of Object.keys(CURATED_PACK.identities)) {
      const treatment = resolveTreatment(name);
      expect(treatment.source, name).toBe('pack');
      expect(treatment.palette, name).toHaveLength(3);
      expect(treatmentBackground(treatment), name).not.toContain('url(');
    }
  });

  it('uses a stable UTF-8 fallback without remote assets', () => {
    const first = resolveTreatment('Unknown Snow-Covered Æther Card');
    const second = resolveTreatment('Unknown Snow-Covered Æther Card');

    expect(fnv1a32('Unknown Snow-Covered Æther Card')).toBe(1239946727);
    expect(first).toEqual(second);
    expect(first).toMatchObject({
      source: 'fallback',
      identityKind: 'unknown',
      rights_ref: 'authored-treatment-v1',
    });
    expect(treatmentBackground(first)).not.toContain('url(');
  });

  it('rejects a remote treatment even when provenance may be remote', () => {
    const invalid = structuredClone(CURATED_PACK) as unknown as Record<string, any>;
    invalid.identities.Island.treatment.image = 'https://example.test/island.png';

    expect(() => validateCuratedPack(invalid)).toThrow(/treatment must be local/);
  });
});
