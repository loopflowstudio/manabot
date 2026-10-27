import { describe, expect, it } from 'vitest';
import { offerFocus, offerTargets } from './offer-map';
import type { InteractionOffer } from './types';
import fixture from '../../../../protocol/fixtures/bolt-target.json';

describe('offer navigation', () => {
  it('keeps historical focus available when typed subjects were not recorded', () => {
    const offers = fixture.recovery.frame.offers as InteractionOffer[];
    for (const offer of offers) {
      expect(offerFocus(offer)).toEqual(offer.focus);
      for (const id of offer.focus) expect(offerTargets(offers).get(id)).toContain(offer.id);
    }
  });

  it('uses exact typed subjects for navigation, including repeated roles', () => {
    const historical = fixture.recovery.frame.offers[0] as InteractionOffer;
    const object = {
      entity_id: 7, display_id: 101, label: 'Same-name creature',
      subject: { kind: 'object' as const, id: { entity: 101, incarnation: 2 } },
    };
    const offer = { ...historical, subjects: [{ role: 'source', object }, { role: 'target', object }] };
    const sibling = { ...offer, id: offer.id + 1 };
    expect(offerFocus(offer)).toEqual([101, 101]);
    expect(offerTargets([offer, sibling]).get(101)).toEqual([offer.id, sibling.id]);
    expect(offerTargets([offer]).get(999)).toBeUndefined();
  });
});
