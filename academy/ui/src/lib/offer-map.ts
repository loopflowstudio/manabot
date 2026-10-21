/** View navigation over exact offers. Historical frames retain their old focus bridge. */
import type { InteractionOffer } from './types';

export function offerFocus(offer: InteractionOffer): number[] {
  return offer.subjects?.length ? offer.subjects.map(subject => subject.object.display_id) : offer.focus;
}

export function offerTargets(offers: InteractionOffer[]): Map<number, number[]> {
  const targets = new Map<number, number[]>();
  for (const offer of offers) {
    for (const id of new Set(offerFocus(offer))) {
      const ids = targets.get(id);
      if (ids) ids.push(offer.id);
      else targets.set(id, [offer.id]);
    }
  }
  return targets;
}
