import { describe, expect, it } from 'vitest';
import fixture from '../../../../protocol/fixtures/bolt-target.json';
import { dragMoved, dragOffer, selectForPlay, selectedOffers, selectionMatches } from './play-selection';
import type { ExperienceFrame } from './types';

function handFrame(): ExperienceFrame {
  const frame = structuredClone(fixture.recovery.frame) as ExperienceFrame;
  const card = { id: 100, registry_key: 1, name: 'Lightning Bolt', zone: 'HAND', owner_id: 0,
    mana_value: 1, power: 0, toughness: 0, types: { is_creature: false, is_land: false, is_spell: true,
    is_artifact: false, is_enchantment: false, is_planeswalker: false, is_battle: false } };
  frame.projection.agent.hand = [card];
  frame.offers = [{ ...frame.offers[0], id: 42, verb: 'cast', focus: [card.id],
    source: { kind: 'object', id: { entity: card.id, incarnation: 0 } }, subjects: [] }];
  return frame;
}

describe('frame-bound Play selection', () => {
  it('commits only one exact hand source and keeps ambiguous choices explicit', () => {
    const frame = handFrame();
    const selected = selectForPlay(frame, frame.projection.agent.hand[0].id)!;
    expect(dragOffer(selected, frame)?.id).toBe(42);
    frame.offers.push({ ...frame.offers[0], id: 43 });
    const ambiguous = selectForPlay(frame, selected.object)!;
    expect(selectedOffers(ambiguous, frame).map(offer => offer.id)).toEqual([42, 43]);
    expect(dragOffer(ambiguous, frame)).toBeNull();
    expect(selectForPlay(frame, selected.object + 100)!.offers).toEqual([]);
  });
  it('refuses reused IDs after every frame identity change', () => {
    const frame = handFrame();
    const selected = selectForPlay(frame, frame.projection.agent.hand[0].id)!;
    for (const changed of [{ ...frame, revision: frame.revision + 1 }, { ...frame, frame_hash: 'different' },
      { ...frame, match_id: 'other' }, { ...frame, prompt: { ...frame.prompt!, id: frame.prompt!.id + 1 } }]) {
      expect(selectionMatches(selected, changed)).toBe(false);
      expect(selectedOffers(selected, changed)).toEqual([]);
    }
    expect(selectionMatches(selected, null)).toBe(false);
  });
  it('never infers a cast source from target membership or a non-cast verb', () => {
    const frame = handFrame();
    const selected = selectForPlay(frame, frame.projection.agent.hand[0].id)!;
    frame.offers[0].source = null;
    expect(dragOffer(selected, frame)).toBeNull();
    frame.offers[0].source = { kind: 'object', id: { entity: selected.object, incarnation: 0 } };
    frame.offers[0].verb = 'activate';
    expect(dragOffer(selected, frame)).toBeNull();
  });
  it('requires a deliberate eight-pixel drag', () => {
    const frame = handFrame();
    const drag = { selection: selectForPlay(frame, frame.projection.agent.hand[0].id)!, pointer: 1, x: 50, y: 50, moved: false };
    expect(dragMoved(drag, 54, 55)).toBe(false);
    expect(dragMoved(drag, 50, 58)).toBe(true);
    expect(dragMoved({ ...drag, moved: true }, 50, 50)).toBe(true);
  });
});
