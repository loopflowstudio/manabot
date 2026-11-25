import { describe, expect, it } from 'vitest';

import type { OrderLibraryGroup, ScryGroup, SurveilGroup } from '$lib/group';
import {
  arrangePlanOf,
  arrangeWords,
  emptyArrangeDraft,
  isArrangeComplete,
  orderMove,
  orderPlace,
  sanitizeArrangeDraft,
  scryMove,
  scryPlace,
  surveilPlace,
} from './arrange';

const card = (entity: number, label: string) => ({ item: { entity, incarnation: 1 }, label });

const scryGroup: ScryGroup = {
  match_id: 'm', revision: 1, prompt_id: 1, frame_hash: 'h',
  kind: 'scry',
  items: [card(10, 'Island'), card(11, 'Opt'), card(12, 'Bolt')],
  committed_top: [], committed_bottom: [],
};

const surveilGroup: SurveilGroup = {
  match_id: 'm', revision: 2, prompt_id: 2, frame_hash: 'h2',
  kind: 'surveil',
  items: [card(20, 'Forest'), card(21, 'Esper Origins')],
  committed_graveyard: [],
};

const orderTopGroup: OrderLibraryGroup = {
  match_id: 'm', revision: 3, prompt_id: 3, frame_hash: 'h3',
  kind: 'order_library', end: 'top',
  items: [card(30, 'Forest'), card(31, 'Mountain')],
  committed: [],
};

describe('Scry arrangement', () => {
  it('starts unplaced and enables completion only once every card is placed', () => {
    const initial = emptyArrangeDraft(scryGroup);
    expect(initial).toEqual({ kind: 'scry', top: [], bottom: [] });
    expect(isArrangeComplete(initial, scryGroup)).toBe(false);
    expect(arrangeWords(initial, scryGroup).confirmLabel).toBe('Place 3 more cards');
    if (initial.kind !== 'scry') throw new Error('expected scry draft');

    let draft = scryPlace(initial, 10, 'top');
    draft = scryPlace(draft, 11, 'bottom');
    draft = scryPlace(draft, 12, 'bottom');
    expect(isArrangeComplete(draft, scryGroup)).toBe(true);
    expect(arrangeWords(draft, scryGroup).confirmLabel).toBe('Confirm (1 top, 2 bottom)');

    draft = scryMove(draft, 'bottom', 12, -1);
    expect(draft.bottom).toEqual([12, 11]);
    expect(arrangePlanOf(draft, scryGroup)).toEqual({
      kind: 'scry',
      top: [{ entity: 10, incarnation: 1 }],
      bottom: [{ entity: 12, incarnation: 1 }, { entity: 11, incarnation: 1 }],
    });
  });
});

describe('Surveil arrangement', () => {
  it('starts in Keep on top with explicit dynamic Confirm labels', () => {
    const initial = emptyArrangeDraft(surveilGroup);
    expect(isArrangeComplete(initial, surveilGroup)).toBe(true);
    expect(arrangeWords(initial, surveilGroup).confirmLabel).toBe('Keep all 2 on top');
    if (initial.kind !== 'surveil') throw new Error('expected surveil draft');

    let draft = surveilPlace(initial, 20, 'graveyard');
    expect(arrangeWords(draft, surveilGroup).confirmLabel).toBe('Put 1 in graveyard, keep 1 on top');

    draft = surveilPlace(draft, 21, 'graveyard');
    expect(arrangeWords(draft, surveilGroup).confirmLabel).toBe('Put all 2 in graveyard');
    expect(arrangePlanOf(draft, surveilGroup)).toEqual({
      kind: 'surveil',
      graveyard: [{ entity: 20, incarnation: 1 }, { entity: 21, incarnation: 1 }],
    });
  });
});

describe('OrderLibrary arrangement', () => {
  it('orders cards and carries the library end title', () => {
    const initial = emptyArrangeDraft(orderTopGroup);
    expect(isArrangeComplete(initial, orderTopGroup)).toBe(false);
    expect(arrangeWords(initial, orderTopGroup).bucketTitle).toBe('Top of library — drawn first at top');
    if (initial.kind !== 'order_library') throw new Error('expected order_library draft');

    let draft = orderPlace(initial, 31, true);
    draft = orderPlace(draft, 30, true);
    draft = orderMove(draft, 30, -1);
    expect(isArrangeComplete(draft, orderTopGroup)).toBe(true);
    expect(arrangePlanOf(draft, orderTopGroup)).toEqual({
      kind: 'order_library',
      order: [{ entity: 30, incarnation: 1 }, { entity: 31, incarnation: 1 }],
    });
    expect(sanitizeArrangeDraft({ kind: 'order_library', order: [30, 999, 30] }, orderTopGroup))
      .toEqual({ kind: 'order_library', order: [30] });
  });
});
