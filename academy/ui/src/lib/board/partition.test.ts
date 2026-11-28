import { describe, expect, it } from 'vitest';

import type { PartitionPilesDecisionGroup } from '$lib/decision-group';
import {
  partitionPlanOf,
  partitionWords,
  sanitizePartitionDraft,
  swapPiles,
  toggleCardPile,
} from './partition';

const rid = (entity: number) => ({ entity, incarnation: 0 });

const FIVE_ITEMS = [
  { item: rid(4), label: 'Gray Ogre' },
  { item: rid(5), label: 'Hill Giant' },
  { item: rid(3), label: 'Forest' },
  { item: rid(13), label: 'Plains' },
  { item: rid(14), label: 'Swamp' },
];

function partitionGroup(committed = 0): PartitionPilesDecisionGroup {
  return {
    match_id: 'curator-match',
    revision: 20 + committed,
    prompt_id: 220 + committed,
    frame_hash: `curator-${committed}`,
    kind: 'partition_piles',
    items: FIVE_ITEMS.slice(committed),
    committed_face_up: FIVE_ITEMS.slice(0, Math.min(1, committed)),
    committed_face_down: FIVE_ITEMS.slice(Math.min(1, committed), committed),
  };
}

describe('partition drafting helpers', () => {
  it('toggles individual cards, swaps piles, sanitizes saved drafts, and builds a PartitionPilesPlan', () => {
    const group = partitionGroup();
    const up1 = toggleCardPile([], 4);
    expect(up1).toEqual([4]);
    const up2 = toggleCardPile(up1, 3);
    expect(up2).toEqual([4, 3]);
    expect(toggleCardPile(up2, 4)).toEqual([3]);

    expect(swapPiles(group.items, [4, 3])).toEqual([5, 13, 14]);
    expect(sanitizePartitionDraft([4, 999, 4, 13], group)).toEqual([4, 13]);
    expect(sanitizePartitionDraft(null, group)).toEqual([]);

    expect(partitionPlanOf([13, 4], group)).toEqual({
      kind: 'partition_piles',
      face_up: [rid(4), rid(13)],
    });
  });

  it('summarizes empty, partial, all-up, and mid-group splits cleanly', () => {
    const start = partitionGroup();
    expect(partitionWords(start, [])).toEqual({
      text: 'Confirm split (0 up, 5 down)',
      plan: '0 face-up · 5 face-down',
    });
    expect(partitionWords(start, [4, 5])).toEqual({
      text: 'Confirm split (2 up, 3 down)',
      plan: '2 face-up (Gray Ogre, Hill Giant) · 3 face-down',
    });
    expect(partitionWords(start, [4, 5, 3, 13, 14])).toEqual({
      text: 'Confirm split (5 up, 0 down)',
      plan: 'All 5 face-up · 0 face-down',
    });
    expect(partitionWords(partitionGroup(2), [3]).text).toBe('Confirm rest (1 up, 2 down)');
  });
});
