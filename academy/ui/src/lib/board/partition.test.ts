import { readFileSync } from 'node:fs';

import { beforeEach, describe, expect, it } from 'vitest';

import type { CardState, ExperienceFrame, InteractionOffer } from '$lib/types';
import {
  isPartitionPiles, loadPartitionDraft, partitionContext, partitionOffer, partitionSteps,
  partitionWords, savePartitionDraft, swapPiles, toggleCardPile,
} from './partition';

const boltFixture = JSON.parse(readFileSync(new URL('../../../../../protocol/fixtures/bolt-target.json', import.meta.url), 'utf8')) as {
  recovery: { frame: ExperienceFrame };
};

function card(id: number, name: string, owner = 0): CardState {
  const creature = name === 'Gray Ogre' || name === 'Hill Giant';
  return {
    id,
    owner_id: owner,
    zone: 'LIBRARY',
    registry_key: id,
    name,
    mana_value: creature ? 3 : 0,
    types: {
      is_creature: creature,
      is_land: !creature,
      is_spell: false,
      is_artifact: false,
      is_enchantment: false,
      is_planeswalker: false,
      is_battle: false,
    },
    power: name === 'Gray Ogre' ? 2 : name === 'Hill Giant' ? 3 : 0,
    toughness: name === 'Gray Ogre' ? 2 : name === 'Hill Giant' ? 3 : 0,
  };
}

function pileOffer(id: number, cardId: number, name: string, pile: 'face-up' | 'face-down'): InteractionOffer {
  const label = `Put ${name} in the ${pile} pile`;
  return {
    id,
    actor: 0,
    verb: 'choose',
    source: null,
    label,
    help: null,
    choices: [],
    confirm_label: label,
    action_type: 'CHOOSE_MODE',
    focus: [cardId],
  };
}

const FIVE_CARDS = [
  card(4, 'Gray Ogre'),
  card(5, 'Hill Giant'),
  card(3, 'Forest'),
  card(13, 'Plains'),
  card(14, 'Swamp'),
];

function partitionFrame(cursor = 0, cards: CardState[] = FIVE_CARDS): ExperienceFrame {
  const frame = structuredClone(boltFixture.recovery.frame);
  const current = cards[cursor];
  frame.action_space = 'MODAL';
  frame.revision = 20 + cursor;
  frame.frame_hash = `curator-${cursor}`;
  frame.projection.agent.inspected = [...cards];
  frame.projection.decision_context = {
    actor: 0,
    stage: 'partition_piles',
    instruction: 'Put the indicated card into the face-up or face-down pile.',
    source: {
      entity_id: 0,
      subject: { kind: 'object', id: { entity: 42, incarnation: 2 } },
      display_id: 42,
      label: 'Curator of Destinies',
    },
    targets: [],
    reserved_sacrifice: null,
    mana_cost: null,
    chosen_mode: null,
    recent_facts: [],
    pending: [`inspected: ${cards.map(c => c.name).join(', ')}.`],
  };
  frame.offers = [
    pileOffer(0, current.id, current.name, 'face-up'),
    pileOffer(1, current.id, current.name, 'face-down'),
  ];
  return frame;
}

describe('partitionContext', () => {
  it('reads all inspected cards in engine order and splits committed vs remaining at the cursor', () => {
    const start = partitionFrame(0);
    expect(isPartitionPiles(start)).toBe(true);
    const ctx0 = partitionContext(start)!;
    expect(ctx0.sourceName).toBe('Curator of Destinies');
    expect(ctx0.committed).toEqual([]);
    expect(ctx0.remaining.map(c => c.id)).toEqual([4, 5, 3, 13, 14]);
    expect(ctx0.currentId).toBe(4);

    const mid = partitionFrame(2);
    const ctx2 = partitionContext(mid)!;
    expect(ctx2.committed.map(c => c.id)).toEqual([4, 5]);
    expect(ctx2.remaining.map(c => c.id)).toEqual([3, 13, 14]);
    expect(ctx2.currentId).toBe(3);
  });

  it('ignores opponent-owned inspected cards and returns null for non-partition stages', () => {
    const withOpponent = partitionFrame(0, [...FIVE_CARDS, card(99, 'Island', 1)]);
    expect(partitionContext(withOpponent)!.remaining.map(c => c.id)).toEqual([4, 5, 3, 13, 14]);

    const other = partitionFrame(0);
    other.projection.decision_context!.stage = 'choose_pile';
    expect(isPartitionPiles(other)).toBe(false);
    expect(partitionContext(other)).toBeNull();
  });
});

describe('drafting and steps', () => {
  it('toggles individual cards, swaps piles, and builds ordered steps', () => {
    const up1 = toggleCardPile([], 4);
    expect(up1).toEqual([4]);
    const up2 = toggleCardPile(up1, 3);
    expect(up2).toEqual([4, 3]);
    expect(toggleCardPile(up2, 4)).toEqual([3]);

    const swapped = swapPiles(FIVE_CARDS, [4, 3]);
    expect(swapped).toEqual([5, 13, 14]);

    const steps = partitionSteps(FIVE_CARDS, [4, 3]);
    expect(steps).toEqual([
      { cardId: 4, cardName: 'Gray Ogre', pile: 'face_up' },
      { cardId: 5, cardName: 'Hill Giant', pile: 'face_down' },
      { cardId: 3, cardName: 'Forest', pile: 'face_up' },
      { cardId: 13, cardName: 'Plains', pile: 'face_down' },
      { cardId: 14, cardName: 'Swamp', pile: 'face_down' },
    ]);
  });

  it('summarizes empty, partial, and all-up splits cleanly', () => {
    expect(partitionWords(FIVE_CARDS, [], 0)).toEqual({
      text: 'Confirm split (0 up, 5 down)',
      plan: '0 face-up · 5 face-down',
    });
    expect(partitionWords(FIVE_CARDS, [4, 5], 0)).toEqual({
      text: 'Confirm split (2 up, 3 down)',
      plan: '2 face-up (Gray Ogre, Hill Giant) · 3 face-down',
    });
    expect(partitionWords(FIVE_CARDS, [4, 5, 3, 13, 14], 0)).toEqual({
      text: 'Confirm split (5 up, 0 down)',
      plan: 'All 5 face-up · 0 face-down',
    });
    expect(partitionWords(FIVE_CARDS.slice(2), [3], 2).text).toBe('Confirm rest (1 up, 2 down)');
  });
});

describe('partitionOffer', () => {
  it('matches the exact card and pile offer and rejects any other card or stage', () => {
    const f0 = partitionFrame(0);
    expect(partitionOffer(f0, { cardId: 4, cardName: 'Gray Ogre', pile: 'face_up' })?.id).toBe(0);
    expect(partitionOffer(f0, { cardId: 4, cardName: 'Gray Ogre', pile: 'face_down' })?.id).toBe(1);
    expect(partitionOffer(f0, { cardId: 5, cardName: 'Hill Giant', pile: 'face_up' })).toBeNull();

    const wrongStage = partitionFrame(0);
    wrongStage.action_space = 'PRIORITY';
    expect(partitionOffer(wrongStage, { cardId: 4, cardName: 'Gray Ogre', pile: 'face_up' })).toBeNull();
  });
});

describe('saved partition draft', () => {
  beforeEach(() => {
    const store = new Map<string, string>();
    globalThis.sessionStorage = {
      getItem: key => store.get(key) ?? null,
      setItem: (key, value) => void store.set(key, value),
    } as Storage;
  });

  it('restores valid ids for the exact same decision only', () => {
    const f0 = partitionFrame(0);
    savePartitionDraft(f0, [4, 13, 999]);
    expect(loadPartitionDraft(f0, new Set([4, 5, 3, 13, 14]))).toEqual([4, 13]);
    expect(loadPartitionDraft({ ...f0, frame_hash: 'other' }, new Set([4, 5, 3, 13, 14]))).toBeNull();
  });
});
