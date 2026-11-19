import { readFileSync } from 'node:fs';

import { beforeEach, describe, expect, it } from 'vitest';

import type { ExperienceFrame } from '$lib/types';
import {
  drawDraft, draftWords, EMPTY_DRAFT, loadDraft, pileCounts, planKey, planOf, saveDraft, setPileCount, stepOffer, toggleAttacker, toggleBlock,
  type CombatPreview, type DeclarableCreature, type DeclarationStep,
} from './combat';

const fixture = JSON.parse(readFileSync(new URL('../../../../../managym/academy/fixtures/choice-navigation-w17-smart-payment.json', import.meta.url), 'utf8')) as {
  positions: Record<string, { frame: ExperienceFrame }>;
};
/** A real block: Otter-Penguin (110) may block Earth King's Lieutenant (105). */
const block = fixture.positions.DECLARE_BLOCKER.frame;
const [otter, lieutenant] = [110, 105];

const creature = (entity: number, label: string, canBlock: number[] = []): DeclarableCreature => ({
  creature: { entity, incarnation: 2 }, label, can_block: canBlock.map(id => ({ entity: id, incarnation: 3 })), must_block: false,
});
const step = (creature: number, attack: boolean, blocks: number | null = null, index = 0): DeclarationStep => ({
  revision: 1, fingerprint: 'f', creature: { entity: creature, incarnation: 2 }, action_index: index, attack,
  blocks: blocks === null ? null : { entity: blocks, incarnation: 3 },
});
const names = new Map([[1, 'Tiger-Seal'], [2, 'Otter-Penguin'], [3, 'Ally'], [4, 'Ally'], [5, 'Ally'], [otter, 'Otter-Penguin'], [lieutenant, "Earth King's Lieutenant"]]);

describe('draftWords', () => {
  const attackers = [creature(1, 'Tiger-Seal'), creature(2, 'Otter-Penguin'), creature(3, 'Ally'), creature(4, 'Ally')];

  it('reads the whole attack, naming who stays home', () => {
    expect(draftWords('attackers', { attackers: [2, 3, 4], blocks: [] }, attackers, names))
      .toEqual({ text: 'Attack with 3', plan: 'Attack with 3 · not with Tiger-Seal' });
    expect(draftWords('attackers', { attackers: [2], blocks: [] }, attackers, names).plan)
      .toBe('Attack with Otter-Penguin · not with Tiger-Seal and Ally ×2');
    expect(draftWords('attackers', { attackers: [1, 2, 3, 4], blocks: [] }, attackers, names).plan).toBe('Attack with all 4');
    expect(draftWords('attackers', EMPTY_DRAFT, attackers, names).text).toBe('No attack');
  });

  it('reads blocks by the attacker they stop', () => {
    const words = draftWords('blockers', { attackers: [], blocks: [[3, 1], [4, 1], [2, 5]] }, [], names);
    expect(words).toEqual({ text: 'Block with 3', plan: 'Ally ×2 on Tiger-Seal · Otter-Penguin on Ally' });
  });
});

describe('drafting', () => {
  it('toggles attackers and moves or takes back a block', () => {
    expect(toggleAttacker(toggleAttacker(EMPTY_DRAFT, 1), 1)).toEqual(EMPTY_DRAFT);
    const one = toggleBlock(EMPTY_DRAFT, 3, 1);
    expect(toggleBlock(one, 3, 2).blocks).toEqual([[3, 2]]);
    expect(toggleBlock(one, 3, 1).blocks).toEqual([]);
  });

  it('counts a pile of identical creatures into the attack', () => {
    const pile = [3, 4, 5];
    const draft = setPileCount({ attackers: [1], blocks: [] }, pile, 2);
    expect(draft.attackers).toEqual([1, 3, 4]);
    expect(setPileCount(draft, pile, 9).attackers).toEqual([1, 3, 4, 5]);
    const declarable = [creature(1, 'Tiger-Seal'), ...pile.map(id => creature(id, 'Ally'))];
    expect(pileCounts(declarable, draft, id => (pile.includes(id) ? 'ally' : `#${id}`))).toEqual([{ name: 'Ally', ids: pile, drafted: 2 }]);
  });

  it('sends the preview its own exact identities', () => {
    const preview = { declarable: [creature(otter, 'Otter-Penguin', [lieutenant])] } as CombatPreview;
    const plan = planOf({ attackers: [], blocks: [[otter, lieutenant]] }, preview);
    expect(plan.blocks).toEqual([{ blocker: { entity: otter, incarnation: 2 }, attacker: { entity: lieutenant, incarnation: 3 } }]);
    expect(planKey(plan)).toBe(planKey({ attackers: [], blocks: [...plan.blocks] }));
  });

  it('draws the draft on the board without touching the frame', () => {
    const drawn = drawDraft(block, { attackers: [], blocks: [[otter, lieutenant]] });
    expect(drawn.projection.agent.battlefield.find(p => p.id === otter)?.blocking).toBe(lieutenant);
    expect(block.projection.agent.battlefield.find(p => p.id === otter)?.blocking).toBeUndefined();
    expect(drawDraft(block, EMPTY_DRAFT)).toBe(block);
  });
});

describe('stepOffer', () => {
  it('finds the predicted offer and refuses any other decision', () => {
    expect(stepOffer(block, 'blockers', step(otter, false, lieutenant, 0))?.id).toBe(0);
    expect(stepOffer(block, 'blockers', step(otter, false, null, 1))?.id).toBe(1);
    // The offer at that index says something else, or asks about another creature, or another kind.
    expect(stepOffer(block, 'blockers', step(otter, false, null, 0))).toBeNull();
    expect(stepOffer(block, 'blockers', step(112, false, null, 1))).toBeNull();
    expect(stepOffer(block, 'attackers', step(otter, true, null, 0))).toBeNull();
  });
});

describe('a saved draft', () => {
  beforeEach(() => {
    const store = new Map<string, string>();
    globalThis.sessionStorage = {
      getItem: key => store.get(key) ?? null, setItem: (key, value) => void store.set(key, value),
    } as Storage;
  });

  it('comes back for the same decision only', () => {
    saveDraft(block, { attackers: [], blocks: [[otter, lieutenant]] });
    expect(loadDraft(block)?.blocks).toEqual([[otter, lieutenant]]);
    expect(loadDraft({ ...block, frame_hash: 'another' })).toBeNull();
  });
});
