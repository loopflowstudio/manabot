import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { groupPlanKey, type DeclareAttackersGroup, type DeclareBlockersGroup } from '$lib/group';
import type { ExperienceFrame } from '$lib/types';
import {
  addAttackers, combatPlanOf, committedFromGroup, drawDraft, draftWords, EMPTY_DRAFT, pileCounts,
  sanitizeCombatDraft, setBlock, setPileCount, toggleAttacker, toggleBlock,
} from './combat';

const fixture = JSON.parse(readFileSync(new URL('../../../../../managym/academy/fixtures/choice-navigation-w17-smart-payment.json', import.meta.url), 'utf8')) as {
  positions: Record<string, { frame: ExperienceFrame }>;
};
/** A real block: Otter-Penguin (110) may block Earth King's Lieutenant (105). */
const block = fixture.positions.DECLARE_BLOCKER.frame;
const [otter, lieutenant] = [110, 105];

const id = (entity: number, incarnation = 2) => ({ entity, incarnation });
const attackerItem = (entity: number, label: string, must_attack = false) => ({
  item: id(entity), label, must_attack,
});
const blockerItem = (entity: number, label: string, canBlock: number[] = [], must_block = false) => ({
  item: id(entity), label, can_block: canBlock.map(target => id(target, 3)), must_block,
});

const attackersGroup = (
  declarable: DeclareAttackersGroup['declarable'],
  committed: number[] = [],
): DeclareAttackersGroup => ({
  match_id: 'm', revision: 1, prompt_id: 1, frame_hash: 'h',
  kind: 'declare_attackers', declarable, committed: committed.map(entity => id(entity)),
});

const blockersGroup = (
  declarable: DeclareBlockersGroup['declarable'],
  committed: [number, number][] = [],
): DeclareBlockersGroup => ({
  match_id: 'm', revision: 1, prompt_id: 1, frame_hash: 'h',
  kind: 'declare_blockers', declarable,
  committed: committed.map(([blocker, attacker]) => ({ blocker: id(blocker), attacker: id(attacker, 3) })),
});

const names = new Map([
  [1, 'Tiger-Seal'], [2, 'Otter-Penguin'], [3, 'Ally'], [4, 'Ally'], [5, 'Ally'],
  [otter, 'Otter-Penguin'], [lieutenant, "Earth King's Lieutenant"],
]);

describe('draftWords', () => {
  const attackers = attackersGroup([
    attackerItem(1, 'Tiger-Seal'), attackerItem(2, 'Otter-Penguin'), attackerItem(3, 'Ally'), attackerItem(4, 'Ally'),
  ]);

  it('reads the whole attack, naming who stays home', () => {
    expect(draftWords(attackers, { attackers: [2, 3, 4], blocks: [] }, names))
      .toEqual({ text: 'Attack with 3', plan: 'Attack with 3 · not with Tiger-Seal' });
    expect(draftWords(attackers, { attackers: [2], blocks: [] }, names).plan)
      .toBe('Attack with Otter-Penguin · not with Tiger-Seal and Ally ×2');
    expect(draftWords(attackers, { attackers: [1, 2, 3, 4], blocks: [] }, names).plan).toBe('Attack with all 4');
    expect(draftWords(attackers, EMPTY_DRAFT, names).text).toBe('No attack');
  });

  it('reads blocks by the attacker they stop', () => {
    const words = draftWords(blockersGroup([]), { attackers: [], blocks: [[3, 1], [4, 1], [2, 5]] }, names);
    expect(words).toEqual({ text: 'Block with 3', plan: 'Ally ×2 on Tiger-Seal · Otter-Penguin on Ally' });
  });

  it('counts what the engine already committed without asking', () => {
    const juggernaut = new Map([...names, [9, 'Juggernaut']]);
    const withCommitted = attackersGroup(attackers.declarable, [9]);
    expect(draftWords(withCommitted, EMPTY_DRAFT, juggernaut).text).toBe('Attack with Juggernaut');
    expect(draftWords(withCommitted, { attackers: [1], blocks: [] }, juggernaut).text).toBe('Attack with 2');
    const forced = blockersGroup([], [[3, 1]]);
    expect(draftWords(forced, EMPTY_DRAFT, names)).toEqual({ text: 'Block with 1', plan: 'Ally on Tiger-Seal' });
    expect(draftWords(forced, { attackers: [], blocks: [[2, 5]] }, names).text).toBe('Block with 2');
    expect(committedFromGroup(forced)).toEqual({ attackers: [], blocks: [[3, 1]] });
  });
});

describe('drafting', () => {
  it('toggles attackers and moves or takes back a block', () => {
    expect(toggleAttacker(toggleAttacker(EMPTY_DRAFT, 1), 1)).toEqual(EMPTY_DRAFT);
    const one = toggleBlock(EMPTY_DRAFT, 3, 1);
    expect(toggleBlock(one, 3, 2).blocks).toEqual([[3, 2]]);
    expect(toggleBlock(one, 3, 1).blocks).toEqual([]);
  });

  it('adds newly swept attackers without toggling earlier selections', () => {
    const draft = addAttackers(EMPTY_DRAFT, new Set([3, 2]));
    expect(addAttackers(draft, new Set([2, 1])).attackers).toEqual([3, 2, 1]);
    expect(addAttackers(draft, new Set([2]))).toBe(draft);
  });

  it('dropping a blocker keeps its assignment or moves only that blocker', () => {
    const draft = setBlock(setBlock(EMPTY_DRAFT, 3, 1), 4, 1);
    expect(setBlock(draft, 3, 1)).toBe(draft);
    expect(setBlock(draft, 9, null)).toBe(draft);
    expect(setBlock(draft, 3, 2).blocks).toEqual([[4, 1], [3, 2]]);
  });

  it('counts a pile of identical creatures into the attack', () => {
    const pile = [3, 4, 5];
    const draft = setPileCount({ attackers: [1], blocks: [] }, pile, 2);
    expect(draft.attackers).toEqual([1, 3, 4]);
    expect(setPileCount(draft, pile, 9).attackers).toEqual([1, 3, 4, 5]);
    const declarable = [attackerItem(1, 'Tiger-Seal'), ...pile.map(entity => attackerItem(entity, 'Ally'))];
    expect(pileCounts(declarable, draft, entity => (pile.includes(entity) ? 'ally' : `#${entity}`)))
      .toEqual([{ name: 'Ally', ids: pile, drafted: 2 }]);
  });

  it('sends the group plan using the descriptor exact identities', () => {
    const group = blockersGroup([blockerItem(otter, 'Otter-Penguin', [lieutenant])]);
    const plan = combatPlanOf({ attackers: [], blocks: [[otter, lieutenant]] }, group);
    expect(plan).toEqual({
      kind: 'declare_blockers',
      blocks: [{ blocker: { entity: otter, incarnation: 2 }, attacker: { entity: lieutenant, incarnation: 3 } }],
    });
    if (plan.kind !== 'declare_blockers') throw new Error('expected declare_blockers plan');
    expect(groupPlanKey(plan)).toBe(groupPlanKey({ kind: 'declare_blockers', blocks: [...plan.blocks] }));
  });

  it('sanitizes restored drafts against declarable creatures', () => {
    const group = blockersGroup([blockerItem(otter, 'Otter-Penguin', [lieutenant])]);
    expect(sanitizeCombatDraft({ attackers: [999], blocks: [[otter, lieutenant], [otter, 1], [99, lieutenant]] }, group))
      .toEqual({ attackers: [], blocks: [[otter, lieutenant]] });
  });

  it('draws the draft on the board without touching the frame', () => {
    const drawn = drawDraft(block, { attackers: [], blocks: [[otter, lieutenant]] });
    expect(drawn.projection.agent.battlefield.find(p => p.id === otter)?.blocking).toBe(lieutenant);
    expect(block.projection.agent.battlefield.find(p => p.id === otter)?.blocking).toBeUndefined();
    expect(drawDraft(block, EMPTY_DRAFT)).toBe(block);
  });
});
