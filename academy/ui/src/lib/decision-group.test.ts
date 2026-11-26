import { beforeEach, describe, expect, it } from 'vitest';

import {
  decisionGroupDisclosureCopy,
  decisionGroupPlanKey,
  loadDecisionGroupDraft,
  saveDecisionGroupDraft,
  validDecisionGroupPlan,
} from './decision-group';

describe('decisionGroupPlanKey', () => {
  it('treats unordered plans as order-insensitive and ordered plans as order-sensitive', () => {
    const a = { entity: 10, incarnation: 1 };
    const b = { entity: 20, incarnation: 1 };
    expect(decisionGroupPlanKey({ kind: 'declare_attackers', attackers: [a, b] }))
      .toBe(decisionGroupPlanKey({ kind: 'declare_attackers', attackers: [b, a] }));
    expect(decisionGroupPlanKey({ kind: 'surveil', graveyard: [a, b] }))
      .toBe(decisionGroupPlanKey({ kind: 'surveil', graveyard: [b, a] }));
    expect(decisionGroupPlanKey({ kind: 'partition_piles', face_up: [a, b] }))
      .toBe(decisionGroupPlanKey({ kind: 'partition_piles', face_up: [b, a] }));
    expect(decisionGroupPlanKey({ kind: 'scry', top: [a, b], bottom: [] }))
      .not.toBe(decisionGroupPlanKey({ kind: 'scry', top: [b, a], bottom: [] }));
    expect(decisionGroupPlanKey({ kind: 'order_library', order: [a, b] }))
      .not.toBe(decisionGroupPlanKey({ kind: 'order_library', order: [b, a] }));
  });
});

describe('validDecisionGroupPlan', () => {
  it('validates all six plan kinds and rejects malformed values', () => {
    const id = { entity: 5, incarnation: 0 };
    expect(validDecisionGroupPlan({ kind: 'declare_attackers', attackers: [id] })).toBe(true);
    expect(validDecisionGroupPlan({ kind: 'declare_blockers', blocks: [{ blocker: id, attacker: id }] })).toBe(true);
    expect(validDecisionGroupPlan({ kind: 'scry', top: [id], bottom: [] })).toBe(true);
    expect(validDecisionGroupPlan({ kind: 'surveil', graveyard: [] })).toBe(true);
    expect(validDecisionGroupPlan({ kind: 'order_library', order: [id] })).toBe(true);
    expect(validDecisionGroupPlan({ kind: 'partition_piles', face_up: [id] })).toBe(true);
    expect(validDecisionGroupPlan({ kind: 'partition_piles', face_up: ['bad'] })).toBe(false);
    expect(validDecisionGroupPlan({ kind: 'unknown', items: [] })).toBe(false);
    expect(validDecisionGroupPlan({ kind: 'scry', top: ['bad'], bottom: [] })).toBe(false);
  });
});

describe('unified group draft persistence', () => {
  beforeEach(() => {
    const store = new Map<string, string>();
    globalThis.sessionStorage = {
      getItem: key => store.get(key) ?? null,
      setItem: (key, value) => void store.set(key, value),
    } as Storage;
  });

  it('restores a draft only for the exact match, frame hash and kind', () => {
    const scope = { match_id: 'm1', frame_hash: 'h1', kind: 'scry' as const };
    saveDecisionGroupDraft(scope, { kind: 'scry', top: [1], bottom: [2] });
    expect(loadDecisionGroupDraft(scope)).toEqual({ kind: 'scry', top: [1], bottom: [2] });
    expect(loadDecisionGroupDraft({ ...scope, frame_hash: 'h2' })).toBeNull();
    expect(loadDecisionGroupDraft({ ...scope, kind: 'surveil' })).toBeNull();
    expect(loadDecisionGroupDraft({ ...scope, match_id: 'm2' })).toBeNull();
  });

  it('names disclosure copy per group kind', () => {
    expect(decisionGroupDisclosureCopy('declare_attackers').summary).toBe("Each creature's choice");
    expect(decisionGroupDisclosureCopy('scry').summary).toBe("Each card's choice");
    expect(decisionGroupDisclosureCopy('partition_piles').summary).toBe("Each card's choice");
    expect(decisionGroupDisclosureCopy('partition_piles').caption).toContain('split');
  });
});
