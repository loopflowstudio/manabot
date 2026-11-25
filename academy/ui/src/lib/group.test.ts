import { beforeEach, describe, expect, it } from 'vitest';

import {
  groupDisclosureCopy,
  groupPlanKey,
  loadGroupDraft,
  saveGroupDraft,
  validGroupPlan,
} from './group';

describe('groupPlanKey', () => {
  it('treats unordered plans as order-insensitive and ordered plans as order-sensitive', () => {
    const a = { entity: 10, incarnation: 1 };
    const b = { entity: 20, incarnation: 1 };
    expect(groupPlanKey({ kind: 'declare_attackers', attackers: [a, b] }))
      .toBe(groupPlanKey({ kind: 'declare_attackers', attackers: [b, a] }));
    expect(groupPlanKey({ kind: 'surveil', graveyard: [a, b] }))
      .toBe(groupPlanKey({ kind: 'surveil', graveyard: [b, a] }));
    expect(groupPlanKey({ kind: 'scry', top: [a, b], bottom: [] }))
      .not.toBe(groupPlanKey({ kind: 'scry', top: [b, a], bottom: [] }));
    expect(groupPlanKey({ kind: 'order_library', order: [a, b] }))
      .not.toBe(groupPlanKey({ kind: 'order_library', order: [b, a] }));
  });
});

describe('validGroupPlan', () => {
  it('validates all five plan kinds and rejects malformed values', () => {
    const id = { entity: 5, incarnation: 0 };
    expect(validGroupPlan({ kind: 'declare_attackers', attackers: [id] })).toBe(true);
    expect(validGroupPlan({ kind: 'declare_blockers', blocks: [{ blocker: id, attacker: id }] })).toBe(true);
    expect(validGroupPlan({ kind: 'scry', top: [id], bottom: [] })).toBe(true);
    expect(validGroupPlan({ kind: 'surveil', graveyard: [] })).toBe(true);
    expect(validGroupPlan({ kind: 'order_library', order: [id] })).toBe(true);
    expect(validGroupPlan({ kind: 'unknown', items: [] })).toBe(false);
    expect(validGroupPlan({ kind: 'scry', top: ['bad'], bottom: [] })).toBe(false);
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
    saveGroupDraft(scope, { kind: 'scry', top: [1], bottom: [2] });
    expect(loadGroupDraft(scope)).toEqual({ kind: 'scry', top: [1], bottom: [2] });
    expect(loadGroupDraft({ ...scope, frame_hash: 'h2' })).toBeNull();
    expect(loadGroupDraft({ ...scope, kind: 'surveil' })).toBeNull();
    expect(loadGroupDraft({ ...scope, match_id: 'm2' })).toBeNull();
  });

  it('names disclosure copy per group kind', () => {
    expect(groupDisclosureCopy('declare_attackers').summary).toBe("Each creature's choice");
    expect(groupDisclosureCopy('scry').summary).toBe("Each card's choice");
  });
});
