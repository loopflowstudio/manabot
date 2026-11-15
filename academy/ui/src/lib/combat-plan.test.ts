import { describe, expect, it } from 'vitest';
import fixture from '../../../../protocol/fixtures/combat-declaration.json';
import type { ExperienceFrame } from './types';
import { assignmentFor, clearCombatDraft, combatKey, declarationPlan, newCombatDraft, objectKey,
  selectAllAttackers, selectCombatObject, setAssignment } from './combat-plan';

const frame = fixture.recovery.frame as ExperienceFrame;
const projection = frame.declaration!;

describe('combat intent over native subjects', () => {
  it('starts with explicit declines and only edits on deliberate input', () => {
    const draft = newCombatDraft(projection, combatKey(frame, 1));
    expect(draft.assignments).toHaveLength(3);
    expect(draft.assignments.every(answer => answer.destination === null)).toBe(true);
    const all = selectAllAttackers(draft, projection);
    expect(all.assignments.every(answer => answer.destination !== null)).toBe(true);
    expect(draft.assignments.every(answer => answer.destination === null)).toBe(true);
    expect(clearCombatDraft(all).assignments).toEqual(draft.assignments);
    expect(declarationPlan(all, projection).binding).toEqual(projection.binding);
  });
  it('keeps same-name incarnations distinct and fixed choices outside clear', () => {
    const current = structuredClone(projection);
    current.subjects[0].fixed = true;
    current.subjects[0].assigned_to = current.subjects[0].destinations[0];
    const draft = selectAllAttackers(newCombatDraft(current, 'prefix'), current);
    expect(draft.assignments).toHaveLength(2);
    expect(clearCombatDraft(draft).assignments).toHaveLength(2);
    expect(current.subjects[0].assigned_to).toEqual(current.subjects[0].destinations[0]);
    expect(new Set(projection.subjects.map(row => row.object.subject.kind === 'object' ? objectKey(row.object.subject.id) : '')).size).toBe(3);
  });
  it('board navigation never assigns until a supported destination is selected', () => {
    const row = projection.subjects[0];
    if (row.object.subject.kind !== 'object') throw new Error('fixture subject');
    const empty = newCombatDraft(projection, 'root');
    const active = selectCombatObject(empty, projection, row.object.display_id);
    expect(assignmentFor(active, row.object.subject.id)).toBeNull();
    const assigned = selectCombatObject(active, projection, projection.destinations[0].display_id);
    expect(assignmentFor(assigned, row.object.subject.id)).toEqual(row.destinations[0]);
    expect(selectCombatObject(active, projection, -999)).toBe(active);
    expect(setAssignment(assigned, row.object.subject.id, null).assignments).toEqual(empty.assignments);
  });
  it('recovery of even the identical frame creates a fresh draft scope', () => {
    expect(combatKey(frame, 1)).not.toBe(combatKey(frame, 2));
    expect(combatKey(frame, 1)).not.toBe(combatKey({ ...frame, revision: frame.revision + 1 }, 1));
  });
});
