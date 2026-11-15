/** Local intent over an exact native declaration. Legality stays on the server. */
import type {
  DeclarationAnswer, DeclarationPlan, DeclarationProjection, ExperienceFrame,
  ObjectRenderId, SubjectRef,
} from './types';

export interface CombatDraft {
  key: string;
  assignments: DeclarationAnswer[];
  active: string | null;
}

export function objectKey(object: ObjectRenderId): string {
  return `${object.entity}:${object.incarnation}`;
}

export function destinationKey(destination: SubjectRef | null | undefined): string {
  if (!destination) return '';
  return destination.kind === 'object'
    ? `object:${objectKey(destination.id)}`
    : `${destination.kind}:${destination.id}`;
}

export function combatKey(frame: ExperienceFrame, update: number): string {
  return `${frame.match_id}:${frame.revision}:${frame.prompt?.id}:${frame.frame_hash}:${update}`;
}

export function newCombatDraft(projection: DeclarationProjection, key: string): CombatDraft {
  return {
    key, active: null,
    assignments: projection.subjects.flatMap(row => row.fixed || row.object.subject.kind !== 'object'
      ? [] : [{ subject: { ...row.object.subject.id }, destination: null }]),
  };
}

export function setAssignment(draft: CombatDraft, subject: ObjectRenderId, destination: SubjectRef | null): CombatDraft {
  const key = objectKey(subject);
  return { ...draft, assignments: draft.assignments.map(answer => objectKey(answer.subject) === key
    ? { subject: answer.subject, destination } : answer) };
}

export function clearCombatDraft(draft: CombatDraft): CombatDraft {
  return { ...draft, active: null, assignments: draft.assignments.map(answer => ({ ...answer, destination: null })) };
}

/** Bulk editing is explicit only when there is one shared native destination. */
export function bulkAttackDestination(projection: DeclarationProjection): SubjectRef | null {
  if (projection.stage !== 'attackers') return null;
  const rows = projection.subjects.filter(row => !row.fixed);
  const destination = rows[0]?.destinations[0];
  return destination && rows.every(row => row.destinations.length === 1
    && destinationKey(row.destinations[0]) === destinationKey(destination)) ? destination : null;
}

export function selectAllAttackers(draft: CombatDraft, projection: DeclarationProjection): CombatDraft {
  const destination = bulkAttackDestination(projection);
  return destination ? { ...draft, assignments: draft.assignments.map(answer => ({ ...answer, destination })) } : draft;
}

export function declarationPlan(draft: CombatDraft, projection: DeclarationProjection): DeclarationPlan {
  return { binding: { ...projection.binding }, assignments: draft.assignments };
}

export function assignmentFor(draft: CombatDraft, subject: ObjectRenderId): SubjectRef | null {
  return draft.assignments.find(answer => objectKey(answer.subject) === objectKey(subject))?.destination ?? null;
}

/** Choice focus is navigation, separate from both events and the full draft. */
export function combatFocus(draft: CombatDraft | null, projection: DeclarationProjection | null): Set<number> {
  const focus = new Set<number>();
  const active = projection?.subjects.find(row => row.object.subject.kind === 'object'
    && objectKey(row.object.subject.id) === draft?.active);
  if (!active || !draft) return focus;
  focus.add(active.object.display_id);
  if (active.object.subject.kind === 'object') {
    const destination = assignmentFor(draft, active.object.subject.id);
    if (destination?.kind === 'object') focus.add(destination.id.entity);
  }
  return focus;
}

/** A board click selects a subject, or explicitly assigns its native destination. */
export function selectCombatObject(draft: CombatDraft, projection: DeclarationProjection, displayId: number): CombatDraft {
  const row = projection.subjects.find(row => !row.fixed && row.object.display_id === displayId);
  if (row?.object.subject.kind === 'object') return { ...draft, active: objectKey(row.object.subject.id) };
  const active = projection.subjects.find(row => row.object.subject.kind === 'object'
    && objectKey(row.object.subject.id) === draft.active && !row.fixed);
  const destination = projection.destinations.find(row => row.display_id === displayId)?.subject;
  if (active?.object.subject.kind === 'object'
    && destination && active.destinations.some(candidate => destinationKey(candidate) === destinationKey(destination))) {
    return setAssignment(draft, active.object.subject.id, destination);
  }
  return draft;
}

/** Mark all editable assignments without moving choice focus or lane navigation. */
export function combatPlanMarkers(draft: CombatDraft | null, projection: DeclarationProjection | null): Map<number, string> {
  const markers = new Map<number, string>();
  if (!draft || !projection) return markers;
  for (const row of projection.subjects) {
    if (!row.fixed && row.object.subject.kind === 'object' && assignmentFor(draft, row.object.subject.id)) {
      markers.set(row.object.display_id, projection.stage === 'attackers' ? 'Planned attack' : 'Planned block');
    }
  }
  return markers;
}
