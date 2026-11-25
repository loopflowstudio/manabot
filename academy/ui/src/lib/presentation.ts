import type {
  ExperienceFrame,
  ObjectRenderId,
  Observation,
  PlayerState,
  PresentationEvent,
  PresentationLabels,
  SubjectRef,
} from './types';
import { PRESENTATION_RULES } from './types';

export type {
  ObjectRenderId,
  PresentationEvent,
  PresentationImportance,
  PresentationKind,
  PresentationLabels,
  SubjectRef,
} from './types';

export interface PresentationSequence {
  id: string;
  title: string;
  labels: PresentationLabels;
  events: PresentationEvent[];
}

export function mergePresentationLabels(
  ...sources: readonly PresentationLabels[]
): PresentationLabels {
  const merged: PresentationLabels = { objects: {}, players: {}, stacks: {} };
  for (const source of sources) {
    Object.assign(merged.objects, source.objects);
    Object.assign(merged.players, source.players);
    Object.assign(merged.stacks, source.stacks);
  }
  return merged;
}

export function objectLabelKey(id: ObjectRenderId): string {
  return `${id.entity}:${id.incarnation}`;
}

function addPlayerLabels(
  labels: PresentationLabels,
  player: PlayerState,
  playerLabel: string,
): void {
  // Prompt actors currently use player indexes while projections also expose
  // engine IDs. Accept either viewer-safe reference during this bridge.
  labels.players[String(player.player_index)] = playerLabel;
  labels.players[String(player.id)] = playerLabel;

  for (const card of [
    ...player.hand,
    ...player.graveyard,
    ...player.exile,
    ...player.stack,
  ]) {
    // Protocol-v1's legacy Observation does not expose incarnation yet. This
    // indexes only incarnation zero; nonzero authoritative refs intentionally
    // remain unnamed rather than being guessed by the client.
    labels.objects[objectLabelKey({ entity: card.id, incarnation: 0 })] = card.name;
  }
  for (const card of player.stack) {
    // The legacy stack has no separate StackRenderId. This is a display-only
    // compatibility lookup for an authority that reuses the visible card ID.
    labels.stacks[String(card.id)] = card.name;
  }
  for (const permanent of player.battlefield) {
    if (permanent.name) {
      labels.objects[objectLabelKey({ entity: permanent.id, incarnation: 0 })] = permanent.name;
    }
  }
}

export function presentationLabelsFromObservation(
  observation: Observation,
): PresentationLabels {
  const labels: PresentationLabels = { objects: {}, players: {}, stacks: {} };
  addPlayerLabels(labels, observation.agent, 'Hero');
  addPlayerLabels(labels, observation.opponent, 'Opponent');
  return labels;
}

export function presentationLabelsFromFrame(frame: ExperienceFrame): PresentationLabels {
  return presentationLabelsFromObservation(frame.projection);
}

function eventCopy(event: PresentationEvent): PresentationEvent {
  return structuredClone(event);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function requireInteger(value: unknown, field: string, seq: number): number {
  if (!Number.isSafeInteger(value)) {
    throw new Error(`Presentation event ${seq} has invalid ${field}`);
  }
  return value as number;
}

function validateObjectRenderId(value: unknown, field: string, seq: number): void {
  if (!isRecord(value)) {
    throw new Error(`Presentation event ${seq} has invalid ${field}`);
  }
  requireInteger(value.entity, `${field}.entity`, seq);
  requireInteger(value.incarnation, `${field}.incarnation`, seq);
}

function validateSubjectRef(value: unknown, field: string, seq: number): void {
  if (!isRecord(value) || typeof value.kind !== 'string') {
    throw new Error(`Presentation event ${seq} has invalid ${field}`);
  }
  if (value.kind === 'object') {
    validateObjectRenderId(value.id, `${field}.id`, seq);
    return;
  }
  if (value.kind === 'player' || value.kind === 'stack') {
    requireInteger(value.id, `${field}.id`, seq);
    return;
  }
  throw new Error(`Presentation event ${seq} has unsupported ${field}.kind`);
}

function validatePresentationKind(value: unknown, seq: number): void {
  if (!isRecord(value) || typeof value.kind !== 'string') {
    throw new Error(`Presentation event ${seq} has invalid kind`);
  }
  switch (value.kind) {
    case 'cast':
      validateObjectRenderId(value.object, 'kind.object', seq);
      requireInteger(value.controller, 'kind.controller', seq);
      requireInteger(value.stack, 'kind.stack', seq);
      return;
    case 'targeted':
      validateSubjectRef(value.source, 'kind.source', seq);
      validateSubjectRef(value.target, 'kind.target', seq);
      return;
    case 'resolved':
      requireInteger(value.stack, 'kind.stack', seq);
      return;
    case 'damage':
      if (value.source !== null) {
        validateSubjectRef(value.source, 'kind.source', seq);
      }
      validateSubjectRef(value.target, 'kind.target', seq);
      requireInteger(value.amount, 'kind.amount', seq);
      return;
    case 'destroyed':
    case 'died':
      if (!Array.isArray(value.objects)) {
        throw new Error(`Presentation event ${seq} has invalid kind.objects`);
      }
      for (const object of value.objects) {
        validateObjectRenderId(object, 'kind.objects[]', seq);
      }
      return;
    case 'attack_group':
      if (!Array.isArray(value.attackers)) {
        throw new Error(`Presentation event ${seq} has invalid kind.attackers`);
      }
      for (const attacker of value.attackers) {
        validateObjectRenderId(attacker, 'kind.attackers[]', seq);
      }
      validateSubjectRef(value.defender, 'kind.defender', seq);
      return;
    case 'blocked':
      validateObjectRenderId(value.attacker, 'kind.attacker', seq);
      if (!Array.isArray(value.blockers)) {
        throw new Error(`Presentation event ${seq} has invalid kind.blockers`);
      }
      for (const blocker of value.blockers) {
        validateObjectRenderId(blocker, 'kind.blockers[]', seq);
      }
      return;
    case 'turn_started':
      requireInteger(value.player, 'kind.player', seq);
      requireInteger(value.turn_number, 'kind.turn_number', seq);
      return;
    case 'land_played':
      validateObjectRenderId(value.object, 'kind.object', seq);
      requireInteger(value.player, 'kind.player', seq);
      return;
    case 'drew':
      requireInteger(value.player, 'kind.player', seq);
      if (value.object !== undefined) validateObjectRenderId(value.object, 'kind.object', seq);
      return;
    case 'discarded':
      if (!Array.isArray(value.objects)) {
        throw new Error(`Presentation event ${seq} has invalid kind.objects`);
      }
      for (const object of value.objects) {
        validateObjectRenderId(object, 'kind.objects[]', seq);
      }
      requireInteger(value.player, 'kind.player', seq);
      return;
    case 'revealed':
      if (!Array.isArray(value.cards) || !value.cards.every(card => typeof card === 'string')) {
        throw new Error(`Presentation event ${seq} has invalid kind.cards`);
      }
      requireInteger(value.player, 'kind.player', seq);
      return;
    case 'activated':
    case 'triggered':
      validateObjectRenderId(value.source, 'kind.source', seq);
      requireInteger(value.controller, 'kind.controller', seq);
      requireInteger(value.stack, 'kind.stack', seq);
      return;
    case 'ability_resolved':
      validateObjectRenderId(value.source, 'kind.source', seq);
      requireInteger(value.stack, 'kind.stack', seq);
      return;
    case 'countered':
      requireInteger(value.stack, 'kind.stack', seq);
      return;
    case 'passed':
      requireInteger(value.player, 'kind.player', seq);
      return;
    case 'counters':
      validateObjectRenderId(value.object, 'kind.object', seq);
      if (typeof value.counter !== 'string') {
        throw new Error(`Presentation event ${seq} has invalid kind.counter`);
      }
      requireInteger(value.count, 'kind.count', seq);
      requireInteger(value.total, 'kind.total', seq);
      return;
    default:
      throw new Error(`Presentation event ${seq} has unsupported kind ${value.kind}`);
  }
}

export function validatePresentationEvents(events: readonly PresentationEvent[]): void {
  let previousSeq = -1;
  for (const event of events) {
    if (!Number.isSafeInteger(event.seq) || event.seq < 0) {
      throw new Error(`Presentation event sequence must be a non-negative integer: ${event.seq}`);
    }
    if (event.seq <= previousSeq) {
      throw new Error(
        `Presentation events must be strictly ordered; received ${event.seq} after ${previousSeq}`,
      );
    }
    if (event.to_revision < event.from_revision) {
      throw new Error(
        `Presentation event ${event.seq} moves revision backwards (${event.from_revision} -> ${event.to_revision})`,
      );
    }
    if (!Number.isFinite(event.suggested_ms) || event.suggested_ms < 0) {
      throw new Error(`Presentation event ${event.seq} has invalid suggested_ms`);
    }
    validatePresentationKind(event.kind, event.seq);
    if (
      event.cause !== undefined &&
      !(Number.isSafeInteger(event.cause) && event.cause >= 0 && event.cause < event.seq)
    ) {
      throw new Error(`Presentation event ${event.seq} has invalid cause`);
    }
    if (event.rule !== undefined && !(PRESENTATION_RULES as readonly string[]).includes(event.rule)) {
      throw new Error(`Presentation event ${event.seq} has invalid rule`);
    }
    if (event.labels !== undefined && (!isRecord(event.labels) || Object.entries(event.labels).some(
      ([table, names]) => !['objects', 'stacks'].includes(table) || !isRecord(names)
        || Object.values(names).some(name => typeof name !== 'string'),
    ))) {
      throw new Error(`Presentation event ${event.seq} has invalid labels`);
    }
    previousSeq = event.seq;
  }
}

/**
 * Validate an authority-addressed recovery/live tail and return its next cursor.
 *
 * The cursor names the first event in the array. An empty array therefore
 * keeps the cursor unchanged. Clients reject gaps instead of inventing or
 * reordering semantic events locally.
 */
export function validatePresentationTail(
  events: readonly PresentationEvent[],
  cursor: number,
): number {
  if (!Number.isSafeInteger(cursor) || cursor < 0) {
    throw new Error(`Presentation cursor must be a non-negative safe integer: ${cursor}`);
  }
  validatePresentationEvents(events);
  for (let offset = 0; offset < events.length; offset += 1) {
    const expected = cursor + offset;
    if (events[offset].seq !== expected) {
      throw new Error(
        `Presentation cursor gap: expected event ${expected}, received ${events[offset].seq}`,
      );
    }
  }
  return cursor + events.length;
}

export function validatePresentationUpdate(
  events: readonly PresentationEvent[],
  fromRevision: number,
  toRevision: number,
): void {
  validatePresentationEvents(events);
  let previousSeq: number | null = null;
  let previousFrom = fromRevision;
  let previousTo = fromRevision;
  for (const event of events) {
    if (
      event.from_revision < fromRevision
      || event.to_revision > toRevision
      || event.from_revision >= event.to_revision
      || event.from_revision < previousFrom
      || event.to_revision < previousTo
    ) {
      throw new Error(
        `Presentation event ${event.seq} spans invalid sub-transition ${event.from_revision} -> ${event.to_revision} inside ${fromRevision} -> ${toRevision}`,
      );
    }
    if (previousSeq !== null && event.seq !== previousSeq + 1) {
      throw new Error(
        `Presentation update sequence gap: expected ${previousSeq + 1}, received ${event.seq}`,
      );
    }
    previousSeq = event.seq;
    previousFrom = event.from_revision;
    previousTo = event.to_revision;
  }
}

export function copyPresentationEvents(
  events: readonly PresentationEvent[],
): PresentationEvent[] {
  validatePresentationEvents(events);
  return events.map(eventCopy);
}

/** Exact subjects mentioned by an authoritative event, never inferred causes. */
export function presentationSubjects(event: PresentationEvent): SubjectRef[] {
  const object = (id: ObjectRenderId): SubjectRef => ({ kind: 'object', id });
  const kind = event.kind;
  switch (kind.kind) {
    case 'cast': return [object(kind.object), { kind: 'player', id: kind.controller }, { kind: 'stack', id: kind.stack }];
    case 'targeted': return [kind.source, kind.target];
    case 'resolved': return [{ kind: 'stack', id: kind.stack }];
    case 'damage': return [...(kind.source ? [kind.source] : []), kind.target];
    case 'died': case 'destroyed': return kind.objects.map(object);
    case 'attack_group': return [...kind.attackers.map(object), kind.defender];
    case 'blocked': return [object(kind.attacker), ...kind.blockers.map(object)];
    case 'turn_started': return [{ kind: 'player', id: kind.player }];
    case 'land_played': return [object(kind.object), { kind: 'player', id: kind.player }];
    case 'drew': return [...(kind.object ? [object(kind.object)] : []), { kind: 'player', id: kind.player }];
    case 'discarded': return [...kind.objects.map(object), { kind: 'player', id: kind.player }];
    case 'revealed': return [{ kind: 'player', id: kind.player }];
    case 'activated': case 'triggered': return [object(kind.source), { kind: 'player', id: kind.controller }, { kind: 'stack', id: kind.stack }];
    case 'ability_resolved': return [object(kind.source), { kind: 'stack', id: kind.stack }];
    case 'countered': return [{ kind: 'stack', id: kind.stack }];
    case 'passed': return [{ kind: 'player', id: kind.player }];
    case 'counters': return [object(kind.object)];
  }
}

/** Retain only the viewer-safe names this event actually needs. */
export function presentationEventLabels(event: PresentationEvent, labels: PresentationLabels): PresentationLabels {
  const retained: PresentationLabels = { objects: {}, players: {}, stacks: {} };
  for (const subject of presentationSubjects(event)) {
    const table = subject.kind === 'object' ? 'objects' : subject.kind === 'player' ? 'players' : 'stacks';
    const key = subject.kind === 'object' ? objectLabelKey(subject.id) : String(subject.id);
    const value = event.labels && table !== 'players' ? event.labels[table]?.[key] : labels[table][key];
    if (value !== undefined) retained[table][key] = value;
  }
  return retained;
}

/** Read-only board context; narration never changes choice focus or lane paging. */
export const PRESENTATION_CONTEXT = Symbol('presentation');
