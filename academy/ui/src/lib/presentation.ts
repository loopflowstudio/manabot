import type {
  ExperienceFrame,
  ObjectRenderId,
  Observation,
  PlayerState,
  PresentationEvent,
  PresentationImportance,
  PresentationLabels,
  SubjectRef,
} from './types';

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

export interface PresentationBeat {
  seq: number;
  group: number;
  fromRevision: number;
  toRevision: number;
  importance: PresentationImportance;
  heading: string;
  detail: string;
  ariaLabel: string;
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

function objectLabel(id: ObjectRenderId, labels: PresentationLabels): string {
  return labels.objects[objectLabelKey(id)] ?? `object ${objectLabelKey(id)}`;
}

function subjectLabel(subject: SubjectRef, labels: PresentationLabels): string {
  if (subject.kind === 'object') {
    return objectLabel(subject.id, labels);
  }
  if (subject.kind === 'player') {
    return labels.players[String(subject.id)] ?? `player ${subject.id}`;
  }
  return labels.stacks[String(subject.id)] ?? `stack object ${subject.id}`;
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

export function presentationBeat(
  event: PresentationEvent,
  labels: PresentationLabels,
): PresentationBeat {
  let heading: string;
  let detail: string;

  switch (event.kind.kind) {
    case 'cast': {
      const spell = objectLabel(event.kind.object, labels);
      const controller = labels.players[String(event.kind.controller)] ?? `Player ${event.kind.controller}`;
      heading = 'Spell cast';
      detail = `${controller} casts ${spell}.`;
      break;
    }
    case 'targeted': {
      heading = 'Target locked';
      detail = `${subjectLabel(event.kind.source, labels)} targets ${subjectLabel(event.kind.target, labels)}.`;
      break;
    }
    case 'resolved': {
      heading = 'Resolving';
      detail = `${labels.stacks[String(event.kind.stack)] ?? `Stack object ${event.kind.stack}`} resolves.`;
      break;
    }
    case 'damage': {
      const source = event.kind.source
        ? subjectLabel(event.kind.source, labels)
        : 'An effect';
      heading = 'Damage';
      detail = `${source} deals ${event.kind.amount} damage to ${subjectLabel(event.kind.target, labels)}.`;
      break;
    }
    case 'destroyed':
    case 'died': {
      const objects = event.kind.objects.map((object) => objectLabel(object, labels));
      heading = objects.length === 1 ? 'Creature dies' : 'Creatures die';
      detail = `${objects.join(', ')} ${objects.length === 1 ? 'dies' : 'die'}.`;
      break;
    }
    case 'attack_group': {
      const attackers = event.kind.attackers.map((object) => objectLabel(object, labels));
      heading = 'Attackers declared';
      detail = `${attackers.join(', ')} ${attackers.length === 1 ? 'attacks' : 'attack'} ${subjectLabel(event.kind.defender, labels)}.`;
      break;
    }
    case 'blocked': {
      const attacker = objectLabel(event.kind.attacker, labels);
      const blockers = event.kind.blockers.map((object) => objectLabel(object, labels));
      heading = 'Blockers declared';
      detail = `${blockers.join(', ')} ${blockers.length === 1 ? 'blocks' : 'block'} ${attacker}.`;
      break;
    }
    case 'turn_started': {
      const player = labels.players[String(event.kind.player)] ?? `Player ${event.kind.player}`;
      heading = `Turn ${event.kind.turn_number}`;
      detail = `${player}'s turn begins.`;
      break;
    }
  }

  return {
    seq: event.seq,
    group: event.group,
    fromRevision: event.from_revision,
    toRevision: event.to_revision,
    importance: event.importance,
    heading,
    detail,
    ariaLabel: `${heading}. ${detail}`,
  };
}
