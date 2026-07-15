import type {
  ExperienceFrame,
  ObjectRenderId,
  Observation,
  PlayerState,
  PresentationEvent,
  PresentationImportance,
  SubjectRef,
} from './types';

export type {
  ObjectRenderId,
  PresentationEvent,
  PresentationImportance,
  PresentationKind,
  SubjectRef,
} from './types';

/** Display names are a viewer-safe projection sidecar, never rules facts. */
export interface PresentationLabels {
  objects: Record<string, string>;
  players: Record<string, string>;
  stacks: Record<string, string>;
}

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

export interface PresentationInspectorRow extends PresentationBeat {
  causedBy: string | null;
  event: PresentationEvent;
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

export function validatePresentationUpdate(
  events: readonly PresentationEvent[],
  fromRevision: number,
  toRevision: number,
): void {
  validatePresentationEvents(events);
  for (const event of events) {
    if (
      event.from_revision !== fromRevision
      || event.to_revision !== toRevision
    ) {
      throw new Error(
        `Presentation event ${event.seq} spans ${event.from_revision} -> ${event.to_revision}, expected ${fromRevision} -> ${toRevision}`,
      );
    }
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

/**
 * The decision inspector consumes the same events and semantic projection as
 * the table. It may add policy/search data beside these rows, but it must not
 * reconstruct a second narration from snapshots.
 */
export function presentationInspectorRows(
  events: readonly PresentationEvent[],
  labels: PresentationLabels,
): PresentationInspectorRow[] {
  validatePresentationEvents(events);
  return events.map((event) => ({
    ...presentationBeat(event, labels),
    causedBy: event.caused_by,
    event: eventCopy(event),
  }));
}

const bolt = { entity: 31, incarnation: 0 } satisfies ObjectRenderId;
const ally = { entity: 77, incarnation: 0 } satisfies ObjectRenderId;
const boltOnStack = { kind: 'stack', id: 4001 } satisfies SubjectRef;
const allyTarget = { kind: 'object', id: ally } satisfies SubjectRef;

/**
 * Recorded vertical-slice fixture: one authoritative update, five semantic
 * beats. Both live and replay players consume this exact array in tests.
 */
export const LIGHTNING_BOLT_PRESENTATION: PresentationSequence = {
  id: 'bolt-kills-ally-v1',
  title: 'An Ally token dies after Lightning Bolt deals lethal damage',
  labels: {
    objects: {
      [objectLabelKey(bolt)]: 'Lightning Bolt',
      [objectLabelKey(ally)]: 'Ally token',
    },
    players: {
      '0': 'Hero',
      '1': 'Opponent',
    },
    stacks: {
      '4001': 'Lightning Bolt',
    },
  },
  events: [
    {
      seq: 900,
      from_revision: 42,
      to_revision: 43,
      caused_by: 'command-bolt-42',
      group: 700,
      importance: 'emphasized',
      suggested_ms: 650,
      sound: 'spell.cast',
      kind: { kind: 'cast', object: bolt, controller: 0, stack: 4001 },
    },
    {
      seq: 901,
      from_revision: 42,
      to_revision: 43,
      caused_by: 'command-bolt-42',
      group: 701,
      importance: 'normal',
      suggested_ms: 500,
      sound: null,
      kind: { kind: 'targeted', source: boltOnStack, target: allyTarget },
    },
    {
      seq: 902,
      from_revision: 42,
      to_revision: 43,
      caused_by: 'command-bolt-42',
      group: 702,
      importance: 'emphasized',
      suggested_ms: 450,
      sound: 'spell.resolve',
      kind: { kind: 'resolved', stack: 4001 },
    },
    {
      seq: 903,
      from_revision: 42,
      to_revision: 43,
      caused_by: 'command-bolt-42',
      group: 703,
      importance: 'critical',
      suggested_ms: 700,
      sound: 'damage.fire',
      kind: { kind: 'damage', source: boltOnStack, target: allyTarget, amount: 3 },
    },
    {
      seq: 904,
      from_revision: 42,
      to_revision: 43,
      caused_by: 'command-bolt-42',
      group: 704,
      importance: 'critical',
      suggested_ms: 650,
      sound: 'creature.dies',
      kind: { kind: 'died', objects: [ally] },
    },
  ],
};
