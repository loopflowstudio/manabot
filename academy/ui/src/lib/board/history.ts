/**
 * One history view over the viewer's presentation track (plus client-local
 * automation notes such as a spent browser-only "once" stop).
 *
 * `buildHistory` keeps events strictly in authority order, folds targets into
 * their cast/activation/trigger, folds an immediate resolution (ignoring
 * passes) into its cast or activation, indents consequences under the nearest
 * open block that caused them, folds opponent/automation priority passes into
 * quiet runs, and marks turns with more than four consequences as busy so
 * their moves collapse by default.
 *
 * `legacyMoveHistory` converts flat `GameLogEntry[]` from recordings written
 * before moves were recorded as presentation events into the same `HistoryTurn`
 * shape so the rail renders one structure everywhere.
 */
import { presentationSubjects } from '$lib/presentation';
import type {
  GameLogEntry,
  PresentationEvent,
  PresentationLabels,
  PresentationRule,
} from '$lib/types';
import type { StripMark } from './replay';
import {
  playerSide,
  segmentText,
  segmentsFromText,
  tell,
  type FoldedTarget,
  type Segment,
  type TellPlayers,
} from './tell';
import { possessive } from './view';

export type HistoryActor = 'you' | 'opponent' | 'auto' | 'note';

export interface PassCount {
  actor: 'opponent' | 'auto';
  who: string;
  count: number;
}

/** A client-local automation, yield, or divider note anchored to a point on the track. */
export interface HistoryNote {
  id: string;
  actor: 'you' | 'auto' | 'note';
  turn: number | null;
  /** Emitted after the event with `seq <= afterSeq` (or at turn start when -1). */
  afterSeq: number;
  text: string;
  /** Review position key this note jumps to, when set. */
  jumpKey?: string | null;
}

export interface HistoryLine {
  id: string;
  /** Primary event seq for this line, or null for client-local / legacy lines. */
  seq: number | null;
  /** Board seq shown when selecting this line (cast board if folded with consequences, else resolved board). */
  boardSeq: number | null;
  /** All event seqs folded into this line. */
  seqs: number[];
  /** On a standalone resolution, the seq of the original cast / activation / trigger. */
  sourceSeq: number | null;
  /** Revision of the step that produced this line (for Review jumps). */
  fromRevision: number | null;
  actor: HistoryActor;
  who: string;
  /** Player side associated with this event (for `.theirs` in "Since your last move"). */
  side: 'you' | 'opponent' | null;
  /** Stack id of a `cast` event, for matching resolved-spell thumbnails in the recap. */
  castStackId: number | null;
  segments: Segment[];
  /** Complete sentence form worded by `tell`, with inspectable card segments, for the recap. */
  sentenceSegments: Segment[];
  /** Complete sentence form worded by `tell`, for walk captions. */
  sentence: string;
  /** True when this line is a non-pass player move (marks heavier scrubber ticks). */
  isMove: boolean;
  /** Indented consequences directly following this move or resolution. */
  children: HistoryLine[];
  rule: PresentationRule | null;
  /** Object entity ids referenced by this line. */
  subjects: number[];
  /** Consecutive identical moves collapsed (legacy fallback). */
  count: number;
  /** Folded run of opponent and/or automation priority passes. */
  passes: PassCount[] | null;
  /** Review position key this line jumps to, when known. */
  jumpKey: string | null;
}

export interface HistoryTurn {
  turn: number | null;
  owner: string | null;
  ownerSide: 'you' | 'opponent' | null;
  lines: HistoryLine[];
  /** Count of indented consequence lines in this turn. */
  consequences: number;
  /** Turns with more than `BUSY_CONSEQUENCES` consequences collapse moves by default. */
  busy: boolean;
}

export interface HistoryOptions {
  notes?: readonly HistoryNote[];
  /** Turn number for events preceding the first retained `turn_started`. */
  initialTurn?: number | null;
  initialOwner?: 'you' | 'opponent' | null;
  /** Maps an event to a Review position key for jumping. */
  jumpKeyFor?: (event: PresentationEvent, isMove: boolean) => string | null;
  /** Per-event pilot mapping when mid-game seat swaps change who piloted each seat. */
  playersFor?: (event: PresentationEvent) => TellPlayers;
  /** Override automatic-action detection when reading a cross-swap track in Review. */
  isAutomatic?: (event: PresentationEvent) => boolean;
}

/** More consequences than this in one turn collapse under their moves by default. */
export const BUSY_CONSEQUENCES = 4;

const MOVE_KINDS = new Set([
  'cast',
  'activated',
  'land_played',
  'attack_group',
  'blocked',
  'passed',
]);

// The 12 original pre-history presentation kinds; any other kind guarantees
// a recording carries modern track history (immune to drift as kinds grow).
const LEGACY_KINDS = new Set([
  'cast',
  'targeted',
  'resolved',
  'damage',
  'destroyed',
  'died',
  'attack_group',
  'blocked',
  'turn_started',
  'land_played',
  'discarded',
  'revealed',
]);

/** Exact definition of a player move: move kind with no cause and no rule. */
export function isMoveEvent(event: PresentationEvent): boolean {
  return (
    event.cause === undefined
    && event.rule === undefined
    && MOVE_KINDS.has(event.kind.kind)
  );
}

/** Whether a recording's track carries modern move/label/cause semantics. */
export function hasTrackHistory(events: readonly PresentationEvent[]): boolean {
  return events.some(
    event =>
      event.labels !== undefined
      || event.cause !== undefined
      || event.rule !== undefined
      || !LEGACY_KINDS.has(event.kind.kind),
  );
}

function objectEntities(event: PresentationEvent): number[] {
  return presentationSubjects(event).flatMap(subject => (subject.kind === 'object' ? [subject.id.entity] : []));
}

function eventSeat(event: PresentationEvent): number | null {
  const kind = event.kind;
  switch (kind.kind) {
    case 'cast':
    case 'activated':
    case 'triggered':
    case 'control_changed':
      return kind.controller;
    case 'attack_group':
      return kind.defender.kind === 'player' ? 1 - kind.defender.id : null;
    case 'land_played':
    case 'passed':
    case 'discarded':
    case 'drew':
    case 'revealed':
    case 'searched':
    case 'looked':
    case 'sacrificed':
    case 'lost':
    case 'life':
    case 'token_created':
      return kind.player;
    default:
      return null;
  }
}

function actorForMove(
  event: PresentationEvent,
  players: TellPlayers,
  turnOwner: 'you' | 'opponent' | null,
  isAutomatic?: (event: PresentationEvent) => boolean,
): HistoryActor {
  if (event.kind.kind === 'blocked') {
    if (!isAutomatic && event.caused_by !== null) return 'you';
    return turnOwner === 'opponent' ? 'you' : 'opponent';
  }
  const seat = eventSeat(event);
  if (seat === null) return event.kind.kind === 'attack_group' ? 'opponent' : 'note';
  if (seat === players.youSeat) {
    const auto = isAutomatic
      ? isAutomatic(event)
      : (event.caused_by === null || event.caused_by.startsWith('authority.auto.'));
    return auto ? 'auto' : 'you';
  }
  return 'opponent';
}

function whoForActor(actor: HistoryActor, players: TellPlayers): string {
  if (actor === 'you') return players.you;
  if (actor === 'opponent') return players.opponent;
  if (actor === 'auto') return 'auto';
  return '';
}

function sameObjects(a: readonly { entity: number; incarnation: number }[], b: readonly { entity: number; incarnation: number }[]): boolean {
  return a.length === b.length && a.every((item, i) => item.entity === b[i].entity && item.incarnation === b[i].incarnation);
}

interface StackOrigin {
  kind: 'cast' | 'activated' | 'triggered';
  seq: number;
}

interface TrackFold {
  /** Indices of redundant `destroyed`, folded `targeted`, folded resolutions, and absorbed passes. */
  consumed: Set<number>;
  /** Targets folded onto their parent `cast` / `activated` / `triggered` event, keyed by parent seq. */
  foldedTargets: Map<number, (FoldedTarget & { seq: number })[]>;
  /** Immediate resolution (ignoring passes) folded onto its `cast` / `activated` index. */
  foldedResolution: Map<number, { resolvedIndex: number; passIndices: number[] }>;
  /** Passes immediately preceding a standalone resolution, keyed by resolution index. */
  absorbedBeforeResolution: Map<number, number[]>;
  /** Stack id -> the `cast` / `activated` / `triggered` event that put it on the stack. */
  stackOrigin: Map<number, StackOrigin>;
}

/** Pass 1: index stack origins, fold targets and immediate resolutions, and absorb passes before resolutions. */
function foldTrackEvents(
  events: readonly PresentationEvent[],
  labelsAt: (index: number) => PresentationLabels,
): TrackFold {
  const stackOrigin = new Map<number, StackOrigin>();
  const consumed = new Set<number>();
  const foldedTargets = new Map<number, (FoldedTarget & { seq: number })[]>();
  const foldedResolution = new Map<number, { resolvedIndex: number; passIndices: number[] }>();
  const absorbedBeforeResolution = new Map<number, number[]>();

  for (let i = 0; i < events.length; i += 1) {
    const event = events[i];
    const kind = event.kind;
    if (kind.kind === 'triggered' || kind.kind === 'activated' || kind.kind === 'cast') {
      stackOrigin.set(kind.stack, { kind: kind.kind, seq: event.seq });
    } else if (kind.kind === 'destroyed') {
      const next = events[i + 1];
      if (next?.kind.kind === 'died' && sameObjects(kind.objects, next.kind.objects)) {
        consumed.add(i);
      }
    } else if (kind.kind === 'targeted') {
      let parentSeq: number | null = event.cause ?? null;
      if (parentSeq === null && kind.source.kind === 'stack') {
        // Legacy fallback when events carry no `cause`: look back for matching stack cast/activation/trigger.
        for (let back = i - 1; back >= 0; back -= 1) {
          const prev = events[back];
          if (
            (prev.kind.kind === 'cast' || prev.kind.kind === 'activated' || prev.kind.kind === 'triggered')
            && prev.kind.stack === kind.source.id
          ) {
            parentSeq = prev.seq;
            break;
          }
          if (prev.kind.kind !== 'targeted') break;
        }
      }
      if (parentSeq !== null) {
        const list = foldedTargets.get(parentSeq) ?? [];
        list.push({ target: kind.target, labels: labelsAt(i), event, seq: event.seq });
        foldedTargets.set(parentSeq, list);
        consumed.add(i);
      }
    }
  }

  for (let i = 0; i < events.length; i += 1) {
    if (consumed.has(i)) continue;
    const event = events[i];
    if (event.kind.kind === 'cast' || event.kind.kind === 'activated') {
      const expectedKind = event.kind.kind === 'cast' ? 'resolved' : 'ability_resolved';
      const stackId = event.kind.stack;
      const passIndices: number[] = [];
      let j = i + 1;
      while (j < events.length) {
        if (consumed.has(j)) { j += 1; continue; }
        if (events[j].kind.kind === 'passed') {
          passIndices.push(j);
          j += 1;
          continue;
        }
        break;
      }
      if (j < events.length) {
        const nextKind = events[j].kind;
        if (nextKind.kind === expectedKind && nextKind.stack === stackId) {
          foldedResolution.set(i, { resolvedIndex: j, passIndices });
          consumed.add(j);
          for (const passIdx of passIndices) consumed.add(passIdx);
        }
      }
    }
  }

  // Absorb passes that immediately precede a standalone resolution or illegal-target
  // fizzle so "Trigger · X" -> "X's trigger resolved" is not split by a pass line.
  for (let i = 0; i < events.length; i += 1) {
    if (consumed.has(i)) continue;
    const event = events[i];
    const kind = event.kind.kind;
    const isFizzle = kind === 'countered' && event.rule === 'illegal_targets';
    if (kind !== 'resolved' && kind !== 'ability_resolved' && !isFizzle) continue;
    const passIndices: number[] = [];
    for (let back = i - 1; back >= 0; back -= 1) {
      if (consumed.has(back)) continue;
      if (events[back].kind.kind === 'passed') {
        passIndices.push(back);
        continue;
      }
      break;
    }
    if (passIndices.length > 0) {
      passIndices.reverse();
      absorbedBeforeResolution.set(i, passIndices);
      for (const passIdx of passIndices) consumed.add(passIdx);
    }
  }

  return { consumed, foldedTargets, foldedResolution, absorbedBeforeResolution, stackOrigin };
}

function sideForEvent(
  event: PresentationEvent,
  players: TellPlayers,
  actor: HistoryActor,
): 'you' | 'opponent' | null {
  if (actor === 'you' || actor === 'opponent') return actor;
  const seat = eventSeat(event);
  return seat === null ? null : playerSide(seat, players.youSeat);
}

function makeLine(
  fields: Pick<HistoryLine, 'id' | 'actor' | 'who' | 'segments' | 'sentence'> & Partial<HistoryLine>,
): HistoryLine {
  return {
    seq: null,
    boardSeq: null,
    seqs: [],
    sourceSeq: null,
    fromRevision: null,
    side: null,
    castStackId: null,
    sentenceSegments: fields.segments,
    isMove: false,
    children: [],
    rule: null,
    subjects: [],
    count: 1,
    passes: null,
    jumpKey: null,
    ...fields,
  };
}

function pushPassRun(
  turn: HistoryTurn,
  actor: 'opponent' | 'auto',
  who: string,
  meta: { id: string; seq: number | null; fromRevision: number | null; jumpKey: string | null },
): void {
  const last = turn.lines.at(-1);
  if (last?.passes) {
    if (meta.seq !== null) last.seqs.push(meta.seq);
    const part = last.passes.find(candidate => candidate.actor === actor);
    if (part) part.count += 1;
    else last.passes.push({ actor, who, count: 1 });
    return;
  }
  turn.lines.push(makeLine({
    ...meta,
    seqs: meta.seq !== null ? [meta.seq] : [],
    actor: 'note',
    who: '',
    segments: [{ text: 'passed', card: null }],
    sentence: `${who} passed`,
    passes: [{ actor, who, count: 1 }],
  }));
}

/** Build the unified turn-grouped history from one viewer's presentation track. */
export function buildHistory(
  events: readonly PresentationEvent[],
  labelsAt: (index: number) => PresentationLabels,
  players: TellPlayers,
  options: HistoryOptions = {},
): HistoryTurn[] {
  const {
    consumed,
    foldedTargets,
    foldedResolution,
    absorbedBeforeResolution,
    stackOrigin,
  } = foldTrackEvents(events, labelsAt);

  // Pass 2: assemble turns, blocks, indented consequences, pass runs, and notes.
  const turns: HistoryTurn[] = [];
  let currentTurn: number | null = options.initialTurn ?? null;
  let currentOwner: 'you' | 'opponent' | null = options.initialOwner ?? null;
  let currentTurnPlayer: number | null = null;
  let openBlock: { line: HistoryLine; seqs: Set<number>; castSeq: number | null; resolvedSeq: number | null } | null = null;

  const pendingNotes = options.notes ?? [];
  let noteCursor = 0;

  const ensureTurn = (
    turnNum: number | null,
    owner: 'you' | 'opponent' | null,
    turnPlayers: TellPlayers = players,
  ): HistoryTurn => {
    const existing = turns.at(-1);
    if (existing && existing.turn === turnNum) {
      if (existing.owner === null && owner !== null) {
        existing.owner = turnPlayers[owner];
        existing.ownerSide = owner;
      }
      return existing;
    }
    const created: HistoryTurn = {
      turn: turnNum,
      owner: owner ? turnPlayers[owner] : null,
      ownerSide: owner,
      lines: [],
      consequences: 0,
      busy: false,
    };
    turns.push(created);
    return created;
  };

  const closeBlock = (): void => {
    if (openBlock && openBlock.castSeq !== null && openBlock.resolvedSeq !== null) {
      openBlock.line.boardSeq = openBlock.line.children.length > 0
        ? openBlock.castSeq
        : openBlock.resolvedSeq;
    }
    openBlock = null;
  };

  const pushNote = (note: HistoryNote): void => {
    closeBlock();
    const targetTurn = note.turn ?? currentTurn;
    const owner = targetTurn === currentTurn ? currentOwner : null;
    ensureTurn(targetTurn, owner).lines.push(makeLine({
      id: note.id,
      actor: note.actor,
      who: whoForActor(note.actor, players),
      segments: [{ text: note.text, card: null }],
      sentence: note.text,
      jumpKey: note.jumpKey ?? null,
    }));
  };

  const flushNotesUpTo = (nextSeq: number | null): void => {
    while (noteCursor < pendingNotes.length) {
      const note = pendingNotes[noteCursor];
      if (nextSeq !== null && note.afterSeq >= nextSeq) break;
      pushNote(note);
      noteCursor += 1;
    }
  };

  for (let i = 0; i < events.length; i += 1) {
    const event = events[i];
    flushNotesUpTo(event.seq);
    if (consumed.has(i)) continue;

    const eventPlayers = options.playersFor ? options.playersFor(event) : players;

    if (event.kind.kind === 'turn_started') {
      closeBlock();
      currentTurn = event.kind.turn_number;
      currentTurnPlayer = event.kind.player;
      currentOwner = playerSide(event.kind.player, eventPlayers.youSeat);
      ensureTurn(currentTurn, currentOwner, eventPlayers);
      continue;
    }

    const eventOwner = currentTurnPlayer !== null
      ? playerSide(currentTurnPlayer, eventPlayers.youSeat)
      : currentOwner;
    const turn = ensureTurn(currentTurn, eventOwner, eventPlayers);
    const labels = labelsAt(i);
    const move = isMoveEvent(event);

    if (event.kind.kind === 'passed') {
      closeBlock();
      const actor = actorForMove(event, eventPlayers, eventOwner, options.isAutomatic);
      const jumpKey = options.jumpKeyFor?.(event, true) ?? null;
      if (actor === 'opponent' || actor === 'auto') {
        pushPassRun(turn, actor, whoForActor(actor, eventPlayers), {
          id: `seq-${event.seq}`,
          seq: event.seq,
          fromRevision: event.from_revision,
          jumpKey,
        });
        continue;
      }
      // Person's own explicit pass stays a move.
      turn.lines.push(makeLine({
        id: `seq-${event.seq}`,
        seq: event.seq,
        seqs: [event.seq],
        fromRevision: event.from_revision,
        actor: 'you',
        who: eventPlayers.you,
        segments: tell(event, labels, eventPlayers, { form: 'move' }),
        sentence: segmentText(tell(event, labels, eventPlayers, { form: 'sentence' })),
        jumpKey,
      }));
      continue;
    }

    const targets = foldedTargets.get(event.seq);
    const fold = foldedResolution.get(i);
    const absorbedPasses = absorbedBeforeResolution.get(i) ?? [];
    const seqs: number[] = [
      ...absorbedPasses.map(idx => events[idx].seq),
      event.seq,
      ...(targets?.map(t => t.seq) ?? []),
      ...(fold ? fold.passIndices.map(idx => events[idx].seq) : []),
      ...(fold ? [events[fold.resolvedIndex].seq] : []),
    ];
    const subjects = [
      ...objectEntities(event),
      ...(targets?.flatMap(t => (t.target.kind === 'object' ? [t.target.id.entity] : [])) ?? []),
    ];

    const indented = !move && event.cause !== undefined && openBlock !== null && openBlock.seqs.has(event.cause);
    const origin =
      event.kind.kind === 'resolved' || event.kind.kind === 'ability_resolved' || event.kind.kind === 'countered'
        ? (stackOrigin.get(event.kind.stack) ?? null)
        : null;
    const isTrigger = (event.kind.kind === 'ability_resolved' || event.kind.kind === 'countered') && origin?.kind === 'triggered';
    const isAbility = event.kind.kind === 'countered' && origin?.kind === 'activated';
    const segments = tell(event, labels, eventPlayers, {
      form: 'move',
      indented,
      trigger: isTrigger,
      ability: isAbility,
      targets,
    });
    const sentenceSegments = tell(event, labels, eventPlayers, {
      form: 'sentence',
      indented: false,
      trigger: isTrigger,
      ability: isAbility,
      targets,
    });
    const sentence = segmentText(sentenceSegments);

    const sourceSeq = origin?.seq ?? null;
    const actor: HistoryActor = move ? actorForMove(event, eventPlayers, eventOwner, options.isAutomatic) : 'note';
    const line = makeLine({
      id: `seq-${event.seq}`,
      seq: event.seq,
      boardSeq: event.seq,
      seqs,
      sourceSeq,
      fromRevision: event.from_revision,
      actor,
      who: whoForActor(actor, eventPlayers),
      side: sideForEvent(event, eventPlayers, actor),
      castStackId: event.kind.kind === 'cast' ? event.kind.stack : null,
      segments,
      sentenceSegments,
      sentence,
      isMove: move,
      rule: event.rule ?? null,
      subjects,
      jumpKey: options.jumpKeyFor?.(event, move) ?? null,
    });

    if (indented && openBlock) {
      openBlock.line.children.push(line);
      for (const s of seqs) openBlock.seqs.add(s);
      continue;
    }

    closeBlock();
    turn.lines.push(line);
    openBlock = {
      line,
      seqs: new Set(seqs),
      castSeq: fold ? event.seq : null,
      resolvedSeq: fold ? events[fold.resolvedIndex].seq : null,
    };
  }

  flushNotesUpTo(null);
  closeBlock();

  for (const turn of turns) {
    turn.consequences = turn.lines.reduce((sum, line) => sum + line.children.length, 0);
    turn.busy = turn.consequences > BUSY_CONSEQUENCES;
  }
  return turns;
}

/**
 * Frozen legacy move list for recordings written before moves were emitted as
 * presentation events. Returns the same `HistoryTurn[]` shape as `buildHistory`.
 */
const LEGACY_ACTORS: Record<GameLogEntry['actor'], HistoryActor> = {
  hero: 'you',
  villain: 'opponent',
  auto: 'auto',
  system: 'note',
};
const LEGACY_PASS = 'Pass priority';

export function legacyMoveHistory(
  entries: readonly GameLogEntry[],
  players: { you: string; opponent: string },
  owners: ReadonlyMap<number, 'you' | 'opponent'>,
  names: readonly string[],
): HistoryTurn[] {
  const tellPlayers: TellPlayers = { youSeat: 0, ...players };
  const turns: HistoryTurn[] = [];
  for (const entry of entries) {
    let turn = turns.at(-1);
    if (!turn || turn.turn !== entry.turn) {
      const owner = entry.turn === null ? undefined : owners.get(entry.turn);
      turn = { turn: entry.turn, owner: owner ? players[owner] : null, ownerSide: owner ?? null, lines: [], consequences: 0, busy: false };
      turns.push(turn);
    }
    const actor = LEGACY_ACTORS[entry.actor];
    const last = turn.lines.at(-1);
    if (entry.text === LEGACY_PASS && (actor === 'opponent' || actor === 'auto')) {
      pushPassRun(turn, actor, whoForActor(actor, tellPlayers), {
        id: entry.id,
        seq: null,
        fromRevision: null,
        jumpKey: entry.id,
      });
      continue;
    }
    if (last && actor !== 'you' && last.actor === actor && segmentText(last.segments) === entry.text) {
      last.count += 1;
      continue;
    }
    turn.lines.push(makeLine({
      id: entry.id,
      actor,
      who: whoForActor(actor, tellPlayers),
      segments: segmentsFromText(entry.text, names),
      sentence: entry.text,
      isMove: actor === 'you' || actor === 'opponent',
      jumpKey: entry.id,
    }));
  }
  return turns;
}

/** Live decision strip marks read directly from move events on the track. */
export function trackMarks(
  events: readonly PresentationEvent[],
  labelsAt: (index: number) => PresentationLabels,
  players: TellPlayers,
): StripMark[] {
  const marks: StripMark[] = [];
  let turn: number | null = null;
  let owner: 'you' | 'opponent' | null = null;
  events.forEach((event, index) => {
    if (event.kind.kind === 'turn_started') {
      turn = event.kind.turn_number;
      owner = playerSide(event.kind.player, players.youSeat);
      return;
    }
    if (!isMoveEvent(event)) return;
    const actor = actorForMove(event, players, owner);
    const label = segmentText(tell(event, labelsAt(index), players, { form: 'move' }));
    const at = marks.length;
    const previousTurn = marks.at(-1)?.turn ?? null;
    marks.push({
      at,
      key: `seq-${event.seq}`,
      turn,
      turnStart: at === 0 || turn !== previousTurn,
      decision: actor === 'you' || actor === 'opponent',
      side: actor === 'you' ? 'you' : actor === 'opponent' ? 'opponent' : null,
      value: null,
      label: `Turn ${turn ?? '?'} · ${label}`,
    });
  });
  return marks;
}

// ———————————————————————————————————————— One cursor over the history

export type CursorScope = 'stop' | 'line';

export interface CursorSelection {
  seq: number;
  scope: CursorScope;
}

export interface RecentHistoryItem {
  seq: number;
  by: 'you' | 'opponent' | null;
  segments: Segment[];
  text: string;
  /** Stack id of a cast spell, to attach its resolved-spell thumbnail when off the stack. */
  spell: number | null;
  line: HistoryLine;
}

export interface FlatHistoryLine {
  line: HistoryLine;
  turn: number | null;
  owner: string | null;
  ownerSide: 'you' | 'opponent' | null;
  parent: HistoryLine | null;
  /** Cursor seq used to look up this line's board (`line.boardSeq`). */
  cursor: number;
}

/** The turn the event `seq` happened in: the last `turn_started` at or before it, or null. */
export function turnAt(events: readonly PresentationEvent[], seq: number): number | null {
  let turn: number | null = null;
  for (const event of events) {
    if (event.seq > seq) break;
    if (event.kind.kind === 'turn_started') turn = event.kind.turn_number;
  }
  return turn;
}

/** Every selectable non-pass line of `turns` in display order (parents before children). */
export function flattenHistory(turns: readonly HistoryTurn[]): FlatHistoryLine[] {
  const flat: FlatHistoryLine[] = [];
  for (const turn of turns) {
    for (const line of turn.lines) {
      if (line.boardSeq !== null) {
        flat.push({ line, turn: turn.turn, owner: turn.owner, ownerSide: turn.ownerSide, parent: null, cursor: line.boardSeq });
      }
      for (const child of line.children) {
        if (child.boardSeq !== null) {
          flat.push({ line: child, turn: turn.turn, owner: turn.owner, ownerSide: turn.ownerSide, parent: line, cursor: child.boardSeq });
        }
      }
    }
  }
  return flat;
}

/**
 * Unread history tail for "Since your last move": selectable non-pass lines
 * with `fromRevision >= sinceRevision` (and `seq <= maxSeq` when bounded by a
 * Review position prefix).
 */
export function recentHistory(
  turns: readonly HistoryTurn[],
  sinceRevision: number | null,
  maxSeq: number | null = null,
): RecentHistoryItem[] {
  if (sinceRevision === null) return [];
  const items: RecentHistoryItem[] = [];
  for (const entry of flattenHistory(turns)) {
    const { line } = entry;
    if (line.fromRevision === null || line.fromRevision < sinceRevision) continue;
    if (maxSeq !== null && line.seq !== null && line.seq > maxSeq) continue;
    items.push({
      seq: line.seq ?? entry.cursor,
      by: line.side,
      segments: line.sentenceSegments,
      text: line.sentence,
      spell: line.castStackId,
      line,
    });
  }
  return items;
}

function cursorIndex(flat: readonly FlatHistoryLine[], cursor: number): number {
  const exact = flat.findIndex(item => item.cursor === cursor);
  return exact >= 0 ? exact : flat.findIndex(item => item.line.seqs.includes(cursor));
}

/** The selectable line matching `cursor` (exact `boardSeq` first, then folded `seqs`). */
export function lineAtCursor(turns: readonly HistoryTurn[], cursor: number): FlatHistoryLine | null {
  const flat = flattenHistory(turns);
  const index = cursorIndex(flat, cursor);
  return index >= 0 ? flat[index] : null;
}

/**
 * Step the cursor one selectable history line backward (`-1`) or forward (`1`).
 * Stepping back from now (`null`) lands on the newest selectable line; stepping
 * forward past the newest line returns to now (`null`).
 */
export function stepCursor(turns: readonly HistoryTurn[], cursor: number | null, direction: -1 | 1): number | null {
  const flat = flattenHistory(turns);
  if (flat.length === 0) return null;
  if (cursor === null) return direction < 0 ? flat.at(-1)!.cursor : null;
  const index = cursorIndex(flat, cursor);
  if (index < 0) return direction < 0 ? flat.at(-1)!.cursor : null;
  if (direction < 0) return flat[Math.max(0, index - 1)].cursor;
  return index + 1 < flat.length ? flat[index + 1].cursor : null;
}

/**
 * Top-bar reading for a historical cursor: names the active line or stop
 * itself, prefixed with the parent move's primary card only when the parent is
 * a named player move (e.g. `"Turn 9 · Your Lightning Bolt · Gray Ogre died ·
 * lethal damage"` or `"Turn 6 · Trigger · Vengeful Bloodwitch, targeting
 * you"`). The caller appends `" · Esc for now"`.
 */
export function cursorReading(
  turns: readonly HistoryTurn[],
  events: readonly PresentationEvent[],
  cursor: number | CursorSelection,
  players: { you: string; opponent: string },
  stop?: { turn: number | null; caption: string; lineIds?: readonly string[] } | null,
): string {
  const selection: CursorSelection = typeof cursor === 'number' ? { seq: cursor, scope: 'line' } : cursor;
  const entry = lineAtCursor(turns, selection.seq);
  const turn = entry?.turn ?? (selection.scope === 'stop' ? stop?.turn ?? null : null) ?? turnAt(events, selection.seq);
  const isMultiLineStop = selection.scope === 'stop' && Boolean(stop?.caption) && (stop?.lineIds?.length ?? 0) > 1;
  const parent = entry?.parent ?? null;
  const parentCard =
    parent && parent.isMove && (parent.actor === 'you' || parent.actor === 'opponent')
      ? (parent.segments.find(segment => segment.card !== null)?.card ?? null)
      : null;
  const parentPrefix = parentCard && parent
    ? `${parent.actor === 'you' ? (players.you === 'You' ? 'Your' : possessive(players.you)) : possessive(players.opponent)} ${parentCard}`
    : null;

  const body = isMultiLineStop && stop
    ? stop.caption
    : entry
      ? (parentPrefix ? segmentText(entry.line.segments) : (entry.line.sentence || segmentText(entry.line.segments)))
      : (stop?.caption ?? '');

  const detail = parentPrefix && parentCard
    ? (!body ? parentPrefix : body.includes(parentCard) ? body : `${parentPrefix} · ${body}`)
    : body;

  if (turn !== null && detail) return `Turn ${turn} · ${detail}`;
  if (turn !== null) return `Turn ${turn}`;
  return detail || 'Earlier step';
}
