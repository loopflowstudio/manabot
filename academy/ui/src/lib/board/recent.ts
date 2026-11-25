/**
 * Board-level marks since the person last decided, and the standing lines that
 * join stack entries and blockers to what they act on.
 *
 * `recentMarks(frame, since, events, labelsAt)` reads the authority's
 * presentation events (never a diff of snapshots) for board marks after
 * revision `since`: `touched` objects, `damage` totals, and `resolved` spells
 * that have left the stack. Recap lines themselves come directly from
 * `history.ts:recentHistory` so there is only one history pass.
 *
 * `boardLines(frame, events)` draws standing facts: every spell or ability on
 * the stack to each of its targets, and each blocker to the attacker it blocks,
 * from the projection while combat lasts and from the block events since the
 * person's last move after it, as long as both creatures are still on the board.
 */
import type {
  ExperienceFrame,
  Observation,
  PresentationEvent,
  PresentationLabels,
  StackEntryView,
  SubjectRef,
} from '$lib/types';
import { eventObjectName, eventStackName } from './tell';

/** Something on screen a line can start or end at. */
export type Endpoint =
  | { kind: 'object'; id: number }
  | { kind: 'player'; index: number }
  /** A spell or ability on (or just off) the stack, by its card or stack-object id. */
  | { kind: 'stack'; id: number }
  /** A spell or ability on the stack, by its stack entry id. */
  | { kind: 'entry'; id: number };

export interface BoardLine {
  from: Endpoint;
  to: Endpoint;
  kind: 'target' | 'block';
}

export type Side = 'you' | 'opponent';

/** A spell cast since the last decision that has left the stack. */
export interface ResolvedSpell {
  id: number;
  name: string;
}

export interface RecentMarks {
  /** Spells cast since `since` that are no longer on the stack, in cast order. */
  resolved: ResolvedSpell[];
  /** Objects that attacked, blocked, were dealt damage or arrived since `since`. */
  touched: Set<number>;
  /** Damage dealt since `since`, by object id and by player index. */
  damage: { objects: Map<number, number>; players: Map<number, number> };
}

function playerSide(projection: Pick<Observation, 'agent' | 'opponent'>, id: number): Side | null {
  const { agent, opponent } = projection;
  // Presentation player references use engine ids; prompts use indexes. Both are 0/1 today.
  if (id === agent.id || id === agent.player_index) return 'you';
  if (id === opponent.id || id === opponent.player_index) return 'opponent';
  return null;
}

/** Resolve a presentation player reference to its `player_index` in `projection`. */
export function playerIndex(projection: Pick<Observation, 'agent' | 'opponent'>, id: number): number | null {
  const side = playerSide(projection, id);
  return side === null ? null : side === 'you' ? projection.agent.player_index : projection.opponent.player_index;
}

/** Stack card ids (spells) and stack-object ids (triggered/activated abilities) currently on the stack. */
function stackIds(frame: ExperienceFrame): Set<number> {
  return new Set([
    ...frame.projection.agent.stack.map(card => card.id),
    ...frame.projection.opponent.stack.map(card => card.id),
    ...(frame.projection.stack_objects ?? []).map(entry => entry.id),
  ]);
}

function battlefieldIds(frame: ExperienceFrame): Set<number> {
  return new Set([...frame.projection.agent.battlefield, ...frame.projection.opponent.battlefield].map(p => p.id));
}

function endpoint(subject: SubjectRef, frame: ExperienceFrame, onBoard: Set<number>, onStack: Set<number>): Endpoint | null {
  if (subject.kind === 'player') {
    const index = playerIndex(frame.projection, subject.id);
    return index === null ? null : { kind: 'player', index };
  }
  const id = subject.kind === 'stack' ? subject.id : subject.id.entity;
  if (onStack.has(id)) return { kind: 'stack', id };
  if (onBoard.has(id)) return { kind: 'object', id };
  return null;
}

/**
 * What a stack entry targets, in words: "you", the opponent's name, a
 * permanent's or a stack entry's name. A target that has left the board
 * since it was chosen reads as its zone-free description.
 */
export function targetNames(entry: StackEntryView, frame: ExperienceFrame, names: { you: string; opponent: string }): string[] {
  const permanents = [...frame.projection.agent.battlefield, ...frame.projection.opponent.battlefield];
  return (entry.targets ?? []).map(target => {
    if (target.kind === 'player') {
      const side = playerSide(frame.projection, target.id);
      return !side ? 'a player' : names[side] === 'You' ? 'you' : names[side];
    }
    if (target.kind === 'stack') return frame.projection.stack_objects?.find(other => other.id === target.id)?.label ?? 'a spell';
    return permanents.find(permanent => permanent.id === target.id.entity)?.name ?? 'a card no longer there';
  });
}

/** Standing lines: stack entries to their targets, blockers to attackers. */
export function boardLines(
  frame: ExperienceFrame,
  events: readonly PresentationEvent[],
  resolved: readonly ResolvedSpell[] = [],
  since: number | null = null,
): BoardLine[] {
  const onStack = stackIds(frame);
  const onBoard = battlefieldIds(frame);
  const lines: BoardLine[] = [];
  const seen = new Set<string>();
  const add = (line: BoardLine): void => {
    const id = JSON.stringify(line);
    if (!seen.has(id)) { seen.add(id); lines.push(line); }
  };
  // Everything on the stack, spell or ability, points at the targets its entry records.
  const entries = frame.projection.stack_objects ?? [];
  const entryIds = new Set(entries.map(entry => entry.id));
  for (const entry of entries) {
    for (const target of entry.targets ?? []) {
      const to = target.kind === 'stack'
        ? (entryIds.has(target.id) ? { kind: 'entry', id: target.id } as const : null)
        : endpoint(target, frame, onBoard, onStack);
      if (to) add({ from: { kind: 'entry', id: entry.id }, to, kind: 'target' });
    }
  }
  // Frames recorded before entries carried targets draw a spell's line from its cast events instead.
  const entriesSayTargets = entries.some(entry => entry.targets?.length);
  // A spell that just resolved is drawn beside the stack; its line stays while its target does.
  const shown = new Set([...(entriesSayTargets ? [] : onStack), ...resolved.map(spell => spell.id)]);
  for (const event of events) {
    if (event.kind.kind !== 'targeted' || event.kind.source.kind !== 'stack' || !shown.has(event.kind.source.id)) continue;
    const to = endpoint(event.kind.target, frame, onBoard, onStack);
    if (to) add({ from: { kind: 'stack', id: event.kind.source.id }, to, kind: 'target' });
  }
  // Targets already chosen for the spell or ability the person is announcing now.
  const context = frame.projection.decision_context;
  if (context?.source && (context.announcing || onStack.has(context.source.display_id))) {
    for (const target of context.targets) {
      const to = endpoint(target.object.subject, frame, onBoard, onStack)
        ?? (onBoard.has(target.object.display_id) ? { kind: 'object', id: target.object.display_id } as const : null);
      if (to) add({ from: { kind: 'stack', id: context.source.display_id }, to, kind: 'target' });
    }
  }
  for (const permanent of [...frame.projection.agent.battlefield, ...frame.projection.opponent.battlefield]) {
    if (permanent.blocking !== undefined && onBoard.has(permanent.blocking)) {
      add({ from: { kind: 'object', id: permanent.id }, to: { kind: 'object', id: permanent.blocking }, kind: 'block' });
    }
  }
  // With default stops combat ends before the person's next decision, so a
  // frame rarely holds a block; the block events since their last move keep
  // it drawn, like the recap that says it, while both creatures remain.
  for (const event of events) {
    if (event.kind.kind !== 'blocked' || since === null || event.from_revision < since) continue;
    const attacker = event.kind.attacker.entity;
    for (const blocker of event.kind.blockers) {
      if (onBoard.has(blocker.entity) && onBoard.has(attacker)) {
        add({ from: { kind: 'object', id: blocker.entity }, to: { kind: 'object', id: attacker }, kind: 'block' });
      }
    }
  }
  return lines;
}

/**
 * Board marks (`touched`, `damage`, `resolved`) after revision `since`, from
 * presentation events alone. `labelsAt(i)` gives fallback names retained for
 * older recordings without `event.labels`.
 */
export function recentMarks(
  frame: ExperienceFrame,
  since: number | null,
  events: readonly PresentationEvent[],
  labelsAt: (index: number) => PresentationLabels,
): RecentMarks {
  const marks: RecentMarks = { resolved: [], touched: new Set(), damage: { objects: new Map(), players: new Map() } };
  if (since === null) return marks;
  const onStack = stackIds(frame);
  const casts = new Map<number, ResolvedSpell>();

  events.forEach((event, index) => {
    if (event.from_revision < since) return;
    const kind = event.kind;
    switch (kind.kind) {
      case 'targeted':
        if (kind.target.kind === 'object') marks.touched.add(kind.target.id.entity);
        break;
      case 'cast': {
        const labels = labelsAt(index);
        const name =
          eventObjectName(event, kind.object, labels)
          ?? eventStackName(event, kind.stack, labels)
          ?? 'a spell';
        casts.set(kind.stack, { id: kind.stack, name });
        break;
      }
      case 'activated':
        marks.touched.add(kind.source.entity);
        break;
      case 'counters':
      case 'land_played':
        marks.touched.add(kind.object.entity);
        break;
      case 'damage':
        if (kind.target.kind === 'object') {
          marks.touched.add(kind.target.id.entity);
          marks.damage.objects.set(kind.target.id.entity, (marks.damage.objects.get(kind.target.id.entity) ?? 0) + kind.amount);
        } else if (kind.target.kind === 'player') {
          const idx = playerIndex(frame.projection, kind.target.id);
          if (idx !== null) marks.damage.players.set(idx, (marks.damage.players.get(idx) ?? 0) + kind.amount);
        }
        break;
      case 'attack_group':
        kind.attackers.forEach(o => marks.touched.add(o.entity));
        break;
      case 'blocked':
        marks.touched.add(kind.attacker.entity);
        kind.blockers.forEach(o => marks.touched.add(o.entity));
        break;
    }
  });
  marks.resolved = [...casts.values()].filter(spell => !onStack.has(spell.id));
  return marks;
}

/** Turn number → whose turn, from the retained turn starts and the current frame. */
export function turnOwners(frame: ExperienceFrame, events: readonly PresentationEvent[]): Map<number, Side> {
  const owners = new Map<number, Side>();
  for (const event of events) {
    if (event.kind.kind !== 'turn_started') continue;
    const side = playerSide(frame.projection, event.kind.player);
    if (side) owners.set(event.kind.turn_number, side);
  }
  const { turn, agent } = frame.projection;
  owners.set(turn.turn_number, turn.active_player_id === agent.id ? 'you' : 'opponent');
  return owners;
}
