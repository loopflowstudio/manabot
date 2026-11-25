/**
 * What changed on the board since the person last decided, and the lines that
 * join cards to what they act on.
 *
 * `recentView(frame, since, events, labelsAt, names)` reads the authority's
 * presentation events (never a diff of snapshots) for what happened after
 * revision `since`: board marks (`touched`, `damage`), resolved spells, and
 * recap facts worded by the shared `tell` module.
 * `boardLines(frame, events)` draws standing facts: a spell on the stack to
 * each of its targets, and each blocker to the attacker it blocks, from the
 * projection while combat lasts and from the block events since the person's
 * last move after it, as long as both creatures are still on the board.
 */
import type {
  ExperienceFrame,
  PlayerState,
  PresentationEvent,
  PresentationLabels,
  SubjectRef,
} from '$lib/types';
import {
  eventObjectName,
  eventStackName,
  segmentText,
  tell,
  type FoldedTarget,
  type Segment,
  type TellPlayers,
} from './tell';
import { inspectableCards } from './view';

/** Something on screen a line can start or end at. */
export type Endpoint =
  | { kind: 'object'; id: number }
  | { kind: 'player'; index: number }
  /** A spell on (or just off) the stack, by its card id. */
  | { kind: 'stack'; id: number };

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
  by: Side;
  targets: string[];
}

/** What a fact reports; beats group consecutive facts of one kind. */
export type FactKind =
  | 'cast'
  | 'activated'
  | 'triggered'
  | 'ability_resolved'
  | 'countered'
  | 'counters'
  | 'damage'
  | 'died'
  | 'destroyed'
  | 'attacked'
  | 'blocked'
  | 'land'
  | 'discarded'
  | 'revealed'
  | 'drew'
  | 'searched'
  | 'looked'
  | 'sacrificed'
  | 'lost';

export interface RecentFact {
  seq: number;
  by: Side | null;
  kind: FactKind;
  text: string;
  segments: Segment[];
  /** The card id of the spell a cast fact names. */
  spell: number | null;
  /** The objects on the board the fact is about (attackers, blockers, targets, what was dealt damage), by id. */
  subjects: number[];
}

export interface RecentView {
  /** Spells cast since `since` that are no longer on the stack, in cast order. */
  resolved: ResolvedSpell[];
  /** Objects that attacked, blocked, were dealt damage or arrived since `since`. */
  touched: Set<number>;
  /** One line per fact since `since`, in authority order. */
  facts: RecentFact[];
  /** Damage dealt since `since`, by object id and by player index. */
  damage: { objects: Map<number, number>; players: Map<number, number> };
}

function playerSide(frame: ExperienceFrame, id: number): Side | null {
  const { agent, opponent } = frame.projection;
  // Presentation player references use engine ids; prompts use indexes. Both are 0/1 today.
  if (id === agent.id || id === agent.player_index) return 'you';
  if (id === opponent.id || id === opponent.player_index) return 'opponent';
  return null;
}

function playerIndex(frame: ExperienceFrame, id: number): number | null {
  const side = playerSide(frame, id);
  return side === null ? null : side === 'you' ? frame.projection.agent.player_index : frame.projection.opponent.player_index;
}

function stackCardIds(frame: ExperienceFrame): Set<number> {
  return new Set([...frame.projection.agent.stack, ...frame.projection.opponent.stack].map(card => card.id));
}

function battlefieldIds(frame: ExperienceFrame): Set<number> {
  return new Set([...frame.projection.agent.battlefield, ...frame.projection.opponent.battlefield].map(p => p.id));
}

function endpoint(subject: SubjectRef, frame: ExperienceFrame, onBoard: Set<number>, onStack: Set<number>): Endpoint | null {
  if (subject.kind === 'player') {
    const index = playerIndex(frame, subject.id);
    return index === null ? null : { kind: 'player', index };
  }
  const id = subject.kind === 'stack' ? subject.id : subject.id.entity;
  if (onStack.has(id)) return { kind: 'stack', id };
  if (onBoard.has(id)) return { kind: 'object', id };
  return null;
}

/** Standing lines: spells on the stack to their targets, blockers to attackers. */
export function boardLines(
  frame: ExperienceFrame,
  events: readonly PresentationEvent[],
  resolved: readonly ResolvedSpell[] = [],
  since: number | null = null,
): BoardLine[] {
  const onStack = stackCardIds(frame);
  const onBoard = battlefieldIds(frame);
  const lines: BoardLine[] = [];
  const seen = new Set<string>();
  const add = (line: BoardLine): void => {
    const id = JSON.stringify(line);
    if (!seen.has(id)) { seen.add(id); lines.push(line); }
  };
  // A spell that just resolved is drawn beside the stack; its line stays while its target does.
  const shown = new Set([...onStack, ...resolved.map(spell => spell.id)]);
  for (const event of events) {
    if (event.kind.kind !== 'targeted' || event.kind.source.kind !== 'stack' || !shown.has(event.kind.source.id)) continue;
    const to = endpoint(event.kind.target, frame, onBoard, onStack);
    if (to) add({ from: { kind: 'stack', id: event.kind.source.id }, to, kind: 'target' });
  }
  // Targets already chosen for the spell the person is casting now.
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

function sameObjects(
  a: readonly { entity: number; incarnation: number }[],
  b: readonly { entity: number; incarnation: number }[],
): boolean {
  return a.length === b.length && a.every((o, i) => o.entity === b[i].entity && o.incarnation === b[i].incarnation);
}

/**
 * What happened after revision `since`, from presentation events alone.
 * `labelsAt(i)` gives fallback names retained for older recordings without `event.labels`.
 */
export function recentView(
  frame: ExperienceFrame,
  since: number | null,
  events: readonly PresentationEvent[],
  labelsAt: (index: number) => PresentationLabels,
  names: { you: string; opponent: string },
): RecentView {
  const view: RecentView = { resolved: [], touched: new Set(), facts: [], damage: { objects: new Map(), players: new Map() } };
  if (since === null) return view;
  const players: TellPlayers = {
    youSeat: frame.projection.agent.player_index,
    you: names.you,
    opponent: names.opponent,
  };
  const onStack = stackCardIds(frame);
  const casts = new Map<number, ResolvedSpell>();
  const stackTriggers = new Set<number>();
  for (const event of events) {
    if (event.kind.kind === 'triggered') stackTriggers.add(event.kind.stack);
  }

  // Fold targets onto their cast/activated/triggered event inside the recap window.
  const foldedTargets = new Map<number, FoldedTarget[]>();
  const consumed = new Set<number>();
  for (let i = 0; i < events.length; i += 1) {
    const event = events[i];
    if (event.from_revision < since) continue;
    if (event.kind.kind === 'targeted') {
      let parentSeq: number | null = event.cause ?? null;
      if (parentSeq === null && event.kind.source.kind === 'stack') {
        for (let back = i - 1; back >= 0; back -= 1) {
          const prev = events[back];
          if (
            (prev.kind.kind === 'cast' || prev.kind.kind === 'activated' || prev.kind.kind === 'triggered')
            && prev.kind.stack === event.kind.source.id
          ) {
            parentSeq = prev.seq;
            break;
          }
          if (prev.kind.kind !== 'targeted') break;
        }
      }
      if (parentSeq !== null) {
        const list = foldedTargets.get(parentSeq) ?? [];
        list.push({ target: event.kind.target, labels: labelsAt(i), event });
        foldedTargets.set(parentSeq, list);
        consumed.add(i);
      }
    }
  }

  events.forEach((event, index) => {
    if (event.from_revision < since) return;
    if (consumed.has(index)) {
      if (event.kind.kind === 'targeted') {
        const target = event.kind.target;
        if (target.kind === 'object') view.touched.add(target.id.entity);
        if (event.kind.source.kind === 'stack') {
          const cast = casts.get(event.kind.source.id);
          if (cast) {
            const labels = labelsAt(index);
            const targetName = target.kind === 'player'
              ? (playerSide(frame, target.id) === 'you'
                  ? (names.you === 'You' ? (cast.by === 'you' ? 'yourself' : 'you') : names.you)
                  : names.opponent)
              : target.kind === 'stack'
                ? (eventStackName(event, target.id, labels) ?? 'a spell')
                : (eventObjectName(event, target.id, labels) ?? 'a permanent');
            cast.targets.push(targetName);
          }
        }
      }
      return;
    }

    const labels = labelsAt(index);
    const kind = event.kind;
    const targets = foldedTargets.get(event.seq);
    const targetEntities = targets?.flatMap(t => (t.target.kind === 'object' ? [t.target.id.entity] : [])) ?? [];

    const push = (
      by: Side | null,
      factKind: FactKind,
      spell: number | null = null,
      subjects: number[] = [],
      trigger = false,
    ): void => {
      const segments = tell(event, labels, players, { form: 'sentence', targets, trigger });
      view.facts.push({
        seq: event.seq,
        by,
        kind: factKind,
        text: segmentText(segments),
        segments,
        spell,
        subjects,
      });
    };

    switch (kind.kind) {
      case 'cast': {
        const by = playerSide(frame, kind.controller) ?? 'opponent';
        const name =
          eventObjectName(event, kind.object, labels)
          ?? eventStackName(event, kind.stack, labels)
          ?? 'a spell';
        casts.set(kind.stack, { id: kind.stack, name, by, targets: [] });
        push(by, 'cast', kind.stack, targetEntities);
        break;
      }
      case 'activated': {
        const by = playerSide(frame, kind.controller);
        push(by, 'activated', null, [kind.source.entity, ...targetEntities]);
        view.touched.add(kind.source.entity);
        break;
      }
      case 'triggered':
        push(null, 'triggered', null, [kind.source.entity, ...targetEntities]);
        break;
      case 'ability_resolved':
        push(null, 'ability_resolved', null, [kind.source.entity], stackTriggers.has(kind.stack));
        break;
      case 'countered':
        push(null, 'countered');
        break;
      case 'counters':
        push(null, 'counters', null, [kind.object.entity]);
        view.touched.add(kind.object.entity);
        break;
      case 'damage': {
        const subjects = [kind.source, kind.target].flatMap(s => (s?.kind === 'object' ? [s.id.entity] : []));
        push(null, 'damage', null, subjects);
        if (kind.target.kind === 'object') {
          view.touched.add(kind.target.id.entity);
          view.damage.objects.set(kind.target.id.entity, (view.damage.objects.get(kind.target.id.entity) ?? 0) + kind.amount);
        } else if (kind.target.kind === 'player') {
          const idx = playerIndex(frame, kind.target.id);
          if (idx !== null) view.damage.players.set(idx, (view.damage.players.get(idx) ?? 0) + kind.amount);
        }
        break;
      }
      case 'destroyed':
      case 'died': {
        const next = events[index + 1]?.kind;
        if (kind.kind === 'destroyed' && next?.kind === 'died' && sameObjects(kind.objects, next.objects)) break;
        push(null, kind.kind, null, kind.objects.map(o => o.entity));
        break;
      }
      case 'attack_group': {
        const owner = kind.defender.kind === 'player' ? playerSide(frame, kind.defender.id) : null;
        const by: Side | null = owner === 'you' ? 'opponent' : owner === 'opponent' ? 'you' : null;
        push(by, 'attacked', null, kind.attackers.map(o => o.entity));
        kind.attackers.forEach(o => view.touched.add(o.entity));
        break;
      }
      case 'blocked': {
        push(null, 'blocked', null, [kind.attacker.entity, ...kind.blockers.map(o => o.entity)]);
        view.touched.add(kind.attacker.entity);
        kind.blockers.forEach(o => view.touched.add(o.entity));
        break;
      }
      case 'land_played': {
        const by = playerSide(frame, kind.player);
        push(by, 'land', null, [kind.object.entity]);
        view.touched.add(kind.object.entity);
        break;
      }
      case 'discarded': {
        const by = playerSide(frame, kind.player);
        push(by, 'discarded', null, kind.objects.map(o => o.entity));
        break;
      }
      case 'drew': {
        const by = playerSide(frame, kind.player);
        push(by, 'drew', null, kind.object ? [kind.object.entity] : []);
        break;
      }
      case 'revealed': {
        const by = playerSide(frame, kind.player);
        push(by, 'revealed');
        break;
      }
      case 'searched': {
        const by = playerSide(frame, kind.player);
        push(by, 'searched');
        break;
      }
      case 'looked': {
        const by = playerSide(frame, kind.player);
        push(by, 'looked');
        break;
      }
      case 'sacrificed': {
        const by = playerSide(frame, kind.player);
        push(by, 'sacrificed', null, kind.objects.map(o => o.entity));
        break;
      }
      case 'lost': {
        const by = playerSide(frame, kind.player);
        push(by, 'lost');
        break;
      }
      case 'targeted':
      case 'resolved':
      case 'turn_started':
      case 'passed':
        break;
    }
  });
  view.resolved = [...casts.values()].filter(spell => !onStack.has(spell.id));
  return view;
}

/** Turn number → whose turn, from the retained turn starts and the current frame. */
export function turnOwners(frame: ExperienceFrame, events: readonly PresentationEvent[]): Map<number, Side> {
  const owners = new Map<number, Side>();
  for (const event of events) {
    if (event.kind.kind !== 'turn_started') continue;
    const side = playerSide(frame, event.kind.player);
    if (side) owners.set(event.kind.turn_number, side);
  }
  const { turn, agent } = frame.projection;
  owners.set(turn.turn_number, turn.active_player_id === agent.id ? 'you' : 'opponent');
  return owners;
}

/** Every card name this viewer can see now, for inspecting names in legacy logs. */
export function visibleNames(frame: ExperienceFrame, labels: PresentationLabels): string[] {
  const players: PlayerState[] = [frame.projection.agent, frame.projection.opponent];
  const names = inspectableCards(players).map(card => card.name)
    .concat(players.flatMap(player => player.battlefield.map(p => p.name ?? '')));
  return [...new Set([...names, ...Object.values(labels.objects), ...Object.values(labels.stacks)].filter(Boolean))];
}
