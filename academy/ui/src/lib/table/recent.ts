/**
 * What changed on the table since the person last decided, and the lines that
 * join cards to what they act on.
 *
 * `recentView(frame, since, events, labels)` reads the authority's presentation
 * events (never a diff of snapshots) for what happened after revision `since`:
 * spells cast and whether they are still on the stack, attacks, blocks, damage
 * and deaths, and how much damage each object and player was dealt.
 * `tableLines(frame, events)` draws standing facts: a spell on the stack to
 * each of its targets, and each blocker to the attacker it blocks, from the
 * projection while combat lasts and from the block events since the person's
 * last move after it, as long as both creatures are still on the table.
 * Both are pure, so the table's motion is a rendering of facts that remain
 * readable when motion is reduced.
 *
 * Names: protocol v1's projection carries no incarnation, so presentation
 * labels name only incarnation-zero objects. `nameBook` adds the names the
 * person could read on the frames around their move, by object id, and the
 * names of spells cast since then from the authority's own narration of that
 * move ("Cast Water Tribe Rallier"), in order: an opponent's creature spell
 * comes from a hidden hand and becomes a permanent with another id, so no
 * projection names it.
 */
import type {
  ExperienceFrame,
  GameLogEntry,
  ObjectRenderId,
  PlayerState,
  PresentationEvent,
  PresentationLabels,
  SubjectRef,
} from '$lib/types';

/** Something on screen a line can start or end at. */
export type Endpoint =
  | { kind: 'object'; id: number }
  | { kind: 'player'; index: number }
  /** A spell on (or just off) the stack, by its card id. */
  | { kind: 'stack'; id: number };

export interface TableLine {
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

export interface RecentFact {
  seq: number;
  by: Side | null;
  text: string;
  /** The card id of the spell a cast fact names. */
  spell: number | null;
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

function key(id: ObjectRenderId): string {
  return `${id.entity}:${id.incarnation}`;
}

/** Names the person could read around their move; see the module note. */
export interface NameBook {
  /** Object or card id → name, from the frames before and after the move. */
  ids: ReadonlyMap<number, string>;
  /** Names of spells each side cast since the move, in the order narrated. */
  casts: Record<Side, string[]>;
}

export function nameBook(frames: readonly ExperienceFrame[], moves: readonly GameLogEntry[]): NameBook {
  const ids = new Map<number, string>();
  for (const frame of frames) {
    for (const player of [frame.projection.agent, frame.projection.opponent]) {
      for (const card of [...player.hand, ...player.stack, ...player.graveyard, ...player.exile]) ids.set(card.id, card.name);
      for (const permanent of player.battlefield) if (permanent.name) ids.set(permanent.id, permanent.name);
    }
  }
  const cast = (actor: GameLogEntry['actor']): string[] => moves
    .filter(move => move.actor === actor && move.text.startsWith('Cast '))
    .map(move => move.text.slice('Cast '.length));
  return { ids, casts: { you: cast('hero'), opponent: cast('villain') } };
}

function objectName(id: ObjectRenderId, labels: PresentationLabels, book: NameBook): string | null {
  return labels.objects[key(id)] ?? book.ids.get(id.entity) ?? null;
}

function nameOf(subject: SubjectRef, labels: PresentationLabels, frame: ExperienceFrame, names: { you: string; opponent: string }, book: NameBook): string {
  if (subject.kind === 'player') {
    const side = playerSide(frame, subject.id);
    return side ? names[side] : 'a player';
  }
  if (subject.kind === 'stack') return labels.stacks[String(subject.id)] ?? book.ids.get(subject.id) ?? 'a spell';
  return objectName(subject.id, labels, book) ?? 'a permanent';
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
export function tableLines(
  frame: ExperienceFrame,
  events: readonly PresentationEvent[],
  resolved: readonly ResolvedSpell[] = [],
  since: number | null = null,
): TableLine[] {
  const onStack = stackCardIds(frame);
  const onBoard = battlefieldIds(frame);
  const lines: TableLine[] = [];
  const seen = new Set<string>();
  const add = (line: TableLine): void => {
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

function list(names: string[]): string {
  return names.length <= 1 ? names.join('') : `${names.slice(0, -1).join(', ')} and ${names.at(-1)}`;
}

/**
 * What happened after revision `since`, from presentation events alone.
 * `labelsAt(i)` gives the names retained with `events[i]`.
 */
export function recentView(
  frame: ExperienceFrame,
  since: number | null,
  events: readonly PresentationEvent[],
  labelsAt: (index: number) => PresentationLabels,
  names: { you: string; opponent: string },
  book: NameBook = { ids: new Map(), casts: { you: [], opponent: [] } },
): RecentView {
  const view: RecentView = { resolved: [], touched: new Set(), facts: [], damage: { objects: new Map(), players: new Map() } };
  if (since === null) return view;
  const narrated = { you: [...book.casts.you], opponent: [...book.casts.opponent] };
  const onStack = stackCardIds(frame);
  const casts = new Map<number, ResolvedSpell>();
  const sideName = (side: Side | null): string => side ? names[side] : 'An effect';
  events.forEach((event, index) => {
    if (event.from_revision < since) return;
    const labels = labelsAt(index);
    const kind = event.kind;
    const fact = (by: Side | null, text: string, spell: number | null = null): void => { view.facts.push({ seq: event.seq, by, text, spell }); };
    switch (kind.kind) {
      case 'cast': {
        const by = playerSide(frame, kind.controller) ?? 'opponent';
        const told = narrated[by].shift();
        // A hidden hand's card id names nothing the person saw; trust the narration first.
        const name = labels.objects[key(kind.object)] ?? (by === 'opponent' ? told : undefined)
          ?? book.ids.get(kind.object.entity) ?? labels.stacks[String(kind.stack)] ?? told ?? 'a spell';
        casts.set(kind.stack, { id: kind.stack, name, by, targets: [] });
        fact(by, `${names[by]} cast ${name}`, kind.stack);
        break;
      }
      case 'targeted': {
        const target = nameOf(kind.target, labels, frame, names, book);
        if (kind.source.kind === 'stack') casts.get(kind.source.id)?.targets.push(target);
        const last = view.facts.at(-1);
        if (last) last.text += `${last.text.includes(', targeting ') ? ' and ' : ', targeting '}${target}`;
        if (kind.target.kind === 'object') view.touched.add(kind.target.id.entity);
        break;
      }
      case 'resolved':
        break;
      case 'damage': {
        const source = kind.source ? nameOf(kind.source, labels, frame, names, book) : 'An effect';
        fact(null, `${source} dealt ${kind.amount} to ${nameOf(kind.target, labels, frame, names, book)}`);
        if (kind.target.kind === 'object') {
          view.touched.add(kind.target.id.entity);
          view.damage.objects.set(kind.target.id.entity, (view.damage.objects.get(kind.target.id.entity) ?? 0) + kind.amount);
        } else if (kind.target.kind === 'player') {
          const index = playerIndex(frame, kind.target.id);
          if (index !== null) view.damage.players.set(index, (view.damage.players.get(index) ?? 0) + kind.amount);
        }
        break;
      }
      case 'destroyed':
      case 'died': {
        // A creature destroyed is also reported as dying; say it once.
        const next = events[index + 1]?.kind;
        const same = next?.kind === 'died' && next.objects.map(key).join() === kind.objects.map(key).join();
        if (kind.kind === 'destroyed' && same) break;
        const dead = kind.objects.map(object => objectName(object, labels, book) ?? 'a permanent');
        fact(null, `${list(dead)} ${kind.kind === 'died' ? 'died' : 'was destroyed'}`);
        break;
      }
      case 'attack_group': {
        const attackers = kind.attackers.map(object => objectName(object, labels, book) ?? 'a creature');
        const owner = kind.defender.kind === 'player' ? playerSide(frame, kind.defender.id) : null;
        const by: Side | null = owner === 'you' ? 'opponent' : owner === 'opponent' ? 'you' : null;
        fact(by, `${sideName(by)} attacked with ${list(attackers)}`);
        kind.attackers.forEach(object => view.touched.add(object.entity));
        break;
      }
      case 'blocked': {
        const attacker = objectName(kind.attacker, labels, book) ?? 'an attacker';
        const blockers = kind.blockers.map(object => objectName(object, labels, book) ?? 'a creature');
        fact(null, `${list(blockers)} blocked ${attacker}`);
        [kind.attacker, ...kind.blockers].forEach(object => view.touched.add(object.entity));
        break;
      }
      case 'turn_started':
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

/** Every card name this viewer can see now, for inspecting names in the log. */
export function visibleNames(frame: ExperienceFrame, labels: PresentationLabels): string[] {
  const players: PlayerState[] = [frame.projection.agent, frame.projection.opponent];
  const names = players.flatMap(player => [
    ...player.hand, ...player.graveyard, ...player.exile, ...player.stack, ...(player.inspected ?? []),
  ].map(card => card.name).concat(player.battlefield.map(p => p.name ?? '')));
  return [...new Set([...names, ...Object.values(labels.objects), ...Object.values(labels.stacks)].filter(Boolean))];
}
