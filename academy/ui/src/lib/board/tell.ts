/**
 * One browser teller for every Academy view of a `PresentationEvent`: the
 * rail's history, "since your last move", opponent-turn captions and Review.
 *
 * Reads viewer-safe names from `event.labels` first, falling back to retained
 * frame labels only for older recordings whose events carry no `labels`. Card
 * names are emitted as inspectable `Segment`s directly from the event's own
 * subjects, never parsed out of prose.
 */
import { objectLabelKey } from '$lib/presentation';
import type {
  ObjectRenderId,
  PresentationEvent,
  PresentationLabels,
  PresentationRule,
  SubjectRef,
} from '$lib/types';

export interface Segment {
  text: string;
  /** Set when this segment is a card name the viewer can inspect. */
  card: string | null;
}

export interface TellPlayers {
  /** Which native seat (0 or 1) is named by `you`. */
  youSeat: number;
  /** Short display name for `youSeat` ("You" in Play, seat name in Watch). */
  you: string;
  /** Short display name for the other seat ("Heuristic", "Search 64"). */
  opponent: string;
}

export interface FoldedTarget {
  target: SubjectRef;
  labels: PresentationLabels;
  event?: PresentationEvent;
}

export interface TellOptions {
  /**
   * `'move'` omits the leading player name on move kinds (used in the history
   * rail where `.who` sits in its own column). `'sentence'` includes it (used
   * in the recap and captions).
   */
  form?: 'move' | 'sentence';
  /** True when indented directly under the spell or ability that caused it. */
  indented?: boolean;
  /** True when an `ability_resolved` or `countered` came from a `triggered` ability. */
  trigger?: boolean;
  /** True when a `countered` came from an `activated` ability. */
  ability?: boolean;
  /** Targets folded onto a `cast`, `activated` or `triggered` event. */
  targets?: readonly FoldedTarget[];
}

const RULE_WORDS: Record<PresentationRule, string> = {
  draw_step: 'draw step',
  cleanup: 'cleanup',
  lethal_damage: 'lethal damage',
  zero_toughness: 'toughness 0',
  zero_loyalty: 'loyalty 0',
  zero_defense: 'defense 0',
  legend_rule: 'legend rule',
  unattached_aura: 'Aura had nothing to enchant',
  zero_life: '0 life',
  empty_library: 'drew from an empty library',
  poison: '10 poison counters',
  illegal_targets: 'illegal targets',
};

const COUNTER_WORDS: Record<string, string> = {
  plus_one_plus_one: '+1/+1',
  minus_one_minus_one: '−1/−1',
};

export function tellCounter(counter: string): string {
  return COUNTER_WORDS[counter] ?? counter.replace(/_/g, ' ');
}

export function tellPtDelta(power: number, toughness: number): string {
  const signed = (n: number): string => (n >= 0 ? `+${n}` : `−${Math.abs(n)}`);
  return `${signed(power)}/${signed(toughness)}`;
}

export function tellRule(rule: PresentationRule): string {
  return RULE_WORDS[rule];
}

export function segmentText(parts: readonly Segment[]): string {
  return parts.map(part => part.text).join('');
}

const txt = (text: string): Segment => ({ text, card: null });
const cardSeg = (name: string): Segment => ({ text: name, card: name });
const cardOr = (name: string | null, role: string): Segment => (name ? cardSeg(name) : txt(role));

function joinSegments(items: readonly Segment[]): Segment[] {
  const out: Segment[] = [];
  items.forEach((item, index) => {
    if (index > 0) out.push(txt(index === items.length - 1 ? ' and ' : ', '));
    out.push(item);
  });
  return out;
}

export function playerSide(seat: number, youSeat: number): 'you' | 'opponent' {
  return seat === youSeat ? 'you' : 'opponent';
}

export function eventObjectName(
  event: PresentationEvent,
  id: ObjectRenderId,
  fallback: PresentationLabels,
): string | null {
  const key = objectLabelKey(id);
  return event.labels?.objects?.[key] ?? fallback.objects[key] ?? null;
}

export function eventStackName(
  event: PresentationEvent,
  id: number,
  fallback: PresentationLabels,
): string | null {
  const key = String(id);
  return event.labels?.stacks?.[key] ?? fallback.stacks[key] ?? null;
}

function subjectName(
  seat: number,
  players: TellPlayers,
): string {
  return seat === players.youSeat ? players.you : players.opponent;
}

function midSentencePlayer(
  seat: number,
  players: TellPlayers,
  actorSeat: number | null,
): string {
  const name = subjectName(seat, players);
  if (name !== 'You') return name;
  return actorSeat === seat ? 'yourself' : 'you';
}

function subjectSegment(
  subject: SubjectRef,
  event: PresentationEvent | undefined,
  fallback: PresentationLabels,
  players: TellPlayers,
  actorSeat: number | null,
  capitalize = false,
): Segment {
  if (subject.kind === 'player') {
    const name = capitalize
      ? subjectName(subject.id, players)
      : midSentencePlayer(subject.id, players, actorSeat);
    return txt(name);
  }
  if (subject.kind === 'stack') {
    const name = event
      ? eventStackName(event, subject.id, fallback)
      : (fallback.stacks[String(subject.id)] ?? null);
    return cardOr(name, capitalize ? 'A spell' : 'a spell');
  }
  const name = event
    ? eventObjectName(event, subject.id, fallback)
    : (fallback.objects[objectLabelKey(subject.id)] ?? null);
  return cardOr(name, capitalize ? 'A permanent' : 'a permanent');
}

function targetClause(
  targets: readonly FoldedTarget[] | undefined,
  players: TellPlayers,
  actorSeat: number | null,
): Segment[] {
  if (!targets || targets.length === 0) return [];
  const named = targets.map(entry =>
    subjectSegment(entry.target, entry.event, entry.labels, players, actorSeat),
  );
  return [txt(', targeting '), ...joinSegments(named)];
}

function ruleClause(rule: PresentationRule | undefined): Segment[] {
  return rule ? [txt(` · ${tellRule(rule)}`)] : [];
}

/** Word a single presentation event as inspectable text/card segments. */
export function tell(
  event: PresentationEvent,
  fallbackLabels: PresentationLabels,
  players: TellPlayers,
  options: TellOptions = {},
): Segment[] {
  const form = options.form ?? 'move';
  const kind = event.kind;
  const obj = (id: ObjectRenderId, role: string): Segment =>
    cardOr(eventObjectName(event, id, fallbackLabels), role);
  const stackOrObj = (id: ObjectRenderId, stack: number, role: string): Segment =>
    cardOr(
      eventObjectName(event, id, fallbackLabels) ?? eventStackName(event, stack, fallbackLabels),
      role,
    );

  switch (kind.kind) {
    case 'cast': {
      const spell = stackOrObj(kind.object, kind.stack, 'a spell');
      const targeting = targetClause(options.targets, players, kind.controller);
      return form === 'sentence'
        ? [txt(`${subjectName(kind.controller, players)} cast `), spell, ...targeting]
        : [txt('Cast '), spell, ...targeting];
    }
    case 'activated': {
      const source = stackOrObj(kind.source, kind.stack, 'a permanent');
      const targeting = targetClause(options.targets, players, kind.controller);
      return form === 'sentence'
        ? [txt(`${subjectName(kind.controller, players)} activated `), source, ...targeting]
        : [txt('Activate '), source, ...targeting];
    }
    case 'land_played': {
      const land = obj(kind.object, 'a land');
      return form === 'sentence'
        ? [txt(`${subjectName(kind.player, players)} played `), land]
        : [txt('Play '), land];
    }
    case 'attack_group': {
      const attackers = joinSegments(kind.attackers.map(id => obj(id, 'a creature')));
      if (kind.defender.kind === 'object') {
        const defender = obj(kind.defender.id, 'a planeswalker');
        return form === 'sentence'
          ? [...attackers, txt(' attacked '), defender]
          : [txt('Attack '), defender, txt(' with '), ...attackers];
      }
      if (form === 'sentence') {
        const who = subjectName(1 - kind.defender.id, players);
        return [txt(`${who} attacked with `), ...attackers];
      }
      return [txt('Attack with '), ...attackers];
    }
    case 'blocked': {
      const blockers = joinSegments(kind.blockers.map(id => obj(id, 'a creature')));
      const attacker = obj(kind.attacker, 'an attacker');
      return [...blockers, txt(' blocked '), attacker];
    }
    case 'passed':
      return form === 'sentence'
        ? [txt(`${subjectName(kind.player, players)} passed`)]
        : [txt('Pass priority')];
    case 'triggered': {
      const source = stackOrObj(kind.source, kind.stack, 'an ability');
      const targeting = targetClause(options.targets, players, kind.controller);
      return [txt('Trigger · '), source, ...targeting];
    }
    case 'resolved': {
      const spell = cardOr(eventStackName(event, kind.stack, fallbackLabels), 'A spell');
      return [spell, txt(' resolved')];
    }
    case 'ability_resolved': {
      const name =
        eventObjectName(event, kind.source, fallbackLabels)
        ?? eventStackName(event, kind.stack, fallbackLabels);
      const noun = options.trigger ? 'trigger' : 'ability';
      return name
        ? [cardSeg(name), txt(`'s ${noun} resolved`)]
        : [txt(`An ${noun} resolved`)];
    }
    case 'countered': {
      const name = eventStackName(event, kind.stack, fallbackLabels);
      const subject: Segment[] = options.trigger
        ? (name ? [cardSeg(name), txt("'s trigger")] : [txt('A trigger')])
        : options.ability
          ? (name ? [cardSeg(name), txt("'s ability")] : [txt('An ability')])
          : [cardOr(name, 'A spell')];
      if (event.rule === 'illegal_targets') {
        return [...subject, txt(' did not resolve'), ...ruleClause(event.rule)];
      }
      return [...subject, txt(' was countered'), ...ruleClause(event.rule)];
    }
    case 'damage': {
      const target = subjectSegment(kind.target, event, fallbackLabels, players, null);
      if (options.indented) {
        return [txt(`${kind.amount} damage to `), target];
      }
      const source = kind.source
        ? subjectSegment(kind.source, event, fallbackLabels, players, null, true)
        : txt('An effect');
      return [source, txt(` dealt ${kind.amount} to `), target];
    }
    case 'died':
    case 'destroyed': {
      const dead = joinSegments(kind.objects.map(id => obj(id, 'a permanent')));
      const verb = kind.kind === 'died'
        ? ' died'
        : (kind.objects.length === 1 ? ' was destroyed' : ' were destroyed');
      return [...dead, txt(verb), ...ruleClause(event.rule)];
    }
    case 'counters': {
      const target = obj(kind.object, 'a permanent');
      const label = tellCounter(kind.counter);
      const prefix = kind.count === 1
        ? `${label} counter on `
        : `${kind.count} ${label} counters on `;
      return [txt(prefix), target, txt(` (now ${kind.total})`)];
    }
    case 'drew': {
      const who = subjectName(kind.player, players);
      const card = kind.object ? obj(kind.object, 'a card') : txt('a card');
      return [txt(`${who} drew `), card, ...ruleClause(event.rule)];
    }
    case 'discarded': {
      const who = subjectName(kind.player, players);
      const cards = joinSegments(kind.objects.map(id => obj(id, 'a card')));
      return [txt(`${who} discarded `), ...cards, ...ruleClause(event.rule)];
    }
    case 'revealed': {
      const who = subjectName(kind.player, players);
      const cards = joinSegments(kind.cards.map(name => cardSeg(name)));
      return [txt(`${who} revealed `), ...cards];
    }
    case 'searched': {
      const who = subjectName(kind.player, players);
      const possessiveLibrary = who === 'You' ? 'your library' : 'their library';
      return [txt(`${who} searched ${possessiveLibrary}`)];
    }
    case 'looked': {
      const who = subjectName(kind.player, players);
      return [txt(`${who} ${kind.action === 'scry' ? 'scried' : 'surveilled'}`)];
    }
    case 'sacrificed': {
      const who = subjectName(kind.player, players);
      const permanents = joinSegments(kind.objects.map(id => obj(id, 'a permanent')));
      return [txt(`${who} sacrificed `), ...permanents, ...ruleClause(event.rule)];
    }
    case 'life': {
      const who = subjectName(kind.player, players);
      const verb = kind.amount > 0 ? 'gained' : 'lost';
      return [txt(`${who} ${verb} ${Math.abs(kind.amount)} life (now ${kind.total})`)];
    }
    case 'exiled': {
      const exiled = joinSegments(kind.objects.map(id => obj(id, 'a permanent')));
      const verb = kind.objects.length === 1 ? ' was exiled' : ' were exiled';
      return [...exiled, txt(verb), ...ruleClause(event.rule)];
    }
    case 'token_created': {
      const who = subjectName(kind.player, players);
      const names = kind.objects.map(id => eventObjectName(event, id, fallbackLabels));
      if (names.length === 1) {
        const name = names[0];
        if (!name) return [txt(`${who} created a token`)];
        const article = /^[aeiou]/i.test(name) ? 'an' : 'a';
        return [txt(`${who} created ${article} `), cardSeg(name), txt(' token')];
      }
      const first = names[0];
      if (first && names.every(name => name === first)) {
        return [txt(`${who} created ${names.length} `), cardSeg(first), txt(' tokens')];
      }
      const tokens = joinSegments(names.map(name => cardOr(name, 'a token')));
      return [txt(`${who} created `), ...tokens, txt(' tokens')];
    }
    case 'pt_changed': {
      const targets = joinSegments(kind.objects.map(id => obj(id, 'a creature')));
      return [...targets, txt(` got ${tellPtDelta(kind.power, kind.toughness)}`)];
    }
    case 'control_changed': {
      const target = obj(kind.object, 'a permanent');
      if (kind.returned) {
        const to = midSentencePlayer(kind.controller, players, null);
        return [target, txt(` returned to ${to}`), ...ruleClause(event.rule)];
      }
      const who = subjectName(kind.controller, players);
      return [txt(`${who} gained control of `), target, ...ruleClause(event.rule)];
    }
    case 'lost': {
      const who = subjectName(kind.player, players);
      return [txt(`${who} lost the game`), ...ruleClause(event.rule)];
    }
    case 'targeted': {
      const source = subjectSegment(kind.source, event, fallbackLabels, players, null, true);
      const target = subjectSegment(kind.target, event, fallbackLabels, players, null);
      return [source, txt(' targeted '), target];
    }
    case 'turn_started':
      return [txt(`Turn ${kind.turn_number} · ${subjectName(kind.player, players)}`)];
  }
}

/**
 * Splits legacy pre-track narration text on the longest known card names first,
 * so older recordings retain hover inspection without a separate module.
 */
export function segmentsFromText(text: string, names: readonly string[]): Segment[] {
  const sorted = [...new Set(names)].filter(name => name.length > 1).sort((a, b) => b.length - a.length);
  const out: Segment[] = [];
  let rest = text;
  while (rest) {
    let hit: { at: number; name: string } | null = null;
    for (const name of sorted) {
      const at = rest.indexOf(name);
      if (at >= 0 && (!hit || at < hit.at)) hit = { at, name };
    }
    if (!hit) { out.push(txt(rest)); break; }
    if (hit.at > 0) out.push(txt(rest.slice(0, hit.at)));
    out.push(cardSeg(hit.name));
    rest = rest.slice(hit.at + hit.name.length);
  }
  return out;
}
