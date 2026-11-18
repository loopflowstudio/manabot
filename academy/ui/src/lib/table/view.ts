/**
 * What the table draws for one authority frame.
 *
 * `tableView(frame)` is the one place that decides where each legal offer
 * appears: on the card or player it acts on (an anchor), as a button beside
 * the prompt, or in the tray for subjects the table does not draw (Lessons
 * outside the game, cards in hidden or collapsed zones). Every offer is also a
 * row of the full list, in authority order, so nothing strategic is reachable
 * only by the board. The view chooses nothing: a commit is always one offer.
 *
 * `fitRow` sizes the cards of one battlefield row so the largest boards fit
 * by construction: identical permanents fan into one pile, cards shrink, and
 * a row wraps onto further lines before anything scrolls.
 */
import type {
  CardState,
  ExperienceFrame,
  InteractionOffer,
  PermanentState,
  PlayerState,
  StackEntryView,
} from '$lib/types';

// ———————————————————————————————————————— Types

/** Where clicking acts on an offer. */
export type Anchor =
  | { kind: 'object'; id: number }
  | { kind: 'stack'; id: number }
  | { kind: 'player'; index: number }
  | { kind: 'button' }
  | { kind: 'tray'; id: number | null };

export interface PlacedOffer {
  offer: InteractionOffer;
  /** One-based position in the full list; also its number key. */
  number: number;
  anchor: Anchor;
}

export interface PermanentItem {
  permanent: PermanentState;
  offers: PlacedOffer[];
  /** Auras and Equipment attached to this permanent, drawn behind it. */
  attached: PermanentItem[];
}

/** Identical permanents fanned as one pile; each copy stays its own control. */
export interface Pile {
  key: string;
  items: PermanentItem[];
}

export interface SideView {
  player: PlayerState;
  /** Creatures: the row nearest the centre line. */
  creatures: Pile[];
  /** Lands, then other noncreature permanents: the row nearest the player. */
  support: Pile[];
  offers: PlacedOffer[];
}

export interface HandItem {
  card: CardState;
  offers: PlacedOffer[];
}

export interface StackItem {
  entry: StackEntryView;
  card: CardState | null;
  offers: PlacedOffer[];
}

export interface TrayItem {
  placed: PlacedOffer;
  /** The card the offer names, when the frame lets this viewer see it. */
  name: string | null;
}

export interface TableView {
  you: SideView;
  opponent: SideView;
  hand: HandItem[];
  stack: StackItem[];
  /** Subject-less and declining offers, in list order; the first pass/pay is primary. */
  buttons: PlacedOffer[];
  primary: PlacedOffer | null;
  tray: TrayItem[];
  /** Every offer, in authority order. */
  list: PlacedOffer[];
  /** Objects the decision is about: the creature a combat prompt asks about,
   *  or the spell or ability whose targets or costs are being chosen. */
  asked: Set<number>;
}

// ———————————————————————————————————————— Placement

const PRIMARY_VERBS = new Set(['pass_priority', 'pay']);
/** Declining stays a deliberate button: a click on a card never declines. */
const DECLINE_TYPES = new Set(['DECLINE_CHOICE', 'NO_ATTACK', 'NO_BLOCK']);
const COMBAT_VERBS = new Set(['declare_attackers', 'declare_blockers']);
const ROLE_ORDER = ['target', 'source', 'choice'];

/** Ids of every card the table draws, by the zone it is drawn in. */
function drawnObjects(frame: ExperienceFrame): Set<number> {
  const { agent, opponent } = frame.projection;
  return new Set([
    ...agent.hand, ...agent.stack, ...opponent.stack,
    ...agent.battlefield, ...opponent.battlefield,
  ].map(card => card.id));
}

function isButton(offer: InteractionOffer): boolean {
  if (PRIMARY_VERBS.has(offer.verb) || DECLINE_TYPES.has(offer.action_type)) return true;
  // A combat prompt asks about one creature. Attacking with it is a click on
  // it (the declaration names it as its source), and blocking is a click on
  // the attacker (the target); every other combat answer, declining
  // included, sits beside the prompt as an explicit button.
  return COMBAT_VERBS.has(offer.verb) && !offer.subjects?.some(subject => subject.role === 'target' || subject.role === 'source');
}

function anchorOf(offer: InteractionOffer, frame: ExperienceFrame, drawn: Set<number>): Anchor {
  if (isButton(offer)) return { kind: 'button' };
  const players = [frame.projection.agent, frame.projection.opponent];
  const subjects = [...(offer.subjects ?? [])]
    .sort((a, b) => ROLE_ORDER.indexOf(a.role) - ROLE_ORDER.indexOf(b.role));
  for (const { object } of subjects) {
    const subject = object.subject;
    if (subject.kind === 'player') return { kind: 'player', index: subject.id };
    if (subject.kind === 'stack') return { kind: 'stack', id: subject.id };
    if (drawn.has(object.display_id)) return { kind: 'object', id: object.display_id };
  }
  if (subjects.length) return { kind: 'tray', id: subjects[0].object.display_id };
  // Historical frames carry only the focus bridge.
  for (const id of offer.focus) {
    if (drawn.has(id)) return { kind: 'object', id };
    const player = players.find(candidate => candidate.id === id);
    if (player) return { kind: 'player', index: player.player_index };
  }
  return offer.focus.length ? { kind: 'tray', id: offer.focus[0] } : { kind: 'button' };
}

/** Objects named by combat buttons: the creature the prompt is about. */
function askedObjects(placed: PlacedOffer[], frame: ExperienceFrame): Set<number> {
  const asked = new Set<number>();
  const source = frame.projection.decision_context?.source;
  if (source && source.subject.kind === 'object' && frame.prompt) asked.add(source.display_id);
  for (const { offer, anchor } of placed) {
    if (anchor.kind !== 'button' || !COMBAT_VERBS.has(offer.verb)) continue;
    for (const subject of offer.subjects ?? []) asked.add(subject.object.display_id);
    if (!offer.subjects?.length) offer.focus.forEach(id => asked.add(id));
  }
  return asked;
}

function knownName(frame: ExperienceFrame, id: number | null): string | null {
  if (id === null) return null;
  for (const player of [frame.projection.agent, frame.projection.opponent]) {
    const cards = [...player.graveyard, ...player.exile, ...(player.inspected ?? [])];
    const card = cards.find(candidate => candidate.id === id);
    if (card) return card.name;
    const candidate = player.sideboard?.find(entry => entry.candidate_id === id);
    if (candidate) return candidate.name;
  }
  return null;
}

// ———————————————————————————————————————— Board rows

function isCreature(permanent: PermanentState): boolean {
  return permanent.types?.is_creature ?? (permanent.power !== null && permanent.toughness !== null);
}

function isLand(permanent: PermanentState): boolean {
  return permanent.types?.is_land ?? false;
}

/** Everything a person could tell two permanents apart by, plus their offers. */
function pileKey(item: PermanentItem): string {
  const p = item.permanent;
  if (item.attached.length || p.attacking || p.blocking !== undefined || p.effects?.length || p.copy_of) return `#${p.id}`;
  return JSON.stringify([p.name, p.controller_id, p.tapped, p.power, p.toughness, p.damage, p.summoning_sick,
    p.plus1_counters, p.counters ?? null, p.is_token ?? false, p.abilities_lost ?? false,
    item.offers.map(placed => placed.offer.label)]);
}

function piles(items: PermanentItem[]): Pile[] {
  const byKey = new Map<string, Pile>();
  for (const item of items) {
    const key = pileKey(item);
    const pile = byKey.get(key);
    if (pile) pile.items.push(item);
    else byKey.set(key, { key, items: [item] });
  }
  return [...byKey.values()];
}

function side(player: PlayerState, offersAt: (anchor: Anchor) => PlacedOffer[]): SideView {
  const items = new Map<number, PermanentItem>(player.battlefield.map(permanent => [permanent.id,
    { permanent, offers: offersAt({ kind: 'object', id: permanent.id }), attached: [] }]));
  const hosts: PermanentItem[] = [];
  for (const item of items.values()) {
    const host = item.permanent.attached_to === undefined ? undefined : items.get(item.permanent.attached_to);
    if (host && host !== item) host.attached.push(item);
    else hosts.push(item);
  }
  const creatures = hosts.filter(item => isCreature(item.permanent));
  const rest = hosts.filter(item => !isCreature(item.permanent));
  const support = [...rest.filter(item => isLand(item.permanent)), ...rest.filter(item => !isLand(item.permanent))];
  return {
    player,
    creatures: piles(creatures),
    support: piles(support),
    offers: offersAt({ kind: 'player', index: player.player_index }),
  };
}

/** Stack entries in display order (top first) with their spell cards. */
function stackItems(frame: ExperienceFrame, offersAt: (anchor: Anchor) => PlacedOffer[]): StackItem[] {
  const cards = [...frame.projection.agent.stack, ...frame.projection.opponent.stack];
  const used = new Set<number>();
  const entries = frame.projection.stack_objects ?? [];
  const items = entries.map(entry => {
    const card = entry.kind === 'spell'
      ? cards.find(candidate => !used.has(candidate.id) && candidate.name === entry.label) ?? null
      : null;
    if (card) used.add(card.id);
    const offers = [...offersAt({ kind: 'stack', id: entry.id }), ...(card ? offersAt({ kind: 'object', id: card.id }) : [])];
    return { entry, card, offers };
  });
  // Spells the projection lists without a stack entry (historical frames).
  for (const card of cards) {
    if (used.has(card.id)) continue;
    items.push({ entry: { id: -card.id - 1, kind: 'spell', controller: card.owner_id, label: card.name }, card,
      offers: offersAt({ kind: 'object', id: card.id }) });
  }
  return items.reverse();
}

/** A stack entry's short name: a spell's name, or "X's trigger" / "X's ability" for an ability of X. */
export function stackName(entry: StackEntryView): string {
  if (entry.kind === 'spell') return entry.label;
  const source = entry.label.split(' — ')[0];
  return `${source}'s ${entry.kind.includes('trigger') ? 'trigger' : 'ability'}`;
}

// ———————————————————————————————————————— The view

export function tableView(frame: ExperienceFrame): TableView {
  const drawn = drawnObjects(frame);
  const list: PlacedOffer[] = frame.offers.map((offer, index) => ({ offer, number: index + 1, anchor: anchorOf(offer, frame, drawn) }));
  const at = new Map<string, PlacedOffer[]>();
  const anchorKey = (anchor: Anchor): string =>
    anchor.kind === 'player' ? `player:${anchor.index}` : 'id' in anchor ? `${anchor.kind}:${anchor.id}` : anchor.kind;
  for (const placed of list) {
    const key = anchorKey(placed.anchor);
    at.set(key, [...(at.get(key) ?? []), placed]);
  }
  const placedAt = new Set<PlacedOffer>();
  const offersAt = (anchor: Anchor): PlacedOffer[] => {
    const found = at.get(anchorKey(anchor)) ?? [];
    found.forEach(placed => placedAt.add(placed));
    return found;
  };

  const { agent, opponent } = frame.projection;
  const you = side(agent, offersAt);
  const them = side(opponent, offersAt);
  const hand = agent.hand.map(card => ({ card, offers: offersAt({ kind: 'object', id: card.id }) }));
  const stack = stackItems(frame, offersAt);

  const buttons = list.filter(placed => placed.anchor.kind === 'button');
  buttons.forEach(placed => placedAt.add(placed));
  // Anything anchored to something not drawn falls back to the tray.
  const tray = list.filter(placed => !placedAt.has(placed))
    .map(placed => ({ placed, name: knownName(frame, placed.anchor.kind === 'tray' ? placed.anchor.id : null) }));

  return {
    you,
    opponent: them,
    hand,
    stack,
    buttons,
    primary: buttons.find(placed => PRIMARY_VERBS.has(placed.offer.verb)) ?? null,
    tray,
    list,
    asked: askedObjects(list, frame),
  };
}

/** Every place an offer is drawn, for the invariant that each appears somewhere. */
export function offerPlacements(view: TableView): Map<number, number> {
  const counts = new Map<number, number>();
  const count = (placed: PlacedOffer): void => { counts.set(placed.offer.id, (counts.get(placed.offer.id) ?? 0) + 1); };
  const permanents = (pile: Pile[]): PermanentItem[] =>
    pile.flatMap(p => p.items).flatMap(item => [item, ...item.attached]);
  for (const s of [view.you, view.opponent]) {
    s.offers.forEach(count);
    permanents([...s.creatures, ...s.support]).forEach(item => item.offers.forEach(count));
  }
  view.hand.forEach(item => item.offers.forEach(count));
  view.stack.forEach(item => item.offers.forEach(count));
  view.buttons.forEach(count);
  view.tray.forEach(item => count(item.placed));
  return counts;
}

/**
 * A copy the payer would tap is told apart from its pile: each planned copy
 * stands alone, showing its own place in the plan; the rest stay piled.
 */
export function unpilePlanned(piles: Pile[], planned: ReadonlyMap<number, unknown>): Pile[] {
  return piles.flatMap(pile => {
    const marked = pile.items.filter(item => planned.has(item.permanent.id));
    if (!marked.length) return [pile];
    const rest = pile.items.filter(item => !planned.has(item.permanent.id));
    return [...marked.map(item => ({ key: `${pile.key}#plan${item.permanent.id}`, items: [item] })),
      ...(rest.length ? [{ ...pile, items: rest }] : [])];
  });
}

// ———————————————————————————————————————— Sizing

/** Cards in hand are 5:7, as printed. */
export const CARD_RATIO = 5 / 7;
/** Battlefield frames are landscape: name bar above the art, 4:3. */
export const FRAME_RATIO = 4 / 3;

export interface RowFit {
  /** Card width in px. */
  card: number;
  /** Horizontal offset between copies in one pile, px. */
  fan: number;
  lines: number;
  /** True when even the smallest size needs horizontal scrolling. */
  overflow: boolean;
}

export interface RowLimits {
  gap: number;
  minCard: number;
  maxCard: number;
  maxLines: number;
  /** Card width over height. A tapped card keeps its frame, so this is every card's shape. */
  ratio: number;
}

function linesNeeded(piles: number[], width: number, card: number, fan: number, gap: number): number {
  let lines = 1;
  let used = 0;
  for (const count of piles) {
    const w = card + (count - 1) * fan;
    if (used > 0 && used + gap + w > width) { lines += 1; used = w; }
    else used += (used > 0 ? gap : 0) + w;
  }
  return lines;
}

/**
 * The largest card that fits piles of `piles[i]` copies into a `width` ×
 * `height` row. A row of n lines gives each line height / n; a wider fan is
 * tried before a narrower one, and a smaller card before another line.
 */
export function fitRow(piles: number[], width: number, height: number, limits: RowLimits): RowFit {
  const { gap, minCard, maxCard, maxLines, ratio } = limits;
  if (!piles.length || width <= 0 || height <= 0) return { card: maxCard, fan: Math.round(maxCard * 0.3), lines: 1, overflow: false };
  let best: RowFit | null = null;
  for (let lines = 1; lines <= maxLines; lines += 1) {
    const lineHeight = (height - gap * (lines - 1)) / lines;
    for (let card = Math.min(maxCard, Math.floor(lineHeight * ratio)); card >= minCard; card -= 1) {
      const fan = Math.max(8, Math.round(card * 0.22));
      if (linesNeeded(piles, width, card, fan, gap) <= lines) {
        if (!best || card > best.card) best = { card, fan, lines, overflow: false };
        break;
      }
    }
  }
  if (best) return best;
  return { card: minCard, fan: 8, lines: maxLines, overflow: true };
}

/** What `fitRow` needs of each pile: its cards, attachments included. */
export function pileCounts(piles: Pile[]): number[] {
  return piles.map(pile => pile.items.reduce((total, item) => total + 1 + item.attached.length, 0));
}

export interface HalfSplit {
  /** Height given to the creature row, px; the support row takes the rest. */
  creatures: number;
  creatureFit: RowFit;
  supportFit: RowFit;
}

/**
 * Divides one player's half between its creature row and its land/support row
 * by what each needs, so a crowded row takes height from a sparse one.
 * Creatures are what a player reads most, so they win ties, and the support
 * row may run a little smaller than the creatures before it gains height.
 */
export function splitHalf(
  creatures: number[],
  support: number[],
  width: number,
  height: number,
  limits: { creatures: RowLimits; support: RowLimits; gap: number },
): HalfSplit {
  const usable = height - limits.gap;
  let best: (HalfSplit & { score: number }) | null = null;
  for (let share = 0.3; share <= 0.761; share += 0.02) {
    const top = Math.round(usable * share);
    const creatureFit = fitRow(creatures, width, top, limits.creatures);
    const supportFit = fitRow(support, width, usable - top, limits.support);
    const c = creatures.length ? creatureFit.card : Infinity;
    const s = support.length ? supportFit.card + 12 : Infinity;
    const score = Math.min(c, s) * 1000 + (creatures.length ? creatureFit.card : 0);
    if (!best || score > best.score) best = { creatures: top, creatureFit, supportFit, score };
  }
  return best!;
}
