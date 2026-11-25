/**
 * Cards move between zones instead of jumping.
 *
 * `Travel` remembers where every card was drawn just before a frame changes
 * the board (`snapshot`), then animates each card from there to where it is
 * drawn now (`play`): a card that stayed in place but moved because its row
 * reflowed slides; a card that is new to the board comes from the card of the
 * same name that has just left (a spell from hand to the stack, a permanent
 * from the stack to the battlefield), or else from its owner's plate, as an
 * opponent's card arrives from a hidden hand. This is a rendering of state
 * the frame already shows: identities stay the authority's, names only guide
 * the eye, and with reduced motion nothing animates and nothing is lost.
 *
 * Elements opt in with `data-travel-id` (the object or card id), and
 * `data-travel-name`; `data-travel-side` ("you" | "opponent") names whose plate a
 * card with no earlier position comes from. The animation uses the CSS
 * `translate` and `scale` properties, which compose with a card's own
 * `transform` (tapped rotation, lift) rather than replacing it.
 *
 * A frame can arrive while cards are still travelling. The snapshot records
 * where each card is drawn at that moment, mid-flight included; `play` then
 * cancels every unfinished travel before it measures, so it reads each card's
 * slot rather than a point on the old path, and the new travel starts where
 * the card is drawn and ends in its slot. Every travel ends at the card's own
 * layout position and holds nothing afterwards, so no card stays away from
 * its slot once its travel ends.
 */

import type { Observation, PresentationEvent, PresentationLabels } from '$lib/types';
import { eventObjectName, tellRule } from './tell';

export interface SeenCard {
  id: string;
  name: string;
  side: 'you' | 'opponent' | null;
  rect: DOMRect;
}

/** Motion between two board stops stays within 180 ms (DESIGN.md / docs/academy.md). */
const MOVE_MS = 180;
const ARRIVE_MS = 180;
export const DEATH_MS = 180;
/** Longest card travel after a frame or stop change, in ms: what lines must follow. */
export const LONGEST_TRAVEL_MS = Math.max(MOVE_MS, ARRIVE_MS, DEATH_MS);
const EASE = 'cubic-bezier(0.2, 0.7, 0.2, 1)';
/** Names this module's animations, so cancelling travel leaves CSS transitions alone. */
const TRAVEL = 'travel';
const MAX_KNOWN_RECTS = 128;

export interface DeathMark {
  entity: number;
  name: string;
  rule: string | null;
  side: 'you' | 'opponent';
  /** Position of the card (or row fallback) relative to `.board`, in px. */
  x: number;
  y: number;
  width: number;
  height: number;
  /** Offset from card center to the owner's graveyard count, in px. */
  toDx: number;
  toDy: number;
  /** Unique key per transition so WAAPI runs once per departure. */
  key: string;
  /** True when the card should travel to the graveyard (normal motion & just departed). */
  animate: boolean;
}

function cards(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>('[data-travel-id]')];
}

function visible(rect: DOMRect): boolean {
  return rect.width > 0 && rect.height > 0;
}

function readSide(value: string | undefined): 'you' | 'opponent' | null {
  return value === 'you' || value === 'opponent' ? value : null;
}

export class Travel {
  private seen: SeenCard[] = [];
  private previousById = new Map<string, SeenCard>();
  private known = new Map<string, SeenCard>();

  /** Records where every travelling card is drawn now; call before the DOM updates. */
  snapshot(root: HTMLElement | null): void {
    this.seen = root ? cards(root).flatMap(element => {
      const rect = element.getBoundingClientRect();
      if (!visible(rect)) return [];
      const entry: SeenCard = {
        id: element.dataset.travelId!,
        name: element.dataset.travelName ?? '',
        side: readSide(element.dataset.travelSide),
        rect,
      };
      if (entry.id.startsWith('perm:')) {
        this.known.delete(entry.id);
        this.known.set(entry.id, entry);
        if (this.known.size > MAX_KNOWN_RECTS) {
          const oldest = this.known.keys().next().value;
          if (oldest !== undefined) this.known.delete(oldest);
        }
      }
      return [entry];
    }) : [];
    this.previousById = new Map(this.seen.map(entry => [entry.id, entry]));
  }

  /** Where `id` was drawn on the immediately preceding board, if visible then. */
  seenBefore(id: string): SeenCard | null {
    return this.previousById.get(id) ?? null;
  }

  /** Most recent known board rect for `id` during this session, if any. */
  knownPosition(id: string): SeenCard | null {
    return this.previousById.get(id) ?? this.known.get(id) ?? null;
  }

  /**
   * Animates every card from its snapshot position, or from its source card /
   * owner's plate, to where it is now. Unfinished travels are cancelled first:
   * with nothing to animate (reduced motion, no snapshot) the cards are home.
   */
  play(root: HTMLElement | null, reduced: boolean): void {
    const before = this.seen;
    this.seen = [];
    if (!root) return;
    const now = cards(root);
    for (const element of now) {
      for (const animation of element.getAnimations()) if (animation.id === TRAVEL) animation.cancel();
    }
    if (reduced || !before.length) return;
    const present = new Set(now.map(element => element.dataset.travelId!));
    const byId = new Map(before.map(seen => [seen.id, seen]));
    const nowRects = new Map<string, DOMRect>();
    for (const element of now) {
      const rect = element.getBoundingClientRect();
      if (visible(rect)) nowRects.set(element.dataset.travelId!, rect);
    }
    // Cards that left the board this frame, by name, for new cards to come from.
    const departed = new Map<string, SeenCard[]>();
    for (const seen of before) {
      if (present.has(seen.id)) continue;
      departed.set(seen.name, [...(departed.get(seen.name) ?? []), seen]);
    }
    const plates = new Map([...root.querySelectorAll<HTMLElement>('[data-plate-side]')]
      .map(plate => [plate.dataset.plateSide!, plate.getBoundingClientRect()]));
    for (const element of now) {
      const rect = nowRects.get(element.dataset.travelId!);
      if (!rect) continue;
      const id = element.dataset.travelId!;
      const stayed = byId.get(id);
      if (stayed) {
        if (Math.abs(stayed.rect.left - rect.left) > 1 || Math.abs(stayed.rect.top - rect.top) > 1) {
          move(element, stayed.rect, rect, MOVE_MS, false);
        }
        continue;
      }
      const sourceId = element.dataset.travelSource;
      const sourceRect = sourceId ? (byId.get(sourceId)?.rect ?? nowRects.get(sourceId)) : undefined;
      if (sourceRect) {
        move(element, sourceRect, rect, ARRIVE_MS, false);
        continue;
      }
      const source = departed.get(element.dataset.travelName ?? '')?.shift();
      const plate = plates.get(element.dataset.travelSide ?? '');
      if (source) move(element, source.rect, rect, ARRIVE_MS, false);
      else if (plate && element.dataset.travelSide === 'opponent') move(element, plate, rect, ARRIVE_MS, true);
      else element.animate([{ opacity: 0 }, { opacity: 1 }], { id: TRAVEL, duration: ARRIVE_MS, easing: 'ease-out' });
    }
  }
}

/**
 * Build death marks (travel ghost + rule chip) for creatures dying on the
 * active cursor stop (or since the last decision when staying on now under
 * reduced motion).
 */
export function collectDeathMarks(params: {
  root: HTMLElement | null;
  boardEl: HTMLElement | null;
  travel: Travel;
  projection: Observation;
  events: readonly PresentationEvent[];
  labelsAt: (index: number) => PresentationLabels;
  activeSeqs: ReadonlySet<number>;
  cursor: number | null;
  sinceRevision: number | null;
  matchId: string;
  revision: number;
  reduced: boolean;
}): DeathMark[] {
  const { root, boardEl, travel, projection, events, labelsAt, activeSeqs, cursor, sinceRevision, matchId, revision, reduced } = params;
  const relevant: { event: PresentationEvent; index: number }[] = [];
  if (cursor !== null) {
    events.forEach((event, index) => {
      if (event.kind.kind === 'died' && activeSeqs.has(event.seq)) relevant.push({ event, index });
    });
  } else if (reduced && sinceRevision !== null) {
    events.forEach((event, index) => {
      if (event.kind.kind === 'died' && event.from_revision >= sinceRevision) relevant.push({ event, index });
    });
  }
  if (!relevant.length || !boardEl || !root) return [];

  const onBoard = new Set([...projection.agent.battlefield, ...projection.opponent.battlefield].map(p => p.id));
  const boardRect = boardEl.getBoundingClientRect();
  const marks: DeathMark[] = [];
  for (const { event, index } of relevant) {
    if (event.kind.kind !== 'died') continue;
    const labels = labelsAt(index);
    const rule = event.rule ? tellRule(event.rule) : null;
    for (const obj of event.kind.objects) {
      if (onBoard.has(obj.entity)) continue;
      const travelKey = `perm:${obj.entity}`;
      const before = travel.seenBefore(travelKey);
      const known = travel.knownPosition(travelKey);
      const side: 'you' | 'opponent' = known?.side
        ?? (projection.agent.graveyard.some(c => c.id === obj.entity) ? 'you' : 'opponent');
      const fallbackRow = root.querySelector<HTMLElement>(`.half.${side} .row.creatures`)?.getBoundingClientRect();
      const rect = known?.rect ?? fallbackRow;
      if (!rect || rect.width <= 0 || rect.height <= 0) continue;
      const x = known ? rect.left - boardRect.left : rect.left - boardRect.left + rect.width / 2 - 44;
      const y = known ? rect.top - boardRect.top : rect.top - boardRect.top + rect.height / 2 - 20;
      const width = known ? rect.width : 88;
      const height = known ? rect.height : 40;
      const gyRect = root.querySelector<HTMLElement>(`[data-graveyard-side="${side}"]`)?.getBoundingClientRect();
      const toDx = gyRect ? (gyRect.left + gyRect.width / 2) - (boardRect.left + x + width / 2) : 0;
      const toDy = gyRect ? (gyRect.top + gyRect.height / 2) - (boardRect.top + y + height / 2) : 0;
      marks.push({
        entity: obj.entity,
        name: eventObjectName(event, obj, labels) ?? known?.name ?? 'Creature',
        rule,
        side,
        x,
        y,
        width,
        height,
        toDx,
        toDy,
        key: `${matchId}:${cursor ?? revision}:${event.seq}:${obj.entity}`,
        animate: !reduced && before !== null,
      });
    }
  }
  return marks;
}

/** Animate a dying card ghost from its last battlefield slot to its owner's graveyard count. */
export function animateDeathTravel(element: HTMLElement, mark: DeathMark): Animation {
  return element.animate(
    [
      { translate: '0px 0px', scale: '1', opacity: 1 },
      { translate: `${mark.toDx}px ${mark.toDy}px`, scale: '0.35', opacity: 0 },
    ],
    { id: 'death-travel', duration: DEATH_MS, easing: EASE, fill: 'forwards' },
  );
}

function move(element: HTMLElement, from: DOMRect, to: DOMRect, duration: number, fade: boolean): void {
  const dx = from.left + from.width / 2 - (to.left + to.width / 2);
  const dy = from.top + from.height / 2 - (to.top + to.height / 2);
  const scale = Math.max(0.3, Math.min(3, Math.max(from.width, from.height) / Math.max(to.width, to.height)));
  element.animate([
    { translate: `${dx}px ${dy}px`, scale: `${scale}`, opacity: fade ? 0.2 : 1 },
    { translate: '0 0', scale: '1', opacity: 1 },
  ], { id: TRAVEL, duration, easing: EASE });
}

/** Whether the person asked for reduced motion; read once per frame. */
export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
