/**
 * Pointer drags on the board, as one revision-bound gesture.
 *
 * A drag adds no capability: every drop ends where a click already goes.
 * Dropping a card from hand on the battlefield is clicking it (one offer
 * commits, several open its choices, at the drop point). A connection places
 * a draft block or activates the offers on its recipient. A marquee adds the
 * creatures it touches to the attack draft. This module is the
 * gesture's lifecycle, kept free of the DOM: `Board.svelte` reads the
 * pointer and what is under it, and acts on what `release` returns.
 *
 * A press becomes a drag only once the pointer travels `DRAG_THRESHOLD_PX`;
 * before that the browser's own click runs unchanged. Each gesture belongs
 * to one pointer (`pointerId`) and to the frame revision it started at: a new
 * revision ends it, and the board draws a gesture only at its own revision.
 * `holding` keeps a dropped card where it was released while the Command is
 * pending or its choices are open; cancelling a gesture never ends a hold,
 * since the commit itself makes the board read-only for a moment.
 */
import type { CardState } from '$lib/types';
import type { Endpoint } from './recent';
import type { PlacedOffer } from './view';

export interface Point {
  x: number;
  y: number;
}

/** How far, px, a press travels before it is a drag rather than a click. */
export const DRAG_THRESHOLD_PX = 6;

/** A card in hand carried toward the battlefield to play or cast it. */
export interface PlayDrag {
  kind: 'play';
  card: CardState;
  /** Every offer on the card, as a click on it sees them. */
  offers: PlacedOffer[];
  /** The card's width in hand, px: the carried card is drawn at it. */
  width: number;
  /** Where the card was gripped, from its top-left corner. */
  grab: Point;
  /** The hand card's own control: focus returns to it, and a cancelled drag settles back on it. */
  opener: HTMLElement;
  /** The pointer is over the battlefield, where releasing plays the card. */
  overField: boolean;
}

/** A connection target that activates its own offers on release. */
export interface OfferTarget {
  endpoint: Endpoint;
  title: string;
  offers: PlacedOffer[];
  element: HTMLElement;
}

/** A line pulled from a card to a recipient: either placing a draft block, or choosing the recipient's offers. */
export type ConnectDrag =
  | { kind: 'connect'; from: { kind: 'object'; id: number }; style: 'block'; blocker: number; target: { endpoint: { kind: 'object'; id: number } } | null }
  | { kind: 'connect'; from: Endpoint; style: 'target' | 'block'; blocker?: undefined; target: OfferTarget | null };

type Drag = PlayDrag | ConnectDrag | { kind: 'marquee' };

type Bounds = Pick<DOMRect, 'left' | 'right' | 'top' | 'bottom'>;

/** Viewport bounds of a sweep in either direction, including a horizontal or vertical stroke. */
export function marqueeBounds(start: Point, pointer: Point): Bounds {
  return {
    left: Math.min(start.x, pointer.x), right: Math.max(start.x, pointer.x),
    top: Math.min(start.y, pointer.y), bottom: Math.max(start.y, pointer.y),
  };
}

/** The shared area, including contact along a single edge. */
export function overlap(a: Bounds, b: Bounds): Bounds | null {
  const left = Math.max(a.left, b.left), right = Math.min(a.right, b.right);
  const top = Math.max(a.top, b.top), bottom = Math.min(a.bottom, b.bottom);
  return left <= right && top <= bottom ? { left, right, top, bottom } : null;
}

export type Gesture =
  | { phase: 'idle' }
  | { phase: 'pressed'; pointerId: number; revision: number; start: Point; drag: Drag }
  | { phase: 'dragging'; pointerId: number; revision: number; start: Point; pointer: Point; drag: Drag }
  /** Released on the battlefield: the Command is pending (`pending`) or the card's choices are open (`menu`). */
  | { phase: 'holding'; revision: number; at: Point; drag: PlayDrag; reason: 'pending' | 'menu' };

export const IDLE: Gesture = { phase: 'idle' };

/** What a release does: nothing of its own (`click`: the browser's click follows), play, connect, marquee, or settle back. */
export type Release =
  | { outcome: 'click' }
  | { outcome: 'none' }
  | { outcome: 'marquee' }
  | { outcome: 'connect'; drag: ConnectDrag }
  | { outcome: 'play'; at: Point; drag: PlayDrag }
  | { outcome: 'cancel'; at: Point; drag: PlayDrag };

/** A hand card starts a drag only when it can be played or cast; discards and card choices stay clicks. */
export function playable(offers: readonly PlacedOffer[]): boolean {
  return offers.some(placed => placed.offer.verb === 'cast' || placed.offer.verb === 'play_land');
}

export function press(drag: Drag, pointerId: number, revision: number, start: Point): Gesture {
  return { phase: 'pressed', pointerId, revision, start, drag };
}

/** The pointer moved: a press past the threshold becomes a drag. Other pointers are ignored. */
export function move(gesture: Gesture, pointerId: number, pointer: Point, locate: (drag: Drag) => Drag): Gesture {
  if ((gesture.phase !== 'pressed' && gesture.phase !== 'dragging') || gesture.pointerId !== pointerId) return gesture;
  if (gesture.phase === 'pressed' && Math.hypot(pointer.x - gesture.start.x, pointer.y - gesture.start.y) < DRAG_THRESHOLD_PX) return gesture;
  return { phase: 'dragging', pointerId, revision: gesture.revision, start: gesture.start, pointer, drag: locate(gesture.drag) };
}

/** The carried card's top-left corner, px, while it is drawn. */
export function carriedAt(gesture: Gesture): Point | null {
  if (gesture.phase === 'dragging' && gesture.drag.kind === 'play') return { x: gesture.pointer.x - gesture.drag.grab.x, y: gesture.pointer.y - gesture.drag.grab.y };
  if (gesture.phase === 'holding') return gesture.at;
  return null;
}

/** The pointer was released: the gesture that follows and what the board does. */
export function release(gesture: Gesture, pointerId: number): { gesture: Gesture; release: Release } {
  if (gesture.phase === 'pressed' && gesture.pointerId === pointerId) return { gesture: IDLE, release: { outcome: 'click' } };
  if (gesture.phase !== 'dragging' || gesture.pointerId !== pointerId) return { gesture, release: { outcome: 'none' } };
  const { drag, revision } = gesture;
  if (drag.kind === 'marquee') return { gesture: IDLE, release: { outcome: 'marquee' } };
  if (drag.kind === 'connect') return { gesture: IDLE, release: { outcome: 'connect', drag } };
  const at = carriedAt(gesture)!;
  if (!drag.overField || !drag.offers.length) return { gesture: IDLE, release: { outcome: 'cancel', at, drag } };
  const reason = drag.offers.length === 1 ? 'pending' : 'menu';
  return { gesture: { phase: 'holding', revision, at, drag, reason }, release: { outcome: 'play', at, drag } };
}

/** Escape, a lost pointer, a read-out or the board going read-only: an unreleased gesture ends; a hold stays. */
export function cancel(gesture: Gesture): Gesture {
  return gesture.phase === 'pressed' || gesture.phase === 'dragging' ? IDLE : gesture;
}

/** An offer was chosen from a dropped card's choices: its Command is now pending. */
export function promote(gesture: Gesture): Gesture {
  return gesture.phase === 'holding' && gesture.reason === 'menu' ? { ...gesture, reason: 'pending' } : gesture;
}

/** What releasing here does, in words. */
export function dropLabel(drag: PlayDrag): string {
  if (!drag.overField) return 'Release to cancel';
  return drag.offers.length === 1 ? drag.offers[0].offer.label : `Choose how to play ${drag.card.name}`;
}
