/**
 * A whole pile-partition decision (e.g., Curator of Destinies), drafted before
 * any per-card choice is sent.
 *
 * The engine asks about one inspected card at a time (face-up or face-down).
 * Academy shows all remaining inspected cards together so a human can split
 * them holistically, then on confirm sends the engine's exact per-card offers
 * one by one. `partitionOffer` verifies each subsequent frame asks about the
 * expected card in `partition_piles`; any other frame stops the sequence.
 */
import type { CardState, ExperienceFrame, InteractionOffer } from '$lib/types';

export type PileSide = 'face_up' | 'face_down';

export interface PartitionStep {
  cardId: number;
  cardName: string;
  pile: PileSide;
}

export interface PartitionContext {
  sourceName: string;
  /** Every inspected card for this partition, in the engine's decision order. */
  cards: CardState[];
  /** Cards already committed to the engine (before the current cursor). */
  committed: CardState[];
  /** Undecided cards starting at the current cursor, in engine ask order. */
  remaining: CardState[];
  currentId: number;
}

export interface PartitionWords {
  /** Primary button label: "Confirm split (2 up, 3 down)". */
  text: string;
  /** Prompt-line summary of the drafted split. */
  plan: string;
}

const FACE_UP_RE = /\bin the face-up pile\b/i;
const FACE_DOWN_RE = /\bin the face-down pile\b/i;

function focusedCardId(offer: InteractionOffer): number | null {
  const fromSubject = offer.subjects?.[0]?.object.display_id;
  if (typeof fromSubject === 'number') return fromSubject;
  return typeof offer.focus[0] === 'number' ? offer.focus[0] : null;
}

function pileOffers(frame: ExperienceFrame): { faceUp: InteractionOffer; faceDown: InteractionOffer; cardId: number } | null {
  if (frame.action_space !== 'MODAL' || frame.projection.decision_context?.stage !== 'partition_piles') return null;
  const faceUp = frame.offers.find(offer => FACE_UP_RE.test(offer.label));
  const faceDown = frame.offers.find(offer => FACE_DOWN_RE.test(offer.label));
  if (!faceUp || !faceDown) return null;
  const upId = focusedCardId(faceUp);
  const downId = focusedCardId(faceDown);
  if (upId === null || upId !== downId) return null;
  return { faceUp, faceDown, cardId: upId };
}

export function isPartitionPiles(frame: ExperienceFrame): boolean {
  return pileOffers(frame) !== null;
}

/**
 * Derive the holistic partition state from one authority frame, or null when
 * the frame is not a valid `partition_piles` decision with inspected cards.
 */
export function partitionContext(frame: ExperienceFrame): PartitionContext | null {
  const offers = pileOffers(frame);
  if (!offers) return null;
  const inspected = frame.projection.agent.inspected ?? [];
  const current = inspected.find(card => card.id === offers.cardId);
  if (!current) return null;
  const cards = inspected.filter(card => card.owner_id === current.owner_id);
  const cursor = cards.findIndex(card => card.id === offers.cardId);
  if (cursor < 0) return null;
  return {
    sourceName: frame.projection.decision_context?.source?.label ?? 'Split into two piles',
    cards,
    committed: cards.slice(0, cursor),
    remaining: cards.slice(cursor),
    currentId: offers.cardId,
  };
}

/** Toggle one card between the face-up and face-down piles in the draft. */
export function toggleCardPile(faceUp: readonly number[], cardId: number): number[] {
  return faceUp.includes(cardId) ? faceUp.filter(id => id !== cardId) : [...faceUp, cardId];
}

/** Swap every undecided card between the face-up and face-down piles. */
export function swapPiles(remaining: readonly CardState[], faceUp: readonly number[]): number[] {
  const up = new Set(faceUp);
  return remaining.filter(card => !up.has(card.id)).map(card => card.id);
}

/** Ordered per-card offers to submit on confirm, matching the engine's cursor order. */
export function partitionSteps(remaining: readonly CardState[], faceUp: readonly number[]): PartitionStep[] {
  const up = new Set(faceUp);
  return remaining.map(card => ({
    cardId: card.id,
    cardName: card.name,
    pile: up.has(card.id) ? 'face_up' : 'face_down',
  }));
}

/**
 * The authority offer answering `step` in `frame`, or null when `frame` is not
 * the expected `partition_piles` prompt for `step.cardId`.
 */
export function partitionOffer(frame: ExperienceFrame, step: PartitionStep): InteractionOffer | null {
  const offers = pileOffers(frame);
  if (!offers || offers.cardId !== step.cardId) return null;
  return step.pile === 'face_up' ? offers.faceUp : offers.faceDown;
}

function counted(names: string[]): string[] {
  const counts = new Map<string, number>();
  for (const name of names) counts.set(name, (counts.get(name) ?? 0) + 1);
  return [...counts].map(([name, count]) => (count > 1 ? `${name} ×${count}` : name));
}

export function partitionWords(remaining: readonly CardState[], faceUp: readonly number[], committedCount: number): PartitionWords {
  const upSet = new Set(faceUp);
  const upCards = remaining.filter(card => upSet.has(card.id));
  const upCount = upCards.length;
  const downCount = remaining.length - upCount;
  const verb = committedCount > 0 ? 'Confirm rest' : 'Confirm split';
  const text = `${verb} (${upCount} up, ${downCount} down)`;
  if (upCount === 0) return { text, plan: `0 face-up · ${downCount} face-down` };
  if (downCount === 0) return { text, plan: `All ${upCount} face-up · 0 face-down` };
  const upNames = counted(upCards.map(card => card.name));
  const summary = upNames.length <= 3 ? ` (${upNames.join(', ')})` : '';
  return { text, plan: `${upCount} face-up${summary} · ${downCount} face-down` };
}

// ———————————————————————————————————————— Reload

const DRAFT_KEY = 'academy.table.partition-draft';

/** A draft survives a reload of the same decision in this tab, and no other. */
export function savePartitionDraft(frame: ExperienceFrame, faceUp: readonly number[]): void {
  try {
    sessionStorage.setItem(DRAFT_KEY, JSON.stringify({ match_id: frame.match_id, frame_hash: frame.frame_hash, faceUp }));
  } catch { /* A draft is presentation; losing it loses nothing committed. */ }
}

export function loadPartitionDraft(frame: ExperienceFrame, validIds: ReadonlySet<number>): number[] | null {
  try {
    const saved = JSON.parse(sessionStorage.getItem(DRAFT_KEY) ?? 'null');
    if (!saved || saved.match_id !== frame.match_id || saved.frame_hash !== frame.frame_hash || !Array.isArray(saved.faceUp)) {
      return null;
    }
    return (saved.faceUp as unknown[]).filter((id): id is number => typeof id === 'number' && validIds.has(id));
  } catch {
    return null;
  }
}
