/**
 * Pure drafting helpers for a live `partition_piles` group (e.g., Curator of
 * Destinies).
 *
 * The authority supplies the `PartitionPilesDecisionGroup` descriptor (`items`,
 * `committed_face_up`, `committed_face_down`), validates the drafted split via
 * `/decision-group-preview`, and executes all sequential steps atomically on one
 * `decision_group_command`. Draft persistence uses the unified `academy.table.decision-group-draft`
 * store in `decision-group.ts`.
 */
import { sanitizeEntityIds, type DecisionCard, type PartitionPilesDecisionGroup, type PartitionPilesPlan } from '$lib/decision-group';

export interface PartitionWords {
  /** Primary button label: "Confirm split (2 up, 3 down)". */
  text: string;
  /** Prompt-line summary of the drafted split. */
  plan: string;
}

/** Keep only valid, distinct remaining entity ids from a restored draft. */
export function sanitizePartitionDraft(
  saved: readonly number[] | null | undefined,
  group: PartitionPilesDecisionGroup,
): number[] {
  return sanitizeEntityIds(saved, new Set(group.items.map(card => card.item.entity)));
}

/** Toggle one card between the face-up and face-down piles in the draft. */
export function toggleCardPile(faceUp: readonly number[], cardId: number): number[] {
  return faceUp.includes(cardId) ? faceUp.filter(id => id !== cardId) : [...faceUp, cardId];
}

/** Swap every undecided card between the face-up and face-down piles. */
export function swapPiles(items: readonly DecisionCard[], faceUp: readonly number[]): number[] {
  const up = new Set(faceUp);
  return items.filter(card => !up.has(card.item.entity)).map(card => card.item.entity);
}

/** Convert the drafted face-up entity ids into a wire `PartitionPilesPlan`. */
export function partitionPlanOf(
  faceUp: readonly number[],
  group: PartitionPilesDecisionGroup,
): PartitionPilesPlan {
  const up = new Set(faceUp);
  return {
    kind: 'partition_piles',
    face_up: group.items.filter(card => up.has(card.item.entity)).map(card => card.item),
  };
}

function counted(names: string[]): string[] {
  const counts = new Map<string, number>();
  for (const name of names) counts.set(name, (counts.get(name) ?? 0) + 1);
  return [...counts].map(([name, count]) => (count > 1 ? `${name} ×${count}` : name));
}

export function partitionWords(
  group: PartitionPilesDecisionGroup,
  faceUp: readonly number[],
): PartitionWords {
  const upSet = new Set(faceUp);
  const upCards = group.items.filter(card => upSet.has(card.item.entity));
  const upCount = upCards.length;
  const downCount = group.items.length - upCount;
  const committedCount = group.committed_face_up.length + group.committed_face_down.length;
  const verb = committedCount > 0 ? 'Confirm rest' : 'Confirm split';
  const text = `${verb} (${upCount} up, ${downCount} down)`;
  if (upCount === 0) return { text, plan: `0 face-up · ${downCount} face-down` };
  if (downCount === 0) return { text, plan: `All ${upCount} face-up · 0 face-down` };
  const upNames = counted(upCards.map(card => card.label));
  const summary = upNames.length <= 3 ? ` (${upNames.join(', ')})` : '';
  return { text, plan: `${upCount} face-up${summary} · ${downCount} face-down` };
}
