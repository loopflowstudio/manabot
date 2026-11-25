/**
 * Pure draft model for library arrangement groups (Scry, Surveil, OrderLibrary).
 *
 * Driven strictly by the authority's `ArrangeGroup` descriptor (`items` and
 * `committed*` lists) — no prompt-text or label scraping.
 */
import type {
  ArrangeGroup,
  OrderLibraryPlan,
  ScryPlan,
  SurveilPlan,
} from '$lib/group';
import type { ObjectRenderId } from '$lib/types';

export interface ScryDraft {
  kind: 'scry';
  /** Drawn-first at index 0. */
  top: number[];
  /** Top-most of the bottom group at index 0. */
  bottom: number[];
}

export interface SurveilDraft {
  kind: 'surveil';
  /** Cards marked for graveyard; remaining looked-at cards stay on top. */
  graveyard: number[];
}

export interface OrderLibraryDraft {
  kind: 'order_library';
  /** Drawn-first at index 0. */
  order: number[];
}

export type ArrangeDraft = ScryDraft | SurveilDraft | OrderLibraryDraft;

export function emptyArrangeDraft(group: ArrangeGroup): ArrangeDraft {
  switch (group.kind) {
    case 'scry':
      return { kind: 'scry', top: [], bottom: [] };
    case 'surveil':
      return { kind: 'surveil', graveyard: [] };
    case 'order_library':
      return { kind: 'order_library', order: [] };
  }
}

export function sanitizeArrangeDraft(saved: unknown, group: ArrangeGroup): ArrangeDraft {
  if (!saved || typeof saved !== 'object') return emptyArrangeDraft(group);
  const candidate = saved as Record<string, unknown>;
  if (candidate.kind !== group.kind) return emptyArrangeDraft(group);
  const allowed = new Set(group.items.map(card => card.item.entity));
  const clean = (raw: unknown, seen: Set<number>): number[] => {
    const result: number[] = [];
    for (const id of Array.isArray(raw) ? raw : []) {
      if (typeof id === 'number' && allowed.has(id) && !seen.has(id)) {
        seen.add(id);
        result.push(id);
      }
    }
    return result;
  };
  switch (group.kind) {
    case 'scry': {
      const seen = new Set<number>();
      return { kind: 'scry', top: clean(candidate.top, seen), bottom: clean(candidate.bottom, seen) };
    }
    case 'surveil':
      return { kind: 'surveil', graveyard: clean(candidate.graveyard, new Set()) };
    case 'order_library':
      return { kind: 'order_library', order: clean(candidate.order, new Set()) };
  }
}

function moveInList(items: readonly number[], id: number, delta: -1 | 1): number[] {
  const index = items.indexOf(id);
  const nextIndex = index + delta;
  if (index < 0 || nextIndex < 0 || nextIndex >= items.length) return [...items];
  const next = [...items];
  [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
  return next;
}

export function scryPlace(draft: ScryDraft, id: number, target: 'top' | 'bottom' | 'unplaced'): ScryDraft {
  const top = draft.top.filter(entry => entry !== id);
  const bottom = draft.bottom.filter(entry => entry !== id);
  if (target === 'top') top.push(id);
  else if (target === 'bottom') bottom.push(id);
  return { kind: 'scry', top, bottom };
}

export function scryMove(draft: ScryDraft, bucket: 'top' | 'bottom', id: number, delta: -1 | 1): ScryDraft {
  return bucket === 'top'
    ? { ...draft, top: moveInList(draft.top, id, delta) }
    : { ...draft, bottom: moveInList(draft.bottom, id, delta) };
}

export function surveilPlace(draft: SurveilDraft, id: number, target: 'keep' | 'graveyard'): SurveilDraft {
  const has = draft.graveyard.includes(id);
  if (target === 'graveyard') return has ? draft : { kind: 'surveil', graveyard: [...draft.graveyard, id] };
  return has ? { kind: 'surveil', graveyard: draft.graveyard.filter(entry => entry !== id) } : draft;
}

export function orderPlace(draft: OrderLibraryDraft, id: number, placed: boolean): OrderLibraryDraft {
  const has = draft.order.includes(id);
  if (placed) return has ? draft : { kind: 'order_library', order: [...draft.order, id] };
  return has ? { kind: 'order_library', order: draft.order.filter(entry => entry !== id) } : draft;
}

export function orderMove(draft: OrderLibraryDraft, id: number, delta: -1 | 1): OrderLibraryDraft {
  return { kind: 'order_library', order: moveInList(draft.order, id, delta) };
}

export function isArrangeComplete(draft: ArrangeDraft, group: ArrangeGroup): boolean {
  if (draft.kind !== group.kind) return false;
  switch (draft.kind) {
    case 'scry':
      return draft.top.length + draft.bottom.length === group.items.length;
    case 'surveil':
      return true;
    case 'order_library':
      return draft.order.length === group.items.length;
  }
}

export function arrangeHasChanges(draft: ArrangeDraft): boolean {
  switch (draft.kind) {
    case 'scry':
      return draft.top.length > 0 || draft.bottom.length > 0;
    case 'surveil':
      return draft.graveyard.length > 0;
    case 'order_library':
      return draft.order.length > 0;
  }
}

export function arrangePlanOf(
  draft: ArrangeDraft,
  group: ArrangeGroup,
): ScryPlan | SurveilPlan | OrderLibraryPlan {
  const ids = new Map<number, ObjectRenderId>(group.items.map(card => [card.item.entity, card.item]));
  const id = (entity: number): ObjectRenderId => ids.get(entity) ?? { entity, incarnation: 0 };
  if (draft.kind === 'scry') {
    return { kind: 'scry', top: draft.top.map(id), bottom: draft.bottom.map(id) };
  }
  if (draft.kind === 'surveil') {
    return { kind: 'surveil', graveyard: draft.graveyard.map(id) };
  }
  return { kind: 'order_library', order: draft.order.map(id) };
}

export interface ArrangeWords {
  title: string;
  confirmLabel: string;
  planLine: string;
  bucketTitle?: string;
}

export function arrangeWords(draft: ArrangeDraft, group: ArrangeGroup): ArrangeWords {
  const total = group.items.length;
  if (group.kind === 'scry' && draft.kind === 'scry') {
    const hasCommitted = group.committed_top.length > 0 || group.committed_bottom.length > 0;
    const title = hasCommitted ? `Scry · ${total} left` : `Scry ${total}`;
    const unplaced = total - draft.top.length - draft.bottom.length;
    if (unplaced > 0) {
      const text = `Place ${unplaced} more ${unplaced === 1 ? 'card' : 'cards'}`;
      return { title, confirmLabel: text, planLine: `${title} · ${text}` };
    }
    const confirmLabel = draft.bottom.length === 0
      ? `Put all ${draft.top.length} on top`
      : draft.top.length === 0
        ? `Put all ${draft.bottom.length} on bottom`
        : `Confirm (${draft.top.length} top, ${draft.bottom.length} bottom)`;
    return { title, confirmLabel, planLine: `${title} · ${confirmLabel}` };
  }
  if (group.kind === 'surveil' && draft.kind === 'surveil') {
    const hasCommitted = group.committed_graveyard.length > 0;
    const title = hasCommitted ? `Surveil · ${total} left` : `Surveil ${total}`;
    const graveyard = draft.graveyard.length;
    const kept = total - graveyard;
    const confirmLabel = graveyard === 0
      ? `Keep all ${total} on top`
      : kept === 0
        ? `Put all ${total} in graveyard`
        : `Put ${graveyard} in graveyard, keep ${kept} on top`;
    return { title, confirmLabel, planLine: `${title} · ${confirmLabel}` };
  }
  if (group.kind === 'order_library' && draft.kind === 'order_library') {
    const title = group.end === 'top' ? 'Order top of library' : 'Order bottom of library';
    const bucketTitle = group.end === 'top'
      ? 'Top of library — drawn first at top'
      : 'Bottom of library — drawn first at top';
    const unplaced = total - draft.order.length;
    if (unplaced > 0) {
      const text = `Order ${unplaced} more ${unplaced === 1 ? 'card' : 'cards'}`;
      return { title, bucketTitle, confirmLabel: text, planLine: `${title} · ${text}` };
    }
    const confirmLabel = group.end === 'top' ? 'Confirm top order' : 'Confirm bottom order';
    return { title, bucketTitle, confirmLabel, planLine: `${title} · ${confirmLabel}` };
  }
  return { title: 'Arrange cards', confirmLabel: 'Confirm', planLine: 'Arrange cards' };
}
