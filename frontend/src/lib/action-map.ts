import type { ActionOption } from './types';

export function buildClickableTargets(
  actions: ActionOption[],
): Map<number, number[]> {
  const targets = new Map<number, number[]>();

  for (const action of actions) {
    for (const objectId of action.focus) {
      const existing = targets.get(objectId);
      if (existing) {
        existing.push(action.index);
        continue;
      }
      targets.set(objectId, [action.index]);
    }
  }

  return targets;
}

export function filterActionsForTarget(
  actions: ActionOption[],
  targets: Map<number, number[]>,
  objectId: number | null,
): ActionOption[] {
  if (objectId === null) {
    return actions;
  }

  const allowed = new Set(targets.get(objectId) ?? []);
  return actions.filter((action) => allowed.has(action.index));
}

export function focusIdsForActionIndexes(
  actions: ActionOption[],
  actionIndexes: number[],
): number[] {
  const allowed = new Set(actionIndexes);
  const focusIds = new Set<number>();

  for (const action of actions) {
    if (!allowed.has(action.index)) {
      continue;
    }

    for (const focusId of action.focus) {
      focusIds.add(focusId);
    }
  }

  return [...focusIds];
}
