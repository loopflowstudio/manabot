import type { ActionOption, BoardActionTarget } from './types';

export function buildClickableTargets(
  actions: ActionOption[],
): Map<number, BoardActionTarget> {
  const targets = new Map<number, BoardActionTarget>();

  for (const action of actions) {
    for (const objectId of action.focus) {
      const existing = targets.get(objectId);
      if (existing) {
        existing.actionIndexes.push(action.index);
        continue;
      }
      targets.set(objectId, { objectId, actionIndexes: [action.index] });
    }
  }

  return targets;
}

export function actionIndexesForTarget(
  targets: Map<number, BoardActionTarget>,
  objectId: number,
): number[] {
  return targets.get(objectId)?.actionIndexes ?? [];
}

export function filterActionsForTarget(
  actions: ActionOption[],
  targets: Map<number, BoardActionTarget>,
  objectId: number | null,
): ActionOption[] {
  if (objectId === null) {
    return actions;
  }

  const allowed = new Set(actionIndexesForTarget(targets, objectId));
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
