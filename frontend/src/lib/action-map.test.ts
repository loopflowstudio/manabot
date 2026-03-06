import { describe, expect, it } from 'vitest';

import {
  buildClickableTargets,
  filterActionsForTarget,
  focusIdsForActionIndexes,
} from './action-map';
import type { ActionOption } from './types';

const actions: ActionOption[] = [
  {
    index: 0,
    type: 'PRIORITY_CAST_SPELL',
    focus: [101],
    description: 'Cast spell: Grey Ogre',
  },
  {
    index: 1,
    type: 'PRIORITY_PLAY_LAND',
    focus: [102],
    description: 'Play land: Mountain',
  },
  {
    index: 2,
    type: 'DECLARE_ATTACKER',
    focus: [201, 301],
    description: 'Declare attacker: Grey Ogre',
  },
];

describe('action-map helpers', () => {
  it('builds clickable target maps for focused objects', () => {
    const targets = buildClickableTargets(actions);

    expect(targets.get(101)).toEqual([0]);
    expect(targets.get(201)).toEqual([2]);
    expect(targets.get(999)).toBeUndefined();
  });

  it('filters actions when a board target is selected', () => {
    const targets = buildClickableTargets(actions);

    expect(filterActionsForTarget(actions, targets, 101).map((action) => action.index)).toEqual([0]);
    expect(filterActionsForTarget(actions, targets, null).map((action) => action.index)).toEqual([0, 1, 2]);
  });

  it('collects focus ids for a subset of actions', () => {
    expect(focusIdsForActionIndexes(actions, [2]).sort((left, right) => left - right)).toEqual([201, 301]);
  });
});
