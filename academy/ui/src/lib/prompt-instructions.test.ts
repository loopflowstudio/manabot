import { describe, expect, it } from 'vitest';

import matrix from '../../e2e/release-prompt-matrix.json';
import { DECISION_PROMPTS } from './prompt-instructions';

describe('release prompt instructions', () => {
  it('describes every reachable selected-matchup prompt family', () => {
    for (const family of matrix.action_spaces.reachable) {
      expect(DECISION_PROMPTS[family]?.trim(), family).toBeTruthy();
    }
  });
});
