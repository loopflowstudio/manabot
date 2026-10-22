import { describe, expect, it } from 'vitest';

import { REACHABLE_PROMPT_FAMILIES } from '../../e2e/experience-proof-baseline';
import { DECISION_PROMPTS } from './prompt-instructions';

describe('release prompt instructions', () => {
  it('describes every reachable selected-matchup prompt family', () => {
    for (const family of REACHABLE_PROMPT_FAMILIES) {
      expect(DECISION_PROMPTS[family]?.trim(), family).toBeTruthy();
    }
  });
});
