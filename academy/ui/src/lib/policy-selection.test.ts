import { expect, it } from 'vitest';
import { runnerReplacement } from './policy-selection';
import type { PolicyRuntimeIdentity, TrainedOpponent } from './types';

const old: PolicyRuntimeIdentity = {
  id: 'old', provider: 'example', runner: 'v1', artifacts: { actor: 'abc' },
  config: { nested: { a: 1, b: 2 } }, supports_search: false,
};
const entry = (changes: Partial<PolicyRuntimeIdentity> = {}): TrainedOpponent => ({
  sha256: 'new', name: 'Current', producer: 'example', group: 'example', details: [],
  available: true, reason: null,
  policy: { ...old, id: 'new', runner: 'v2', ...changes },
});

it('offers exactly one matching execution with a changed runner', () => {
  const current = entry({ config: { nested: { b: 2, a: 1 } } });
  expect(runnerReplacement(old, [current])).toBe(current);
  expect(runnerReplacement(old, [current, entry({ id: 'another' })])).toBeNull();
});

it('does not suggest a changed artifact, provider, config or contract', () => {
  for (const change of [
    { artifacts: { actor: 'different' } }, { provider: 'other' },
    { config: {} }, { contract_version: 2 }, { runner: 'v1' },
  ]) expect(runnerReplacement(old, [entry(change)])).toBeNull();
});
