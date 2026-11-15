import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import SearchControl from './SearchControl.svelte';
import SearchThinking from './SearchThinking.svelte';
import { parseMirrorSearches, type MirrorSearchSpec } from '$lib/watch';

const spec: MirrorSearchSpec = { algorithm: 'local_update', belief: 'physical_deal',
  worlds: 4, depth: 2, alpha: 0.02, beta: 1, decision_seconds: 30 };

describe('search setup and thinking', () => {
  it('shows missing capability without synthesizing a searchable choice', () => {
    const { body } = render(SearchControl, { props: { value: null, supported: false,
      reason: 'This action-only policy has no search evaluator.', onChange: () => {} } });
    expect(body).toContain('None (raw policy)');
    expect(body).toContain('value="mirror" disabled');
    expect(body).toContain('This action-only policy has no search evaluator.');
  });
  it('shows the actual budget, physical belief and explicit retry behavior', () => {
    const { body } = render(SearchControl, { props: { value: spec, supported: true, onChange: () => {} } });
    expect(body).toContain('value="4"');
    expect(body).toContain('value="2"');
    expect(body).toContain('Uniform belief (physical deal)');
    expect(body).toContain('explicit retry');
    expect(parseMirrorSearches([spec, null])).toEqual([spec, null]);
    expect(() => parseMirrorSearches([{ ...spec, worlds: 0 }, null])).toThrow('Invalid');
  });
  it('identifies waiting time without showing private search outputs', () => {
    const { body } = render(SearchThinking, { props: { active: true, budget: 'Mirror search 4×2' } });
    expect(body).toContain('role="status"');
    expect(body).toContain('Mirror search 4×2');
    expect(body).toContain('0.0 s');
  });
});
