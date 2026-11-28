import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import SearchControl from './SearchControl.svelte';
import { parseSearchConfigs, type SearchConfig } from '$lib/search';

const spec: SearchConfig = { algorithm: 'local_update', belief: 'physical_deal',
  worlds: 4, depth: 20, alpha: 0.02, beta: 0.01, decision_seconds: 30 };

describe('search setup and thinking', () => {
  it('shows missing capability without synthesizing a searchable choice', () => {
    const { body } = render(SearchControl, { props: { value: null, supported: false,
      reason: 'This action-only policy has no search evaluator.', onChange: () => {} } });
    expect(body).toContain('None (raw policy)');
    expect(body).toContain('value="quick" disabled');
    expect(body).toContain('value="standard" disabled');
    expect(body).toContain('value="strong" disabled');
    expect(body).toContain('value="custom" disabled');
    expect(body).toContain('This action-only policy has no search evaluator.');
  });
  it('hides custom budget inputs for named presets and shows them for custom budgets', () => {
    const presetBody = render(SearchControl, { props: { value: spec, supported: true, onChange: () => {} } }).body;
    expect(presetBody).not.toContain('type="number"');
    expect(presetBody).toContain('Uniform belief (physical deal)');
    expect(presetBody).toContain('β = 0.01');
    expect(presetBody).toContain('explicit retry');

    const customSpec: SearchConfig = { ...spec, worlds: 1, depth: 1 };
    const customBody = render(SearchControl, { props: { value: customSpec, supported: true, onChange: () => {} } }).body;
    expect(customBody).toContain('value="1"');
    expect(parseSearchConfigs([spec, null])).toEqual([spec, null]);
    expect(() => parseSearchConfigs([{ ...spec, worlds: 0 }, null])).toThrow('Invalid');
  });

});
