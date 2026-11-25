import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import DecklistsSelector from './DecklistsSelector.svelte';
import PlayerPicker from './PlayerPicker.svelte';

function optionTag(body: string, value: string): string {
  const match = body.match(new RegExp(`<option[^>]*value="${value}"[^>]*>[^<]*`));
  if (!match) throw new Error(`no option ${value}`);
  return match[0];
}

it('omits standalone rollout Search opponents from the default picker while retaining legacy selections', () => {
  const { body } = render(PlayerPicker, {
    props: { value: 'random', decklists: 'closed' },
  });
  expect(body).not.toContain('value="search-64"');
  for (const value of ['policy', 'random', 'passive']) {
    expect(optionTag(body, value)).not.toContain('disabled');
  }
  const legacyClosed = render(PlayerPicker, {
    props: { value: 'search-64', decklists: 'closed' },
  }).body;
  expect(optionTag(legacyClosed, 'search-64')).toContain('disabled');
  expect(optionTag(legacyClosed, 'search-64')).toContain('— needs open decklists');
});

it('retains a legacy search selection enabled on open decklists', () => {
  const { body } = render(PlayerPicker, { props: { value: 'search-64', decklists: 'open' } });
  const option = optionTag(body, 'search-64');
  expect(option).not.toContain('disabled');
  expect(option).not.toContain('needs open decklists');
});

it('a search opponent disables closed decklists with the reason', () => {
  const { body } = render(DecklistsSelector, {
    props: { value: 'open', opponent: 'search-64' },
  });
  const closed = optionTag(body, 'closed');
  expect(closed).toContain('disabled');
  expect(closed).toContain('— search opponents need open decklists');
});

it('a non-search opponent offers closed decklists', () => {
  for (const opponent of ['random', 'passive', 'policy:abc'] as const) {
    const { body } = render(DecklistsSelector, { props: { value: 'open', opponent } });
    expect(optionTag(body, 'closed')).not.toContain('disabled');
  }
});

it('names a missing bot extension only after the catalog answers', () => {
  const note = (extension: string | null | undefined): string =>
    render(PlayerPicker, { props: { extension } }).body;
  expect(note(undefined)).not.toContain('Trained bots need manabot');
  expect(note(undefined)).not.toContain('No trained checkpoints found');
  expect(note(null)).toContain('No policies registered on this server.');
  expect(note('manabot')).toContain('No policies registered on this server.');
});

it('renders a third-party group and details without training terminology', () => {
  const { body } = render(PlayerPicker, { props: {
    value: 'policy:abc', trained: [{ sha256: 'abc', name: 'Language player',
      producer: 'example', group: 'Text agents', available: true, reason: null,
      details: [{ label: 'Context', value: '8192 tokens' }] }],
  } });
  expect(body).toContain('Text agents');
  expect(body).toContain('Context: 8192 tokens');
  expect(body).toContain('Policy ID: abc');
  expect(body).not.toContain('transitions');
  expect(body).not.toContain('update');
});
