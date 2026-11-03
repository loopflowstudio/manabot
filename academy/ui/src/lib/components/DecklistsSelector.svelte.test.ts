import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import DecklistsSelector from './DecklistsSelector.svelte';
import OpponentSelector from './OpponentSelector.svelte';

function optionTag(body: string, value: string): string {
  const match = body.match(new RegExp(`<option[^>]*value="${value}"[^>]*>[^<]*`));
  if (!match) throw new Error(`no option ${value}`);
  return match[0];
}

it('closed decklists disable the search opponents with the reason', () => {
  const { body } = render(OpponentSelector, {
    props: { value: 'random', decklists: 'closed' },
  });
  for (const value of ['search-16', 'search-64', 'search-256']) {
    const option = optionTag(body, value);
    expect(option).toContain('disabled');
    expect(option).toContain('— needs open decklists');
  }
  for (const value of ['random', 'passive']) {
    expect(optionTag(body, value)).not.toContain('disabled');
  }
});

it('open decklists offer the search opponents', () => {
  const { body } = render(OpponentSelector, { props: { decklists: 'open' } });
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
  for (const opponent of ['random', 'passive', 'checkpoint:abc'] as const) {
    const { body } = render(DecklistsSelector, { props: { value: 'open', opponent } });
    expect(optionTag(body, 'closed')).not.toContain('disabled');
  }
});
