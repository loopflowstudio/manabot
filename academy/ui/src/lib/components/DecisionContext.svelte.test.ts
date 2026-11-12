import { readFileSync } from 'node:fs';
import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import type { ExperienceFrame } from '$lib/types';
import DecisionContext from './DecisionContext.svelte';

it('renders native pending resolution without adding a choice', () => {
  const fixture = JSON.parse(readFileSync(new URL('../../../../../academy/fixtures/choice-navigation-w17.json', import.meta.url), 'utf8'));
  const frame = fixture.positions.LEARN.frame as ExperienceFrame;
  const observation = structuredClone(frame.projection);
  if (!observation.decision_context) throw new Error('expected decision context');
  observation.decision_context.pending = ['Pending: create 2 Food tokens.', 'After this choice: gain 3 life.'];
  const { body } = render(DecisionContext, { props: { observation } });
  expect(body).toContain('Pending resolution');
  expect(body).toContain('create 2 Food tokens');
  expect(body).toContain('gain 3 life');
  expect(body).not.toContain('<button');
});
