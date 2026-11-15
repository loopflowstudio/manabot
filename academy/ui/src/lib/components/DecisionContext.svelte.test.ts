import { readFileSync } from 'node:fs';
import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import type { ExperienceFrame } from '$lib/types';
import DecisionContext from './DecisionContext.svelte';
import combatFixture from '../../../../../protocol/fixtures/combat-declaration.json';

it('renders native pending resolution without adding a choice', () => {
  const fixture = JSON.parse(readFileSync(new URL('../../../../../managym/academy/fixtures/choice-navigation-w17-token-cards.json', import.meta.url), 'utf8'));
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

it('explains whole-plan confirmation only when a declaration composer is available', () => {
  const frame = combatFixture.recovery.frame as ExperienceFrame;
  const { body } = render(DecisionContext, { props: { observation: frame.projection, declaration: frame.declaration } });
  expect(body).toContain('then confirm once');
  expect(body).toContain('already committed assignments stay fixed');
  expect(body).not.toContain(frame.projection.decision_context!.instruction);
  const child = render(DecisionContext, { props: { observation: frame.projection } }).body;
  expect(child).toContain(frame.projection.decision_context!.instruction);
});
