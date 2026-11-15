import { readFileSync } from 'node:fs';
import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import type { ExperienceFrame, Observation } from '$lib/types';
import UnavailableActions from './UnavailableActions.svelte';

function openingHand(): Observation {
  const fixture = JSON.parse(readFileSync(new URL('../../../../../managym/academy/fixtures/choice-navigation-w17-token-cards.json', import.meta.url), 'utf8'));
  return structuredClone((fixture.positions.LEARN.frame as ExperienceFrame).projection);
}

it('reads copies refused for one native reason as one line and keeps distinct reasons', () => {
  const observation = openingHand();
  const hand = observation.agent.hand;
  const name = hand[0].name;
  for (const [index, card] of hand.entries()) card.name = index < 3 ? name : `Other ${index}`;
  const cost = 'The complete casting cost cannot be paid, including additional costs.';
  observation.unavailable_actions = [
    { object: hand[0].id, message: cost },
    { object: hand[1].id, message: cost },
    { object: hand[2].id, message: 'Only at sorcery speed.' },
    { object: hand[3].id, message: cost },
  ] as Observation['unavailable_actions'];
  const { body } = render(UnavailableActions, { props: { observation } });
  expect(body).toContain(`${name} · your hand · 2 copies: ${cost}`);
  expect(body).toContain(`${name} · your hand · copy 3 of 3: Only at sorcery speed.`);
  expect(body).toContain(`Other 3 · your hand: ${cost}`);
  expect(body.match(/<li>/g)).toHaveLength(3);
});
