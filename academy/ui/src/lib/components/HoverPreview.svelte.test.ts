import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import HoverPreview from './HoverPreview.svelte';

it('shows the registered rules text with the card preview', () => {
  const text = 'You have no maximum hand size.\nAt the beginning of combat on your turn…';
  const { body } = render(HoverPreview, {
    props: {
      card: { name: "Proft's Eidetic Memory", text_box: text, power: null, toughness: null },
    },
  });
  expect(body).toContain('data-testid="card-rules-text"');
  expect(body).toContain(text);
});
