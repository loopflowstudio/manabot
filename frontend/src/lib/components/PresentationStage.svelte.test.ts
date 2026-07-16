import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';

import { LIGHTNING_BOLT_PRESENTATION } from '$lib/presentation';
import { createPresentationPlayer } from '$lib/presentation.svelte';

import PresentationStage from './PresentationStage.svelte';

describe('PresentationStage', () => {
  it('renders the current semantic beat rather than deriving text from a board snapshot', () => {
    const player = createPresentationPlayer();
    player.load(LIGHTNING_BOLT_PRESENTATION);

    const { body } = render(PresentationStage, { props: { player } });

    expect(body).toContain('data-presentation-seq="900"');
    expect(body).toContain('data-presentation-kind="cast"');
    expect(body).toContain('Spell cast');
    expect(body).toContain('Hero casts Lightning Bolt.');
    expect(body).toContain('Skip beat');
    expect(body).toContain('Fast-forward');
    // Native buttons preserve pointer activation plus Enter/Space keyboard
    // activation without a parallel interaction model.
    expect(body.match(/<button/g)).toHaveLength(3);
    expect(body.match(/type="button"/g)).toHaveLength(3);
  });
});
