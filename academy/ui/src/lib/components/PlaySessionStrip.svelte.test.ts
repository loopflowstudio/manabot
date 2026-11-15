import { render } from 'svelte/server';
import { afterEach, describe, expect, it } from 'vitest';
import PlaySessionStrip from './PlaySessionStrip.svelte';
import { gameStore } from '$lib/game.svelte';

const noop = (): void => {};

describe('play session strip', () => {
  afterEach(() => {
    gameStore.playSession = null;
    gameStore.resumeFailed = false;
  });

  it('offers no continuation for a session the server no longer has', () => {
    gameStore.playSession = { id: 'ps-1', session_id: 's-1', game_number: 1, revision: 1, draft: {seed: null}, config: {human_seat: 0} };
    expect(render(PlaySessionStrip, {props: {onChangeOpponent: noop, onEditNext: noop}}).body).toContain('Rematch');
    gameStore.resumeFailed = true;
    const { body } = render(PlaySessionStrip, {props: {onChangeOpponent: noop, onEditNext: noop}});
    for (const control of ['Rematch', 'Swap sides', 'Edit next game', 'Change opponent']) expect(body).not.toContain(control);
    expect(body).toContain('Session history');
  });
});
