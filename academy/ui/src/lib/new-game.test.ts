import { afterEach, expect, it, vi } from 'vitest';
import { GameStore } from './game.svelte';
import { startConfiguredWatch } from './new-game';
import { watchStart } from './watch-start.svelte';

afterEach(() => vi.restoreAllMocks());

it('posts Seat 1 first after a game played on the draw', async () => {
  const start = vi.spyOn(watchStart, 'start').mockResolvedValue({ id: 'watched', finished: false });
  const setup = new GameStore();
  // What playing a game on the draw leaves behind in setup.
  setup.humanSeat = 1;
  setup.turnOrder = 'draw';
  setup.watchMode = true;
  setup.watchPlayerChoice = 'random';
  setup.setOpponentChoice('passive');
  setup.setHeroDeck('ur_lessons');
  setup.setVillainDeck('gw_allies');

  expect(await startConfiguredWatch(setup)).toBe('watched');
  expect(start).toHaveBeenCalledWith(expect.objectContaining({
    players: ['random', 'passive'],
    hero_deck: 'ur_lessons',
    villain_deck: 'gw_allies',
  }));
});
