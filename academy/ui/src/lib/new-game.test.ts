import { afterEach, expect, it, vi } from 'vitest';
import { GameStore } from './game.svelte';
import { KEPT_DECKS_KEY, keptDeckChoice } from './kept-decks';
import { startConfiguredWatch } from './new-game';
import { watchStart } from './watch-start.svelte';

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

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

it('sends the saved list behind a kept deck, so Watch starts it', async () => {
  const start = vi.spyOn(watchStart, 'start').mockResolvedValue({ id: 'watched', finished: false });
  const kept = JSON.stringify([{ id: 'mine', name: 'My brew', filename: 'brew.txt', source: '40 Island\n' }]);
  vi.stubGlobal('localStorage', { getItem: (key: string) => (key === KEPT_DECKS_KEY ? kept : null), setItem: () => {} });
  const setup = new GameStore();
  setup.watchMode = true;
  setup.setHeroDeck(keptDeckChoice('mine'));
  setup.setVillainDeck('gw_allies');

  await startConfiguredWatch(setup);
  expect(start).toHaveBeenCalledWith(expect.objectContaining({
    hero_deck: 'kept:mine',
    villain_deck: 'gw_allies',
    deck_import: { hero: { source: '40 Island\n', filename: 'brew.txt', name: 'My brew' } },
  }));
});
