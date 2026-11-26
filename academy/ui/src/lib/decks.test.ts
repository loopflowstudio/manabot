import { expect, it } from 'vitest';
import { deckChoiceLabel, matchupDeckLabel, recordedDeckLabel } from './decks';

it('names a recorded deck, and says a closed one is closed', () => {
  expect(recordedDeckLabel('closed')).toBe('Closed decklist');
  expect(recordedDeckLabel('closed', 'GW Allies')).toBe('Closed decklist');
  expect(recordedDeckLabel('gw_allies')).toBe('gw allies');
  expect(recordedDeckLabel('gw_allies', 'GW Allies')).toBe('GW Allies');
});

it('names a last-matchup side for what Play again starts', () => {
  // Dealt from a pool: the pool, never the deck it dealt or a closed name.
  expect(matchupDeckLabel('pool:held-out', 'TLA Trophy · UR · 0012', 'pool:held-out')).toBe('Deck pool · held-out');
  expect(matchupDeckLabel('pool:held-out', 'Closed decklist', 'gw_allies')).toBe('Deck pool · held-out');
  // A named deck: what was dealt, or the deck setup holds while it is closed.
  expect(matchupDeckLabel(undefined, 'UR Lessons', 'gw_allies')).toBe('UR Lessons');
  expect(matchupDeckLabel(null, 'Closed decklist', 'gw_allies')).toBe(deckChoiceLabel('gw_allies'));
  expect(matchupDeckLabel(null, 'Closed decklist', 'gw_allies')).not.toBe('Closed decklist');
});
