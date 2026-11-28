import { CURATED_PACK, ELEMENTAL_WORLD_PACK, JEONG_INCREMENT_PACK } from './curated-pack';
import { findKeptDeck, isKeptDeckChoice } from './kept-decks';

export type ElementalDeckChoice = `elemental_${'gr' | 'gw' | 'ur' | 'ub' | 'uw' | 'ug' | 'wr' | 'wb' | 'rb' | 'bg'}`;
export type PoolDeckChoice = `pool:${string}`;
export type KeptDeckChoice = `kept:${string}`;
export type DeckChoice = 'ur_lessons' | 'gw_allies' | 'gw_allies_jeong' | 'interactive' | ElementalDeckChoice | PoolDeckChoice | KeptDeckChoice;

export interface DeckPoolSummary {
  id: string;
  key: PoolDeckChoice;
  label: string;
  count: number;
  default_decklists: 'open' | 'closed';
}

export function isPoolDeckChoice(value: unknown): value is PoolDeckChoice {
  return typeof value === 'string' && value.startsWith('pool:') && value.length > 5;
}

export const ELEMENTAL_DECK_CHOICES: readonly { key: ElementalDeckChoice; label: string }[] =
  (ELEMENTAL_WORLD_PACK.roster ?? []).map((row) => ({ key: row.deck_id as ElementalDeckChoice, label: row.display_name }));

// The two curated choices and their defaults are derived from the installed
// manifest. Interactive remains a backwards-compatible, uncurated option.
export const DECK_CHOICES: readonly { key: DeckChoice; label: string }[] = [
  {
    key: CURATED_PACK.matchup.hero.deck_id as DeckChoice,
    label: CURATED_PACK.matchup.hero.display_name,
  },
  {
    key: CURATED_PACK.matchup.villain.deck_id as DeckChoice,
    label: CURATED_PACK.matchup.villain.display_name,
  },
  { key: 'interactive', label: 'Interactive' },
  { key: JEONG_INCREMENT_PACK.matchup.hero.deck_id as DeckChoice, label: JEONG_INCREMENT_PACK.matchup.hero.display_name },
  ...ELEMENTAL_DECK_CHOICES,
];

const DECK_KEYS = new Set<string>(DECK_CHOICES.map((deck) => deck.key));

/** What the server names a closed opponent's deck until the game is over. */
export const CLOSED_DECK_NAME = 'Closed decklist';

/** A recorded deck as History and Replay name it: the stored id reads `closed` while an unfinished game keeps its opponent's deck unnamed. */
export function recordedDeckLabel(deck: string, name?: string | null): string {
  return deck === 'closed' ? CLOSED_DECK_NAME : name ?? deck.replaceAll('_', ' ');
}

/** A setup deck choice in words: a named deck, a pool (never the deck it last dealt) or a saved list. */
export function deckChoiceLabel(key: unknown): string {
  const found = DECK_CHOICES.find(deck => deck.key === key)?.label;
  if (found) return found;
  if (isPoolDeckChoice(key)) return `Deck pool · ${key.slice('pool:'.length)}`;
  if (isKeptDeckChoice(key)) return findKeptDeck(key)?.name ?? 'Saved deck';
  if (typeof key === 'string' && key && key !== 'imported' && key !== 'custom') return key;
  return 'Imported deck';
}

/**
 * One side of "Your last matchup", named for what Play again starts: its pool
 * when the side was dealt from one, the dealt deck otherwise, and the deck
 * setup still holds while a closed opponent's is unnamed.
 */
export function matchupDeckLabel(pool: unknown, dealt: string | undefined, chosen: unknown): string {
  if (isPoolDeckChoice(pool)) return deckChoiceLabel(pool);
  return dealt && dealt !== CLOSED_DECK_NAME ? dealt : deckChoiceLabel(chosen);
}

export const DECKS_STORAGE_KEY = 'etude.gui.decks';

export interface DeckSelection {
  hero: DeckChoice;
  villain: DeckChoice;
}

// Mirrors the server defaults: the Milestone-1 matchup, UR hero vs GW.
export function defaultDeckSelection(): DeckSelection {
  return {
    hero: CURATED_PACK.matchup.hero.deck_id as DeckChoice,
    villain: CURATED_PACK.matchup.villain.deck_id as DeckChoice,
  };
}

export function sanitizeDeckSelection(raw: unknown): DeckSelection | null {
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const candidate = raw as Record<string, unknown>;
  if (
    typeof candidate.hero !== 'string' ||
    typeof candidate.villain !== 'string' ||
    (!DECK_KEYS.has(candidate.hero) && !isPoolDeckChoice(candidate.hero) && !isKeptDeckChoice(candidate.hero)) ||
    (!DECK_KEYS.has(candidate.villain) && !isPoolDeckChoice(candidate.villain) && !isKeptDeckChoice(candidate.villain))
  ) {
    return null;
  }
  return {
    hero: candidate.hero as DeckChoice,
    villain: candidate.villain as DeckChoice,
  };
}

export function loadStoredDeckSelection(): DeckSelection {
  if (typeof localStorage === 'undefined') {
    return defaultDeckSelection();
  }
  try {
    const raw = localStorage.getItem(DECKS_STORAGE_KEY);
    if (!raw) {
      return defaultDeckSelection();
    }
    return sanitizeDeckSelection(JSON.parse(raw)) ?? defaultDeckSelection();
  } catch {
    return defaultDeckSelection();
  }
}

export function saveStoredDeckSelection(selection: DeckSelection): void {
  if (typeof localStorage === 'undefined') {
    return;
  }
  try {
    localStorage.setItem(DECKS_STORAGE_KEY, JSON.stringify(selection));
  } catch {
    // Storage full or unavailable: the selection simply won't persist.
  }
}
