import { CURATED_PACK } from './curated-pack';

export type DeckChoice = 'ur_lessons' | 'gw_allies' | 'interactive';

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
];

const DECK_KEYS = new Set<string>(DECK_CHOICES.map((deck) => deck.key));

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
    !DECK_KEYS.has(candidate.hero) ||
    !DECK_KEYS.has(candidate.villain)
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
