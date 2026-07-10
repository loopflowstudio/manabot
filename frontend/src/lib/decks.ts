// Named decks selectable in the play UI. Keys mirror gui/server.py
// NAMED_DECKS; the server resolves them to full {card: count} lists and
// echoes display names on every payload (deck_names).
export const DECK_CHOICES = [
  { key: 'ur_lessons', label: 'UR Lessons' },
  { key: 'gw_allies', label: 'GW Allies' },
  { key: 'interactive', label: 'Interactive' },
] as const;

export type DeckChoice = (typeof DECK_CHOICES)[number]['key'];

const DECK_KEYS = new Set<string>(DECK_CHOICES.map((deck) => deck.key));

export const DECKS_STORAGE_KEY = 'manabot.gui.decks';

export interface DeckSelection {
  hero: DeckChoice;
  villain: DeckChoice;
}

// Mirrors the server defaults: the Milestone-1 matchup, UR hero vs GW.
export function defaultDeckSelection(): DeckSelection {
  return { hero: 'ur_lessons', villain: 'gw_allies' };
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
