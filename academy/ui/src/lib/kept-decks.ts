/**
 * Browser-local kept decklists (`academy.imported-decks`), re-validated by the
 * server on every use. Keeps original source text so u64/digest rules stay intact.
 */

export interface KeptDeck {
  id: string;
  name: string;
  filename: string;
  source: string;
  saved_at: string;
}

export const KEPT_DECKS_KEY = 'academy.imported-decks';
export const KEPT_DECK_PREFIX = 'kept:';

export function isKeptDeckChoice(value: unknown): value is `kept:${string}` {
  return typeof value === 'string' && value.startsWith(KEPT_DECK_PREFIX) && value.length > KEPT_DECK_PREFIX.length;
}

export function keptDeckId(value: string): string {
  return value.startsWith(KEPT_DECK_PREFIX) ? value.slice(KEPT_DECK_PREFIX.length) : value;
}

export function keptDeckChoice(id: string): `kept:${string}` {
  return `${KEPT_DECK_PREFIX}${id}`;
}

export function loadKeptDecks(): KeptDeck[] {
  if (typeof localStorage === 'undefined') return [];
  try {
    const raw = localStorage.getItem(KEPT_DECKS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is KeptDeck =>
        typeof item === 'object' &&
        item !== null &&
        typeof item.id === 'string' &&
        typeof item.name === 'string' &&
        typeof item.filename === 'string' &&
        typeof item.source === 'string',
    );
  } catch {
    return [];
  }
}

export function saveKeptDeck(input: { name: string; filename: string; source: string }): KeptDeck[] {
  const cleanedName = input.name.trim() || input.filename.replace(/\.(txt|dek)$/i, '').trim() || 'Kept deck';
  const id = cleanedName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 40) || `deck-${Date.now()}`;
  const entry: KeptDeck = {
    id,
    name: cleanedName.slice(0, 120),
    filename: input.filename || `${id}.txt`,
    source: input.source,
    saved_at: new Date().toISOString(),
  };
  const existing = loadKeptDecks().filter(deck => deck.id !== id);
  const next = [entry, ...existing].slice(0, 40);
  if (typeof localStorage !== 'undefined') {
    try {
      localStorage.setItem(KEPT_DECKS_KEY, JSON.stringify(next));
    } catch {
      /* Storage full or disabled; caller keeps in-memory list. */
    }
  }
  return next;
}

export function removeKeptDeck(id: string): KeptDeck[] {
  const next = loadKeptDecks().filter(deck => deck.id !== id);
  if (typeof localStorage !== 'undefined') {
    try {
      localStorage.setItem(KEPT_DECKS_KEY, JSON.stringify(next));
    } catch {
      /* Ignore storage write failures. */
    }
  }
  return next;
}

export function findKeptDeck(choice: string, list: readonly KeptDeck[] = loadKeptDecks()): KeptDeck | null {
  if (!isKeptDeckChoice(choice)) return null;
  const id = keptDeckId(choice);
  return list.find(deck => deck.id === id) ?? null;
}
