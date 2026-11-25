/**
 * Pure viewer-safe decklist tracker computations modeled on Untapped.gg's overlay.
 *
 * Consumes only authoritative viewer-safe fields already present on PlayerState
 * (`main_counts`, `unseen_counts`, `sideboard_counts`, `remaining_sideboard_counts`,
 * `hand_hidden_count`) and printed card metadata from `GET /api/card-faces`.
 */

import type { PlayerState } from '$lib/types';

export interface TrackerCardFace {
  cost: string;
  colors: string[];
  type_line: string;
  keywords: string[];
  power: number | null;
  toughness: number | null;
}

export interface TrackerRow {
  name: string;
  remaining: number;
  starting: number;
  seen: number;
  cost: string;
  manaValue: number;
  isLand: boolean;
  accentColor: string;
  /** Chance the next card drawn from this deck's unseen cards is this one; the same reading for both decks. */
  drawPercent: number;
  /** Chance at least one copy is in the hidden hand; only an opponent's open list has one. */
  handPercent: number | null;
}

export interface DeckTrackerView {
  mode: 'self' | 'opponent-open' | 'opponent-closed';
  rows: TrackerRow[];
  sideboard: TrackerRow[];
  totalRemaining: number;
  landsRemaining: number;
  landPercent: number;
  spellsRemaining: number;
  spellPercent: number;
  totalSeen: number;
}

const COLOR_VAR: Record<string, string> = {
  W: 'var(--vivid-w)',
  U: 'var(--vivid-u)',
  B: 'var(--vivid-b)',
  R: 'var(--vivid-r)',
  G: 'var(--vivid-g)',
};

const BASIC_LAND_COLORS: Record<string, string> = {
  Plains: 'var(--vivid-w)',
  Island: 'var(--vivid-u)',
  Swamp: 'var(--vivid-b)',
  Mountain: 'var(--vivid-r)',
  Forest: 'var(--vivid-g)',
};

export function parseManaValue(cost: string): number {
  if (!cost) return 0;
  let total = 0;
  for (const match of cost.matchAll(/\{([^}]+)\}/g)) {
    const token = match[1];
    if (/^\d+$/.test(token)) total += Number(token);
    else if (token !== 'X' && token !== 'Y' && token !== 'Z') total += 1;
  }
  return total;
}

export function compactCost(cost: string): string {
  return cost.replaceAll('{', '').replaceAll('}', '');
}

function accentFor(name: string, face: TrackerCardFace | undefined): string {
  if (face && face.colors.length > 0) {
    return COLOR_VAR[face.colors[0]] ?? 'var(--border-strong)';
  }
  return BASIC_LAND_COLORS[name] ?? 'var(--border-strong)';
}

function isLandCard(name: string, face: TrackerCardFace | undefined): boolean {
  if (face) return /\bLand\b/i.test(face.type_line);
  return name in BASIC_LAND_COLORS;
}

/** Exact hypergeometric probability P(at least 1 of `copies` in a sample of size `drawCount` from `population`). */
export function probAtLeastOne(copies: number, population: number, drawCount: number): number {
  if (copies <= 0 || population <= 0 || drawCount <= 0) return 0;
  if (copies > population - drawCount) return 100;
  let miss = 1;
  for (let i = 0; i < drawCount; i++) {
    miss *= (population - copies - i) / (population - i);
  }
  return Math.round((1 - miss) * 100);
}

function sortRows(rows: TrackerRow[]): TrackerRow[] {
  return [...rows].sort((a, b) => {
    if (a.isLand !== b.isLand) return a.isLand ? 1 : -1;
    if (a.manaValue !== b.manaValue) return a.manaValue - b.manaValue;
    return a.name.localeCompare(b.name);
  });
}

function resolveCounts(
  raw: Record<string, number> | undefined,
  definitionNames: Readonly<Record<string, string>>,
): Record<string, number> {
  if (!raw) return {};
  const resolved: Record<string, number> = {};
  for (const [key, count] of Object.entries(raw)) {
    const name = definitionNames[key] ?? key;
    resolved[name] = (resolved[name] ?? 0) + count;
  }
  return resolved;
}

function buildSideboardRows(
  sideboard: Record<string, number> | undefined,
  remaining: Record<string, number> | undefined,
  faces: Readonly<Record<string, TrackerCardFace>>,
  definitionNames: Readonly<Record<string, string>>,
): TrackerRow[] {
  if (!sideboard) return [];
  const startMap = resolveCounts(sideboard, definitionNames);
  const leftMap = remaining ? resolveCounts(remaining, definitionNames) : null;
  const names = new Set([...Object.keys(startMap), ...Object.keys(leftMap ?? {})]);
  return sortRows(
    [...names].map(name => {
      const start = startMap[name] ?? 0;
      const left = leftMap ? (leftMap[name] ?? 0) : start;
      const face = faces[name];
      const cost = face?.cost ?? '';
      return {
        name,
        remaining: left,
        starting: start,
        seen: Math.max(0, start - left),
        cost: compactCost(cost),
        manaValue: parseManaValue(cost),
        isLand: isLandCard(name, face),
        accentColor: accentFor(name, face),
        drawPercent: 0,
        handPercent: null,
      };
    }),
  );
}

export function buildSelfTracker(
  player: PlayerState,
  faces: Readonly<Record<string, TrackerCardFace>>,
  definitionNames: Readonly<Record<string, string>> = {},
): DeckTrackerView | null {
  if (!player.main_counts) return null;
  const main = resolveCounts(player.main_counts, definitionNames);
  const unseen = resolveCounts(player.unseen_counts, definitionNames);
  const names = new Set([...Object.keys(main), ...Object.keys(unseen)]);
  const totalRemaining = Object.values(unseen).reduce((sum, count) => sum + count, 0);

  let landsRemaining = 0;
  const rows: TrackerRow[] = [];
  for (const name of names) {
    const starting = main[name] ?? 0;
    const remaining = unseen[name] ?? 0;
    const face = faces[name];
    const land = isLandCard(name, face);
    if (land) landsRemaining += remaining;
    const cost = face?.cost ?? '';
    rows.push({
      name,
      remaining,
      starting,
      seen: Math.max(0, starting - remaining),
      cost: land ? 'Land' : compactCost(cost),
      manaValue: land ? 99 : parseManaValue(cost),
      isLand: land,
      accentColor: accentFor(name, face),
      drawPercent: totalRemaining > 0 ? Math.round((remaining / totalRemaining) * 100) : 0,
      handPercent: null,
    });
  }

  const spellsRemaining = Math.max(0, totalRemaining - landsRemaining);
  const landPercent = totalRemaining > 0 ? Math.round((landsRemaining / totalRemaining) * 100) : 0;
  return {
    mode: 'self',
    rows: sortRows(rows),
    sideboard: buildSideboardRows(player.sideboard_counts, player.remaining_sideboard_counts, faces, definitionNames),
    totalRemaining,
    landsRemaining,
    landPercent,
    spellsRemaining,
    spellPercent: totalRemaining > 0 ? 100 - landPercent : 0,
    totalSeen: Object.values(main).reduce((s, c) => s + c, 0) - totalRemaining,
  };
}

export function buildOpponentTracker(
  opponent: PlayerState,
  faces: Readonly<Record<string, TrackerCardFace>>,
  definitionNames: Readonly<Record<string, string>> = {},
): DeckTrackerView {
  const handHidden = opponent.hand_hidden_count ?? opponent.zone_counts.HAND ?? 0;
  if (opponent.main_counts) {
    const main = resolveCounts(opponent.main_counts, definitionNames);
    const unseen = resolveCounts(opponent.unseen_counts, definitionNames);
    const names = new Set([...Object.keys(main), ...Object.keys(unseen)]);
    const totalRemaining = Object.values(unseen).reduce((sum, count) => sum + count, 0);

    let landsRemaining = 0;
    const rows: TrackerRow[] = [];
    for (const name of names) {
      const starting = main[name] ?? 0;
      const remaining = unseen[name] ?? 0;
      const face = faces[name];
      const land = isLandCard(name, face);
      if (land) landsRemaining += remaining;
      const cost = face?.cost ?? '';
      rows.push({
        name,
        remaining,
        starting,
        seen: Math.max(0, starting - remaining),
        cost: land ? 'Land' : compactCost(cost),
        manaValue: land ? 99 : parseManaValue(cost),
        isLand: land,
        accentColor: accentFor(name, face),
        drawPercent: totalRemaining > 0 ? Math.round((remaining / totalRemaining) * 100) : 0,
        handPercent: probAtLeastOne(remaining, totalRemaining, handHidden),
      });
    }
    const spellsRemaining = Math.max(0, totalRemaining - landsRemaining);
    const landPercent = totalRemaining > 0 ? Math.round((landsRemaining / totalRemaining) * 100) : 0;
    return {
      mode: 'opponent-open',
      rows: sortRows(rows),
      sideboard: buildSideboardRows(opponent.sideboard_counts, opponent.remaining_sideboard_counts, faces, definitionNames),
      totalRemaining,
      landsRemaining,
      landPercent,
      spellsRemaining,
      spellPercent: totalRemaining > 0 ? 100 - landPercent : 0,
      totalSeen: Object.values(main).reduce((s, c) => s + c, 0) - totalRemaining,
    };
  }

  // Closed decklists: tally only cards legally visible to this viewer right now.
  const seenCounts = new Map<string, number>();
  const add = (name: string | null | undefined): void => {
    if (!name) return;
    seenCounts.set(name, (seenCounts.get(name) ?? 0) + 1);
  };
  for (const perm of opponent.battlefield) {
    if (!perm.is_token) add(perm.name);
  }
  for (const card of opponent.graveyard) add(card.name);
  for (const card of opponent.exile) add(card.name);
  for (const card of opponent.hand) add(card.name);
  for (const item of opponent.stack) add(item.name);

  const rows: TrackerRow[] = [];
  let totalSeen = 0;
  for (const [name, seen] of seenCounts.entries()) {
    totalSeen += seen;
    const face = faces[name];
    const land = isLandCard(name, face);
    const cost = face?.cost ?? '';
    rows.push({
      name,
      remaining: seen,
      starting: seen,
      seen,
      cost: land ? 'Land' : compactCost(cost),
      manaValue: land ? 99 : parseManaValue(cost),
      isLand: land,
      accentColor: accentFor(name, face),
      drawPercent: 0,
      handPercent: null,
    });
  }
  return {
    mode: 'opponent-closed',
    rows: sortRows(rows),
    sideboard: [],
    totalRemaining: 0,
    landsRemaining: 0,
    landPercent: 0,
    spellsRemaining: 0,
    spellPercent: 0,
    totalSeen,
  };
}

const faceCache: Record<string, TrackerCardFace> = {};

export async function loadTrackerCardFaces(names: readonly string[]): Promise<Record<string, TrackerCardFace>> {
  const missing = [...new Set(names)].filter(name => name && !(name in faceCache));
  if (missing.length > 0 && typeof fetch === 'function') {
    try {
      const params = new URLSearchParams();
      for (const name of missing.slice(0, 256)) params.append('names', name);
      const response = await fetch(`/api/card-faces?${params.toString()}`);
      if (response.ok) {
        const fetched = (await response.json()) as Record<string, TrackerCardFace>;
        Object.assign(faceCache, fetched);
      }
    } catch {
      /* Offline or test mock: render gracefully with name-only rows. */
    }
  }
  return { ...faceCache };
}
