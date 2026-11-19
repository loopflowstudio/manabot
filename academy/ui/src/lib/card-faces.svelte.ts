/**
 * Printed card faces by name: mana cost, colours, type line, printed keywords and
 * printed power/toughness, from `/api/card-faces`.
 *
 * `cardFace(name)` is reactive: it returns the face once loaded and null until
 * then (or when the engine defines no card of that name). Names asked for in
 * the same tick are fetched in one request, and only names a viewer already
 * sees are ever asked for, as with card art. A face is printed information;
 * the frame's projection stays the authority for what a permanent is now.
 */
import { SvelteMap } from 'svelte/reactivity';

export interface CardFace {
  /** Mana symbols as printed, `{2}{U/B}{R}`; empty for lands and tokens. */
  cost: string;
  /** Printed colours as letters in printed order, `['G', 'W']`; empty when colourless. */
  colors: string[];
  type_line: string;
  keywords: string[];
  power: number | null;
  toughness: number | null;
}

const faces = new SvelteMap<string, CardFace | null>();
const requested = new Set<string>();
let batch: string[] = [];

async function load(names: string[]): Promise<void> {
  const params = new URLSearchParams();
  names.forEach(name => params.append('names', name));
  let found: Record<string, CardFace> = {};
  try {
    const response = await fetch(`/api/card-faces?${params}`);
    if (response.ok) found = await response.json() as Record<string, CardFace>;
  } catch {
    // A face is decoration beside the projection; without one the card shows its name.
  }
  for (const name of names) faces.set(name, found[name] ?? null);
}

export function cardFace(name: string | null | undefined): CardFace | null {
  if (!name) return null;
  if (!requested.has(name) && typeof fetch !== 'undefined') {
    requested.add(name);
    batch.push(name);
    if (batch.length === 1) queueMicrotask(() => { const names = batch; batch = []; void load(names); });
  }
  return faces.get(name) ?? null;
}

/** Mana symbols of a printed cost, `{2}{U/B}` → `['2', 'U/B']`. */
export function costSymbols(cost: string): string[] {
  return [...cost.matchAll(/\{([^}]+)\}/g)].map(match => match[1]);
}

const BASIC_COLORS: Record<string, string> = { Plains: 'W', Island: 'U', Swamp: 'B', Mountain: 'R', Forest: 'G' };

/**
 * The colours a card's frame is drawn in, in printed order: its printed
 * colours, or for a land without colour the colours of its basic land types
 * (a Forest's frame is green, as printed). Empty draws the colourless frame.
 */
export function frameColors(face: CardFace | null): string[] {
  if (!face) return [];
  if (face.colors.length) return face.colors;
  if (!/\bLand\b/.test(face.type_line)) return [];
  const subtypes = face.type_line.split('—')[1]?.trim().split(/\s+/) ?? [];
  return [...new Set(subtypes.map(subtype => BASIC_COLORS[subtype]).filter((color): color is string => color !== undefined))];
}
