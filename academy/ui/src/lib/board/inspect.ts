/**
 * What inspecting a permanent explains beyond its printed face: where its
 * power and toughness come from, and which keywords it has and why.
 *
 * Both read only the frame's projection (effective P/T, effective keywords,
 * counters, the continuous-effect records) and the printed face. The
 * projection's keyword list is the authority on which keywords a permanent
 * has; the effect records and the printed face only say why. Nothing here is
 * inferred beyond that: what the named parts do not explain is reported as
 * "other effects", and a keyword neither printed nor recorded as "another
 * ability" (a static grant from another permanent).
 */
import type { CardFace } from '$lib/card-faces.svelte';
import type { ContinuousEffect, PermanentState } from '$lib/types';

export interface PtPart {
  label: string;
  power: number;
  toughness: number;
  /** True for the starting value (printed or set); false for a change to it. */
  base: boolean;
}

export interface PtProvenance {
  power: number;
  toughness: number;
  parts: PtPart[];
}

const DURATIONS: Record<string, string> = {
  EndOfTurn: 'until end of turn',
  UntilYourNextTurn: 'until your next turn',
  WhileYouControlSource: 'while its source is controlled',
};

/** Effect records that are a status rather than a change to a characteristic. */
const STATUSES: Record<string, string> = { Saddled: 'Saddled' };

/** The statuses a permanent's effect records give it: "Saddled until end of turn". */
export function statusesOf(permanent: PermanentState): string[] {
  return [...(permanent.effects ?? [])].sort((a, b) => a.timestamp - b.timestamp)
    .filter(effect => effect.modification in STATUSES)
    .map(effect => [STATUSES[effect.modification], DURATIONS[effect.duration]].filter(Boolean).join(' '));
}

function effectLabel(effect: ContinuousEffect): string {
  const duration = DURATIONS[effect.duration];
  return [effect.source ?? 'An effect', duration].filter(Boolean).join(', ');
}

/** The permanent's P/T as a sum of named parts, or null when it has none. */
export function ptProvenance(permanent: PermanentState, face: CardFace | null): PtProvenance | null {
  // The projection reports 0/0 for a noncreature; only a creature has power and toughness.
  const creature = permanent.types?.is_creature ?? true;
  if (!creature || permanent.power === null || permanent.toughness === null) return null;
  const effects = [...(permanent.effects ?? [])].sort((a, b) => a.timestamp - b.timestamp);
  const parts: PtPart[] = [];
  const set = effects.filter(effect => effect.modification === 'SetBasePt').at(-1);
  if (set) parts.push({ label: `Set by ${effectLabel(set)}`, power: set.power ?? 0, toughness: set.toughness ?? 0, base: true });
  else if (face?.power != null && face.toughness != null) {
    parts.push({ label: permanent.copy_of ? `Printed (copy of ${permanent.copy_of.name})` : 'Printed', power: face.power, toughness: face.toughness, base: true });
  }
  const counters = permanent.plus1_counters;
  if (counters) parts.push({ label: `${counters} +1/+1 counter${counters === 1 ? '' : 's'}`, power: counters, toughness: counters, base: false });
  for (const effect of effects) {
    if (effect.modification !== 'ModifyPt') continue;
    parts.push({ label: effectLabel(effect), power: effect.power ?? 0, toughness: effect.toughness ?? 0, base: false });
  }
  if (parts.some(part => part.base)) {
    const power = permanent.power - parts.reduce((sum, part) => sum + part.power, 0);
    const toughness = permanent.toughness - parts.reduce((sum, part) => sum + part.toughness, 0);
    if (power || toughness) parts.push({ label: 'Other effects', power, toughness, base: false });
  }
  return { power: permanent.power, toughness: permanent.toughness, parts };
}

export interface KeywordFact {
  keyword: string;
  /** Why it has it: null when printed, else the granting effect or "Another ability". */
  source: string | null;
}

export const OTHER_GRANT = 'Another ability';

/**
 * The keywords a permanent has, from the projection's effective list, each
 * with its reason: printed, the effect record that granted it, or another
 * ability. A permanent the projection lists no keywords for has none.
 */
export function keywordsOf(permanent: PermanentState, face: CardFace | null): KeywordFact[] {
  const printed = new Set(permanent.abilities_lost ? [] : face?.keywords ?? []);
  const granted = new Map<string, string>();
  for (const effect of [...(permanent.effects ?? [])].sort((a, b) => a.timestamp - b.timestamp)) {
    if (effect.modification !== 'AddKeywords') continue;
    for (const keyword of effect.keywords ?? []) if (!granted.has(keyword)) granted.set(keyword, effectLabel(effect));
  }
  return (permanent.keywords ?? []).map(keyword => ({
    keyword,
    source: printed.has(keyword) ? null : granted.get(keyword) ?? OTHER_GRANT,
  }));
}

/** "+2/+0", "−1/−1": a P/T change as printed in rules text. */
export function signed(power: number, toughness: number): string {
  const one = (value: number): string => (value < 0 ? `−${-value}` : `+${value}`);
  return `${one(power)}/${one(toughness)}`;
}

/** A keyword or counter kind as a label: `first_strike` reads "First strike". */
export function labelText(name: string): string {
  const words = name.replace(/_/g, ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
}
