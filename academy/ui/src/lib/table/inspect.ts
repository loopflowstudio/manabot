/**
 * What inspecting a permanent explains beyond its printed face: where its
 * power and toughness come from, and which keywords it has and why.
 *
 * Both read only the frame's projection (effective P/T, counters, the
 * continuous-effect records) and the printed face. The projection carries no
 * layered keyword list, so `keywordsOf` reports printed keywords and the grants
 * and losses its effect records name; a grant from another permanent's static
 * ability is not among them. Nothing here is inferred beyond that arithmetic:
 * what the named parts do not explain is reported as "other effects".
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
  /** The effect that granted it; null when printed. */
  source: string | null;
}

/** Printed keywords and those its effect records grant, less what they remove. */
export function keywordsOf(permanent: PermanentState | null, face: CardFace | null): KeywordFact[] {
  const out = new Map<string, KeywordFact>();
  if (!permanent?.abilities_lost) for (const keyword of face?.keywords ?? []) out.set(keyword, { keyword, source: null });
  const effects = [...(permanent?.effects ?? [])].sort((a, b) => a.timestamp - b.timestamp);
  for (const effect of effects) {
    if (effect.modification === 'AddKeywords') {
      for (const keyword of effect.keywords ?? []) if (!out.has(keyword)) out.set(keyword, { keyword, source: effectLabel(effect) });
    }
    if (effect.modification === 'LoseKeywords') for (const keyword of effect.keywords ?? []) out.delete(keyword);
  }
  return [...out.values()];
}

/** "+2/+0", "−1/−1": a P/T change as printed in rules text. */
export function signed(power: number, toughness: number): string {
  const one = (value: number): string => (value < 0 ? `−${-value}` : `+${value}`);
  return `${one(power)}/${one(toughness)}`;
}
