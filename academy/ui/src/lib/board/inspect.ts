/** Power/toughness provenance and rules text shared by Academy’s inspection surfaces. */
import type { CardFace } from '$lib/card-faces.svelte';
import type { CardState, ContinuousEffect, PermanentState } from '$lib/types';

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

export function effectLabel(effect: ContinuousEffect): string {
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
  if (counters) parts.push({ label: counterTerm('+1/+1', counters), power: counters, toughness: counters, base: false });
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

export function counterTerm(kind: string, count: number): string {
  return `${count} ${kind.replace(/_/g, ' ')} counter${count === 1 ? '' : 's'}`;
}

/** "2 stun counters", "1 stash counter": counters by kind, positive counts only. */
export function counterTerms(counters: Record<string, number> | undefined): string[] {
  return Object.entries(counters ?? {}).filter(([, n]) => n > 0).map(([kind, n]) => counterTerm(kind, n));
}

const SOURCE_CARD_TYPES = new Set([
  'artifact', 'battle', 'creature', 'enchantment',
  'instant', 'kindred', 'land', 'planeswalker', 'sorcery',
]);

/** Readable source quality phrase for hexproof-from and protection. */
export function sourceQuality(quality: string): string {
  return SOURCE_CARD_TYPES.has(quality) ? `${quality} sources` : quality;
}

/** Play permissions from the inspected card's viewer projection. */
export function permissionTerms(card: Pick<CardState, 'cast_permission' | 'cast_permissions'> | null, permanentName: (id: number) => string, playerName: (id: number) => string): string[] {
  const grants = card?.cast_permissions ?? (card?.cast_permission ? [card.cast_permission] : []);
  return grants.flatMap(grant => {
    const player = playerName(grant.player_id);
    const verb = grant.play_lands ? 'play' : 'cast';
    const terms: string[] = [];
    if (grant.source_permanent_id != null) {
      const source = permanentName(grant.source_permanent_id);
      terms.push(grant.follows_source_controller
        ? `The controller of ${source} may ${verb} this (currently ${player}).`
        : `${player} may ${verb} this while they control ${source}.`);
    }
    if (grant.flashback) {
      const duration = grant.this_turn ? ' this turn' : '';
      terms.push(`${player} may cast this with flashback${duration}. Exile it when it leaves the stack.`);
    }
    const rest: string[] = [];
    if (grant.cost != null) rest.push(`for ${grant.cost}`);
    if (grant.on_adventure) rest.push('as the permanent spell only');
    if (grant.this_turn && !grant.flashback) rest.push('this turn only');
    if (grant.through_next_turn) rest.push('until the end of their next turn');
    if (grant.own_turn_only) rest.push('during their own turn only');
    if (grant.after_turn != null) rest.push(`only after turn ${grant.after_turn}`);
    if (grant.flash) rest.push('as though it had flash');
    if (grant.any_mana) rest.push('spending mana of any type');
    if (grant.waterbend) rest.push('paying generic mana by waterbending');
    if (grant.foretold) rest.push('foretold, face down');
    if (grant.face_down) rest.push('face down, seen only while it may be played');
    if (grant.prepared) rest.push('as a prepared copy kept while linked to its source');
    if (grant.condition) rest.push(`only if ${grant.condition}`);
    if (rest.length || !terms.length) {
      terms.push(`${player} may ${verb} this${rest.length ? ' ' + rest.join(', ') : ''}.`);
    }
    return terms;
  });
}
