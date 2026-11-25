/** The effective facts on one permanent. The board fits their short labels;
 * Inspect and the card's accessible name keep every fact and its provenance. */
import type { CardFace } from '$lib/card-faces.svelte';
import type { PermanentRestriction, PermanentState } from '$lib/types';
import { counterTerm, effectLabel, labelText } from './inspect';

export interface Badge {
  label: string;
  kind: 'status' | 'ability' | 'damage' | 'sick' | 'counter' | 'state';
  /** A longer reading for Inspect and speech. */
  detail?: string;
  source?: string;
  granted?: boolean;
  overridden?: boolean;
  inspectOnly?: boolean;
}

const restrictions: Record<PermanentRestriction, string> = {
  cant_be_blocked: "Can't be blocked",
  cant_block: "Can't block",
  cant_attack_or_block: "Can't attack or block",
  doesnt_untap: "Doesn't untap",
  must_attack: 'Must attack',
};

const recordStatuses: Record<string, string> = {
  CantBeBlocked: restrictions.cant_be_blocked,
  CantBlock: restrictions.cant_block,
  DoesntUntap: restrictions.doesnt_untap,
  Saddled: 'Saddled',
  CantActivateAbilities: "Can't activate abilities",
  MustBeBlockedByAllAble: 'Must be blocked by all able',
  ProtectionFromEverything: 'Protection from everything',
  PreventCombatDamage: 'Prevent combat damage to this',
  ExileIfWouldDie: 'Exile if it would die',
  ExileIfWouldLeave: 'Exile if it would leave',
};

function abilitiesOf(p: PermanentState, face: CardFace | null): Badge[] {
  const effects = p.effects ?? [];
  const fromRecord = (timestamp: number | null | undefined): string => {
    const record = effects.find(effect => effect.timestamp === timestamp);
    return record ? effectLabel(record) : 'Another ability';
  };
  const abilities: Badge[] = (p.keywords ?? []).map(keyword => {
    let source: string | undefined;
    const counters = p.counters?.[keyword] ?? 0;
    // A keyword counter grants its keyword (CR 122.1b); the grant has no record.
    if (counters > 0 && (p.keyword_grants?.[keyword] ?? null) === null) source = counterTerm(keyword, counters);
    else if (p.keyword_grants) {
      if (Object.hasOwn(p.keyword_grants, keyword)) source = fromRecord(p.keyword_grants[keyword]);
    } else {
      // Older recordings contain the effective keywords, but only record grants have a source.
      const record = effects.find(e => e.modification === 'AddKeywords' && e.keywords?.includes(keyword));
      if (record) source = effectLabel(record);
      else if (p.abilities_lost || p.face_down || (face && !face.keywords.includes(keyword))) source = 'Another ability';
    }
    const overridden = keyword === 'defender' && p.attacks_despite_defender;
    return {
      label: labelText(keyword), kind: 'ability', source, granted: !!source, inspectOnly: keyword === 'flash',
      overridden, detail: overridden ? 'Defender · can attack anyway' : undefined,
    };
  });
  const triggers = p.keyword_abilities ?? (p.granted_firebending
    ? [{ label: `Firebending ${p.granted_firebending}`, granted: true,
      effect_timestamp: effects.find(e => e.modification === 'Firebending' && e.amount === p.granted_firebending)?.timestamp }]
    : []);
  // Prowess instances stack. Combine only instances with the same provenance.
  const prowess = new Map<string | undefined, { badge: Badge; count: number }>();
  for (const trigger of triggers) {
    const source = trigger.granted ? fromRecord(trigger.effect_timestamp) : undefined;
    const badge: Badge = { label: trigger.label, kind: 'ability', source, granted: trigger.granted };
    if (trigger.label === 'Prowess') {
      const group = prowess.get(source);
      if (group) {
        group.badge.label = `Prowess ×${++group.count}`;
        continue;
      }
      prowess.set(source, { badge, count: 1 });
    }
    abilities.push(badge);
  }
  return abilities;
}

export function badgesOf(p: PermanentState, face: CardFace | null = null, blocks?: string | null): Badge[] {
  const badges: Badge[] = [];
  const abilities = abilitiesOf(p, face);
  if (p.damage) badges.push({ kind: 'damage', label: `${p.damage} damage`, detail: `${p.damage} damage marked` });
  if (p.summoning_sick && (p.types?.is_creature ?? p.power !== null)) {
    badges.push({ kind: 'sick', label: 'Summoning sick' });
  }

  // Static rules and resolved effects can give the same restriction. Keep
  // one board label and every named source/duration in its inspection.
  const statuses = new Map<string, Set<string>>();
  for (const restriction of p.restrictions ?? []) statuses.set(restrictions[restriction], new Set());
  for (const effect of p.effects ?? []) {
    const label = recordStatuses[effect.modification];
    if (!label) continue;
    const sources = statuses.get(label) ?? new Set<string>();
    sources.add(effectLabel(effect));
    statuses.set(label, sources);
  }
  for (const [label, sources] of statuses) {
    badges.push({ kind: 'status', label, source: sources.size ? [...sources].join(' · ') : 'Static ability' });
  }
  if (p.abilities_lost) {
    const loss = p.effects?.findLast(e => e.modification === 'LoseAllAbilities' || e.modification === 'LoseNonManaAbilities');
    badges.push({ kind: 'status', label: abilities.length ? 'Lost base abilities' : 'No abilities',
      source: loss ? effectLabel(loss) : undefined });
  }
  if (p.first_strike_participant) badges.push({ kind: 'status', label: 'Struck first', detail: 'first-strike participant' });
  const counters = { '+1/+1': p.plus1_counters, ...p.counters };
  for (const [kind, amount] of Object.entries(counters)) {
    if (amount <= 0) continue;
    // The keyword chip already names a keyword counter as its source.
    badges.push({ kind: 'counter', label: kind === '+1/+1' ? `+${amount}/+${amount}` : `${labelText(kind)} ×${amount}`,
      detail: counterTerm(kind, amount), inspectOnly: p.keywords?.includes(kind) });
  }
  badges.push(...abilities);
  const states = [
    p.tapped ? 'Tapped' : 'Untapped',
    p.attacking && 'attacking',
    p.blocking !== undefined && `blocking ${blocks ?? 'an attacker'}`,
    p.entered_this_turn && 'entered this turn',
    p.attacked_this_turn && 'attacked this turn',
    p.is_token && 'token',
    ...(p.combat_damage_to_players ?? []).map(seat => `dealt combat damage to seat ${seat} this turn`),
  ];
  for (const label of states) if (label) badges.push({ label, kind: 'state', inspectOnly: true });
  return badges;
}

export function badgeReading(badge: Badge): string {
  return [badge.detail ?? badge.label, badge.source].filter(Boolean).join(' · ');
}

/** Greedy in reading order. Skip oversized words, then make room for the
 * exact overflow count (which can gain a digit as we remove the last chip). */
export function fitBadges(badges: Badge[], width: number, rows: number, measure: (text: string) => number) {
  const visible: { badge: Badge; row: number; used: number }[] = [];
  const board = badges.filter(badge => !badge.inspectOnly);
  function place(size: number) {
    const last = visible.at(-1);
    if (!last) return { row: 1, used: size };
    const used = last.used + 2 + size;
    return used <= width ? { row: last.row, used } : { row: last.row + 1, used: size };
  }
  for (const badge of board) {
    const size = badge.kind === 'sick' ? 16 : measure(badge.label);
    const position = place(size);
    if (size <= width && position.row <= rows) visible.push({ badge, ...position });
  }
  let hidden = board.length - visible.length;
  while (hidden && visible.length && place(measure(`+${hidden}`)).row > rows) {
    visible.pop();
    hidden++;
  }
  return { visible: visible.map(item => item.badge), hidden };
}

let canvas: CanvasRenderingContext2D | null = null;

/** Use the chip's inherited typeface. Keep one canvas for the entire table. */
export function chipWidth(text: string, font: string): number {
  if (typeof CanvasRenderingContext2D !== 'undefined') canvas ??= document.createElement('canvas').getContext('2d');
  if (canvas) canvas.font = font;
  const ink = canvas ? canvas.measureText(text).width : text.length * 10;
  return Math.ceil(ink) + 12; // 5px padding on each side plus 2px of slack
}
