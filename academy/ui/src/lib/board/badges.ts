/** The effective facts on one permanent. The board fits their short labels;
 * Inspect and the card's accessible name keep every fact and its provenance. */
import type { CardFace } from '$lib/card-faces.svelte';
import type { PermanentRestriction, PermanentState } from '$lib/types';
import { counterTerm, effectLabel, labelText, sourceQuality } from './inspect';

export interface Badge {
  label: string;
  shortLabel?: string;
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

const shortStatuses: Record<string, string> = {
  [restrictions.cant_be_blocked]: 'Unblockable',
  [restrictions.cant_attack_or_block]: 'No combat',
  [restrictions.doesnt_untap]: 'No untap',
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
  const fromRecord = (timestamp: number | undefined): string => {
    const record = effects.find(effect => effect.timestamp === timestamp);
    return record ? effectLabel(record) : 'Another ability';
  };
  const abilities: Badge[] = (p.keywords ?? []).map(keyword => {
    let source: string | undefined;
    const counters = p.counters?.[keyword] ?? 0;
    if (p.keyword_grants) {
      const grant = p.keyword_grants[keyword];
      if (grant) source = grant.counter ? counterTerm(keyword, counters) : grant.source ?? fromRecord(grant.effect_timestamp);
    } else if (counters > 0) source = counterTerm(keyword, counters);
    else {
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
    const source = trigger.granted ? trigger.source ?? fromRecord(trigger.effect_timestamp) : undefined;
    const badge: Badge = { label: trigger.label, kind: 'ability', source, granted: trigger.granted };
    if (badge.label.startsWith('Firebending ')) badge.shortLabel = badge.label.replace('Firebending', 'Firebend');
    if (badge.label.startsWith('Ward—')) badge.shortLabel = 'Ward';
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

export function badgesOf(p: PermanentState, face: CardFace | null = null, blocks?: string | null, enteredUnder?: string | null): Badge[] {
  const badges: Badge[] = [];
  const abilities = abilitiesOf(p, face);
  if (p.damage) badges.push({ kind: 'damage', label: `${p.damage} damage`, detail: `${p.damage} damage marked` });
  if (p.summoning_sick && (p.types?.is_creature ?? p.power !== null)) {
    badges.push({ kind: 'sick', label: 'Summoning sick' });
  }

  // Static rules and resolved effects can give the same restriction. Keep
  // one board label and every named source/duration in its inspection.
  const statuses = new Map<string, Set<string>>();
  function addStatus(label: string, source?: string) {
    const sources = statuses.get(label) ?? new Set<string>();
    if (source) sources.add(source);
    statuses.set(label, sources);
  }
  for (const rule of p.restrictions ?? []) addStatus(restrictions[rule.kind], rule.source);
  for (const effect of p.effects ?? []) {
    const label = recordStatuses[effect.modification];
    if (label) addStatus(label, effectLabel(effect));
  }
  for (const [label, sources] of statuses) {
    badges.push({ kind: 'status', label, shortLabel: shortStatuses[label], source: [...sources].join(' · ') || undefined });
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
    ...(p.hexproof_from ?? []).map(quality => `hexproof from ${sourceQuality(quality)}`),
    ...(p.protection_from ?? []).map(quality => `protection from ${sourceQuality(quality)}`),
    p.blocked && 'blocked',
    p.entry_controller_id !== undefined && p.entry_controller_id !== p.controller_id
      && `entered under ${enteredUnder ?? `player #${p.entry_controller_id}'s`} control`,
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

interface FittedChip {
  badge: Badge | null; // null is the overflow count
  label: string;
}

interface Position {
  row: number;
  used: number;
}

/** Each width describes a physical row from top to bottom. The P/T plate can
 * narrow the final row. Inspect keeps the full reading of every compact chip. */
export function fitBadges(badges: Badge[], widths: number[], measure: (text: string) => number) {
  const board = badges.filter(b => !b.inspectOnly);
  if (!board.length || !widths.length) return { rows: [], hidden: board };
  const widest = Math.max(...widths);
  const chips = board.map(badge => {
    const full = badge.kind === 'sick' ? 16 : measure(badge.label);
    if (full > widest && badge.shortLabel) {
      return { badge, label: badge.shortLabel, width: measure(badge.shortLabel) };
    }
    return { badge, label: badge.label, width: full };
  });

  function position(after: Position, width: number): Position | undefined {
    for (let row = after.row; row < widths.length; row++) {
      const used = (row === after.row && after.used ? after.used + 2 : 0) + width;
      if (used <= widths[row]) return { row, used };
    }
  }

  function pack(reserveCount: boolean) {
    const rows: FittedChip[][] = widths.map(() => []);
    const hidden: Badge[] = [];
    let cursor = { row: 0, used: 0 };
    let remaining = chips.length;
    for (const chip of chips) {
      const next = position(cursor, chip.width);
      if (next && (!reserveCount || position(next, measure(`+${remaining - 1}`)))) {
        cursor = next;
        rows[cursor.row].push(chip);
        remaining--;
      } else {
        hidden.push(chip.badge);
      }
    }
    if (reserveCount && hidden.length) {
      const label = `+${hidden.length}`;
      const next = position(cursor, measure(label));
      if (!next) return { rows: [], hidden: board }; // Count beside the name.
      cursor = next;
      rows[next.row].push({ label, badge: null });
    }
    // Anchor occupied rows at the foot of the art. Leave the plate's row
    // vacant when the last line needs the full card width.
    rows.length = cursor.row + 1;
    if (cursor.used > widths[widths.length - 1]) rows.push([]);
    return { rows, hidden };
  }

  const all = pack(false);
  return all.hidden.length ? pack(true) : all;
}

let canvas: CanvasRenderingContext2D | null = null;

/** Use the chip's inherited typeface. Keep one canvas for the entire table. */
export function chipWidth(text: string, font: string): number {
  if (typeof CanvasRenderingContext2D !== 'undefined') canvas ??= document.createElement('canvas').getContext('2d');
  if (canvas) canvas.font = font;
  const ink = canvas ? canvas.measureText(text).width : text.length * 10;
  return Math.ceil(ink) + 10; // 4px padding on each side plus 2px of slack
}
