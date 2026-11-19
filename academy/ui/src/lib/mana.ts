/** Wording for committed floating mana already present in observations. */
import type { PlayerState, RestrictedMana } from './types';

type PooledManaView = NonNullable<PlayerState['pooled_mana_classes']>[number];

/** A line split into plain words and runs of braced mana symbols: "Pay {2}{R} now" → "Pay ", "{2}{R}", " now". */
export function manaRuns(text: string): { text: string; mana: boolean }[] {
  return text.split(/((?:\{[^{}]+\})+)/).filter(Boolean).map(part => ({ text: part, mana: part.startsWith('{') }));
}

export function manaText(amounts: number[]): string {
  return amounts.map((amount, index) => amount ? `${amount} {${['W', 'U', 'B', 'R', 'G', 'C'][index]}}` : '').filter(Boolean).join(' · ') || 'Empty';
}

/** What sets a class of pooled mana apart, after its amount: "only for Lesson spells, normally expires at step end". */
export function pooledManaTerms(pool: PooledManaView): string {
  const sets = [pool.restriction && `only for ${pool.restriction}`, pool.effect && `that puts ${pool.effect}`].filter(Boolean).join(' ');
  return `${sets ? `${sets}, ` : ''}normally expires ${pool.combat ? 'at end of combat' : 'at step end'}`;
}

/** What restricted mana may be spent on: "Lesson spells", "abilities of Shrine sources". */
export function spendingText(restricted: RestrictedMana): string {
  if (restricted.except_nonartifact_spells) return 'anything but nonartifact spells';
  const parts = [];
  if (restricted.spells?.length) parts.push(`${restricted.spells.join(' or ')} spells`);
  if (restricted.abilities?.length) parts.push(`abilities of ${restricted.abilities.join(' or ')} sources`);
  return parts.join(' or ');
}

/**
 * Everything floating, by colour (WUBRGC). Native `mana` holds unrestricted
 * mana only; restricted mana is counted separately in `lesson_mana`, so the
 * floating total is their sum.
 */
export function floatingTotal(player: PlayerState): number[] {
  return (player.mana ?? []).map((amount, color) => amount + (player.lesson_mana?.[color] ?? 0));
}
