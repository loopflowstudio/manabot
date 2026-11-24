/** Wording for committed floating mana already present in observations. */
import type { PlayerState, RestrictedMana } from './types';

type PooledManaView = NonNullable<PlayerState['pooled_mana_classes']>[number];

export function manaText(amounts: number[]): string {
  return amounts.map((amount, index) => amount ? `${amount} {${['W', 'U', 'B', 'R', 'G', 'C'][index]}}` : '').filter(Boolean).join(' · ') || 'Empty';
}

export function pooledManaText(pool: PooledManaView): string {
  return `${manaText(pool.mana)}${pool.restriction ? ` ${spendPhrase(pool.restriction)}` : ''}${pool.effect ? ` that puts ${pool.effect}` : ''}, normally expires ${pool.combat ? 'at end of combat' : 'at step end'}`;
}

/** "only for Lesson spells", but "for anything but spells from hand": an exclusion needs no "only". */
export function spendPhrase(spending: string): string {
  return spending.startsWith('anything but ') ? `for ${spending}` : `only for ${spending}`;
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
