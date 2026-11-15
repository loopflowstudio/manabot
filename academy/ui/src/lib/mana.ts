/** Wording for committed floating mana already present in observations. */
import type { PlayerState } from './types';

type PooledManaView = NonNullable<PlayerState['pooled_mana_classes']>[number];

export function manaText(amounts: number[]): string {
  return amounts.map((amount, index) => amount ? `${amount} {${['W', 'U', 'B', 'R', 'G', 'C'][index]}}` : '').filter(Boolean).join(' · ') || 'Empty';
}

export function pooledManaText(pool: PooledManaView): string {
  return `${manaText(pool.mana)}${pool.restriction ? ` only for ${pool.restriction}` : ''}${pool.effect ? ` that puts ${pool.effect}` : ''}, normally expires ${pool.combat ? 'at end of combat' : 'at step end'}`;
}
