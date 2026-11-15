/** Shared wording for committed pools and proposed source output. */
import type { PooledManaView } from './types';

export function manaText(amounts: number[]): string {
  return amounts.map((amount, index) => amount ? `${amount} {${['W', 'U', 'B', 'R', 'G', 'C'][index]}}` : '').filter(Boolean).join(' · ') || 'Empty';
}

export function pooledManaText(pool: PooledManaView): string {
  return `${manaText(pool.mana)}${pool.restriction ? ` only for ${pool.restriction}` : ''}${pool.effect ? ` that puts ${pool.effect}` : ''}, normally expires ${pool.combat ? 'at end of combat' : 'at step end'}`;
}
