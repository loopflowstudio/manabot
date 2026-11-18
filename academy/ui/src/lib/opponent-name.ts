/**
 * An opponent's display name with how it decides: "Search 64 · search",
 * "Heuristic · scripted", "<checkpoint> · checkpoint w17". The server names a
 * running table's opponent (`TableSnapshot.opponent_label`) and each catalog
 * policy's kind; built-in players are named here, before any table exists.
 */
import type { PlayerChoice } from './game.svelte';
import type { TrainedOpponent } from './types';

export function builtinOpponentName(choice: PlayerChoice): string | null {
  if (choice.startsWith('search-')) return `Search ${choice.slice('search-'.length)} · search`;
  if (choice === 'random') return 'Random';
  if (choice === 'passive') return 'Passive · scripted';
  return null;
}

export function catalogOpponentName(entry: Pick<TrainedOpponent, 'name' | 'kind'>): string {
  return entry.kind ? `${entry.name} · ${entry.kind}` : entry.name;
}
