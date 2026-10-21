/** Presentation-only lanes over native card types; never derives legality. */
import type { PermanentState } from './types';

export type BattlefieldLane = 'creatures' | 'lands' | 'other' | 'tokens';
export type BattlefieldGroups = Record<BattlefieldLane, PermanentState[]>;

export function groupBattlefield(permanents: PermanentState[]): BattlefieldGroups {
  const groups: BattlefieldGroups = { creatures: [], lands: [], other: [], tokens: [] };
  for (const permanent of permanents) {
    const types = permanent.types;
    // Animated lands and artifact creatures belong with creatures. Historical
    // frames without type evidence stay inspectable in Other permanents.
    const lane = types?.is_creature ? 'creatures' : types?.is_land ? 'lands' :
      types?.is_artifact && permanent.is_token ? 'tokens' : 'other';
    groups[lane].push(permanent);
  }
  return groups;
}

export function battlefieldPageLayout(width: number, height: number, count: number): { cardWidth: number; capacity: number } {
  // Budget landscape width as well as portrait height, keeping tapped cards
  // inside their lane. Below 44px cards stop shrinking and the lane paginates.
  for (let cardWidth = 72; cardWidth >= 44; cardWidth -= 4) {
    const footprint = cardWidth * 1.4;
    const columns = Math.floor((width + 6) / (footprint + 6));
    const rows = Math.floor((height + 6) / (footprint + 6));
    if (columns < 1 || rows < 1) continue;
    const capacity = columns * rows;
    if (capacity >= count || cardWidth === 44) return { cardWidth, capacity };
  }
  return { cardWidth: 44, capacity: 1 };
}
