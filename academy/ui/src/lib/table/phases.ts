/** The turn as a track of priority windows, with the stops that hold them. */
import { STOP_STEPS } from '$lib/stops';
import type { Observation, StopsConfig } from '$lib/types';

/** Native step names for each stop window (managym/academy/server.py). */
const ENGINE_STEPS: Record<string, string[]> = {
  upkeep: ['BEGINNING_UNTAP', 'BEGINNING_UPKEEP'],
  draw: ['BEGINNING_DRAW'],
  main1: ['PRECOMBAT_MAIN_STEP'],
  begin_combat: ['COMBAT_BEGIN'],
  declare_attackers: ['COMBAT_DECLARE_ATTACKERS', 'DECLARE_ATTACKERS_STEP'],
  declare_blockers: ['COMBAT_DECLARE_BLOCKERS', 'DECLARE_BLOCKERS_STEP'],
  combat_damage: ['COMBAT_DAMAGE', 'COMBAT_FIRST_STRIKE_DAMAGE', 'COMBAT_END'],
  main2: ['POSTCOMBAT_MAIN_STEP'],
  end_step: ['ENDING_END', 'END_STEP', 'ENDING_CLEANUP'],
};

const SHORT: Record<string, string> = {
  upkeep: 'Upkeep', draw: 'Draw', main1: 'Main', begin_combat: 'Combat', declare_attackers: 'Attack',
  declare_blockers: 'Block', combat_damage: 'Damage', main2: 'Main 2', end_step: 'End',
};

export interface PhaseMark {
  key: string;
  short: string;
  label: string;
  current: boolean;
  /** The active player's stop holds priority here. */
  stop: boolean;
}

export function phaseTrack(turn: Observation['turn'], stops: StopsConfig, yourTurn: boolean): PhaseMark[] {
  const held = new Set(stops.auto_pass ? (yourTurn ? stops.my : stops.opponent) : STOP_STEPS.map(step => step.key));
  return STOP_STEPS.map(step => ({
    key: step.key,
    short: SHORT[step.key],
    label: step.label,
    current: ENGINE_STEPS[step.key].includes(turn.step),
    stop: held.has(step.key),
  }));
}
