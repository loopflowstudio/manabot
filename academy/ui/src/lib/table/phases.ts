/**
 * The turn as a track of priority windows, with the stops set on each.
 *
 * A stop belongs to a step and a side (your turn or the opponent's, as
 * `StopsConfig` keeps them). The track shows both sides' stops on every
 * step, so a person can set the opponent's end-step stop during their own
 * turn; the side whose turn it is decides which stop holds now.
 */
import { STOP_STEPS } from '$lib/stops';
import type { Observation, StopsConfig, StopSide } from '$lib/types';

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
  /** Whether each side's stop is set here, as configured. */
  set: Record<StopSide, boolean>;
}

export interface PhaseTrack {
  marks: PhaseMark[];
  /** The side whose turn it is: its stops are the ones that hold now. */
  active: StopSide;
  /** Priority stops are off: every step holds, whatever is set. */
  every: boolean;
}

export function phaseTrack(turn: Observation['turn'], stops: StopsConfig, yourTurn: boolean): PhaseTrack {
  const my = new Set(stops.my);
  const opponent = new Set(stops.opponent);
  return {
    active: yourTurn ? 'my' : 'opponent',
    every: !stops.auto_pass,
    marks: STOP_STEPS.map(step => ({
      key: step.key,
      short: SHORT[step.key],
      label: step.label,
      current: ENGINE_STEPS[step.key].includes(turn.step),
      set: { my: my.has(step.key), opponent: opponent.has(step.key) },
    })),
  };
}
