/**
 * The turn as a bar of priority windows, with the stops set on each.
 *
 * A stop belongs to a step and a side (your turn or the opponent's, as
 * `StopsConfig` keeps them) and is in one of three states: off, every time,
 * or once. "Once" is the client's own automation over the same config: the
 * stop is set in `StopsConfig` and also listed in `OnceStops`; when the game
 * stops there the client clears it (`spentOnce`), and the move list says so.
 * The bar shows both sides' stops on every step, so a person can set the
 * opponent's end-step stop during their own turn; the side whose turn it is
 * decides which stop holds now.
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
  upkeep: 'Upkeep', draw: 'Draw', main1: 'Main', begin_combat: 'Combat', declare_attackers: 'Attackers',
  declare_blockers: 'Blockers', combat_damage: 'Damage', main2: 'Main 2', end_step: 'End',
};

/** The first step of each phase: the bar rules a line before it. */
const PHASE_STARTS = new Set(['upkeep', 'main1', 'begin_combat', 'main2', 'end_step']);

/** Stops that hold only the next time the game reaches them, by side. */
export type OnceStops = Record<StopSide, string[]>;

export type StopState = 'off' | 'every' | 'once';

export const STOP_STATE_WORDS: Record<StopState, string> = { off: 'off', every: 'every time', once: 'once' };

export interface PhaseMark {
  key: string;
  short: string;
  label: string;
  current: boolean;
  /** The first step of its phase. */
  phaseStart: boolean;
  /** Each side's stop here. */
  state: Record<StopSide, StopState>;
}

export interface PhaseTrack {
  marks: PhaseMark[];
  /** The side whose turn it is: its stops are the ones that hold now. */
  active: StopSide;
  /** Priority stops are off: every step holds, whatever is set. */
  every: boolean;
}

export function emptyOnce(): OnceStops {
  return { my: [], opponent: [] };
}

/** A pressed stop's next state: off → every time → once → off. */
export function nextStopState(state: StopState): StopState {
  return state === 'off' ? 'every' : state === 'every' ? 'once' : 'off';
}

function stateOf(stops: StopsConfig, once: OnceStops, side: StopSide, step: string): StopState {
  if (!stops[side].includes(step)) return 'off';
  return once[side].includes(step) ? 'once' : 'every';
}

/** The stops and once-list that put `step` on `side` in `state`. */
export function withStopState(stops: StopsConfig, once: OnceStops, side: StopSide, step: string, state: StopState): { stops: StopsConfig; once: OnceStops } {
  const steps = stops[side].filter(existing => existing !== step);
  const onceSteps = once[side].filter(existing => existing !== step);
  return {
    stops: { ...stops, [side]: state === 'off' ? steps : [...steps, step] },
    once: { ...once, [side]: state === 'once' ? [...onceSteps, step] : onceSteps },
  };
}

/** A once-list holds only steps the config still stops at. */
export function pruneOnce(once: OnceStops, stops: StopsConfig): OnceStops {
  return { my: once.my.filter(step => stops.my.includes(step)), opponent: once.opponent.filter(step => stops.opponent.includes(step)) };
}

/** The stop window a native step belongs to, if any. */
export function stopStepOf(step: string): string | null {
  return Object.entries(ENGINE_STEPS).find(([, steps]) => steps.includes(step))?.[0] ?? null;
}

/**
 * The once stop a decision spends: the game stopped for the person at a step
 * where the turn's side has a once stop. Clearing it is the client's
 * automation, so it is reported, never hidden.
 */
export function spentOnce(
  turn: Observation['turn'],
  yourTurn: boolean,
  stops: StopsConfig,
  once: OnceStops,
  actionSpace = 'PRIORITY',
): { side: StopSide; step: string } | null {
  if (actionSpace !== 'PRIORITY') return null;
  const step = stopStepOf(turn.step);
  const side: StopSide = yourTurn ? 'my' : 'opponent';
  if (!step || !stops.auto_pass || stateOf(stops, once, side, step) !== 'once') return null;
  return { side, step };
}

export function stopLabel(step: string): string {
  return STOP_STEPS.find(candidate => candidate.key === step)?.label ?? step;
}

/** The stop bar for a turn; `step` is the engine step the board shows, or null when the board is not at a step of this turn. */
export function phaseTrack(step: string | null, stops: StopsConfig, yourTurn: boolean, once: OnceStops = emptyOnce()): PhaseTrack {
  return {
    active: yourTurn ? 'my' : 'opponent',
    every: !stops.auto_pass,
    marks: STOP_STEPS.map(stop => ({
      key: stop.key,
      short: SHORT[stop.key],
      label: stop.label,
      current: step !== null && ENGINE_STEPS[stop.key].includes(step),
      phaseStart: PHASE_STARTS.has(stop.key),
      state: { my: stateOf(stops, once, 'my', stop.key), opponent: stateOf(stops, once, 'opponent', stop.key) },
    })),
  };
}
