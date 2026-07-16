import type { StopsConfig } from './types';

// Stop steps mirror gui/server.py STOP_STEP_TO_ENGINE_STEP: the priority
// windows a player can choose to stop on, in turn order.
export const STOP_STEPS = [
  { key: 'upkeep', label: 'Upkeep' },
  { key: 'draw', label: 'Draw' },
  { key: 'main1', label: 'Main 1' },
  { key: 'begin_combat', label: 'Begin combat' },
  { key: 'declare_attackers', label: 'Attackers' },
  { key: 'declare_blockers', label: 'Blockers' },
  { key: 'combat_damage', label: 'Damage' },
  { key: 'main2', label: 'Main 2' },
  { key: 'end_step', label: 'End step' },
] as const;

export type StopStepKey = (typeof STOP_STEPS)[number]['key'];

const STOP_STEP_KEYS = new Set<string>(STOP_STEPS.map((step) => step.key));

export const STOPS_STORAGE_KEY = 'etude.gui.stops';

// Mirrors the server defaults (gui/server.py DEFAULT_STOPS): act in your own
// main phases, hold up interaction at the opponent's end step, always stop
// when the stack is non-empty.
export function defaultStops(): StopsConfig {
  return {
    my: ['main1', 'main2'],
    opponent: ['end_step'],
    stop_on_stack: true,
    auto_pass: true,
  };
}

function sanitizeSteps(value: unknown): string[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const steps: string[] = [];
  for (const step of value) {
    if (typeof step !== 'string' || !STOP_STEP_KEYS.has(step)) {
      return null;
    }
    if (!steps.includes(step)) {
      steps.push(step);
    }
  }
  return steps;
}

export function sanitizeStops(raw: unknown): StopsConfig | null {
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const candidate = raw as Record<string, unknown>;
  const my = sanitizeSteps(candidate.my);
  const opponent = sanitizeSteps(candidate.opponent);
  if (my === null || opponent === null) {
    return null;
  }
  if (
    typeof candidate.stop_on_stack !== 'boolean' ||
    typeof candidate.auto_pass !== 'boolean'
  ) {
    return null;
  }
  return {
    my,
    opponent,
    stop_on_stack: candidate.stop_on_stack,
    auto_pass: candidate.auto_pass,
  };
}

export function loadStoredStops(): StopsConfig {
  if (typeof localStorage === 'undefined') {
    return defaultStops();
  }
  try {
    const raw = localStorage.getItem(STOPS_STORAGE_KEY);
    if (!raw) {
      return defaultStops();
    }
    return sanitizeStops(JSON.parse(raw)) ?? defaultStops();
  } catch {
    return defaultStops();
  }
}

export function saveStoredStops(stops: StopsConfig): void {
  if (typeof localStorage === 'undefined') {
    return;
  }
  try {
    localStorage.setItem(STOPS_STORAGE_KEY, JSON.stringify(stops));
  } catch {
    // Storage full or unavailable: stops simply won't persist.
  }
}
