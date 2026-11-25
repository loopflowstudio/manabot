import type { StopsConfig } from './types';

// Stop steps mirror gui/server.py STOP_STEP_TO_ENGINE_STEP: the priority
// windows a player can choose to stop on, in turn order.
export const STOP_STEPS = [
  { key: 'upkeep', label: 'Upkeep' },
  { key: 'draw', label: 'Draw' },
  { key: 'main1', label: 'First main' },
  { key: 'begin_combat', label: 'Beginning of combat' },
  { key: 'declare_attackers', label: 'Declare attackers' },
  { key: 'declare_blockers', label: 'Declare blockers' },
  { key: 'combat_damage', label: 'Combat damage' },
  { key: 'main2', label: 'Second main' },
  { key: 'end_step', label: 'End step' },
] as const;

const STOP_STEP_KEYS = new Set<string>(STOP_STEPS.map((step) => step.key));

export const STOPS_STORAGE_KEY = 'etude.gui.stops';
export const STOPS_DEFAULTS_VERSION_KEY = 'academy.stops.defaults_version';
const STOPS_DEFAULTS_VERSION = '2';

const LEGACY_V1_MY = ['main1', 'main2'];
const LEGACY_V1_OPPONENT = ['end_step'];

// Mirrors the server defaults (managym/academy/server.py DEFAULT_STOPS): act in
// your own main phases, pause in active combat steps when you have a non-mana
// play, hold up interaction at the opponent's end step, and stop when the stack
// is non-empty.
export function defaultStops(): StopsConfig {
  return {
    my: ['main1', 'declare_attackers', 'declare_blockers', 'main2'],
    opponent: ['declare_attackers', 'declare_blockers', 'end_step'],
    stop_on_stack: true,
    auto_pass: true,
  };
}

function sameSteps(actual: readonly string[], expected: readonly string[]): boolean {
  return actual.length === expected.length && expected.every((step) => actual.includes(step));
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
    const parsed = sanitizeStops(JSON.parse(raw));
    if (!parsed) {
      return defaultStops();
    }
    const version = localStorage.getItem(STOPS_DEFAULTS_VERSION_KEY);
    if (
      version !== STOPS_DEFAULTS_VERSION &&
      sameSteps(parsed.my, LEGACY_V1_MY) &&
      sameSteps(parsed.opponent, LEGACY_V1_OPPONENT)
    ) {
      const defaults = defaultStops();
      const migrated: StopsConfig = {
        ...parsed,
        my: defaults.my,
        opponent: defaults.opponent,
      };
      saveStoredStops(migrated);
      return migrated;
    }
    return parsed;
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
    localStorage.setItem(STOPS_DEFAULTS_VERSION_KEY, STOPS_DEFAULTS_VERSION);
  } catch {
    // Storage full or unavailable: stops simply won't persist.
  }
}

/** Stops set to hold once (`OnceStops`); the config holds them as ordinary stops. */
export const ONCE_STORAGE_KEY = 'academy.stops.once';

export function loadStoredOnce(): Record<'my' | 'opponent', string[]> {
  const empty = { my: [], opponent: [] };
  if (typeof localStorage === 'undefined') return empty;
  try {
    const raw = JSON.parse(localStorage.getItem(ONCE_STORAGE_KEY) ?? 'null') as Record<string, unknown> | null;
    const my = sanitizeSteps(raw?.my);
    const opponent = sanitizeSteps(raw?.opponent);
    return my && opponent ? { my, opponent } : empty;
  } catch {
    return empty;
  }
}

export function saveStoredOnce(once: Record<'my' | 'opponent', string[]>): void {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(ONCE_STORAGE_KEY, JSON.stringify(once));
  } catch {
    // Storage full or unavailable: once stops simply won't persist.
  }
}
