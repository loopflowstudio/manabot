import { afterEach, describe, expect, it } from 'vitest';

import {
  defaultStops,
  loadStoredStops,
  sanitizeStops,
  saveStoredStops,
  STOP_STEPS,
  STOPS_STORAGE_KEY,
} from './stops';

function stubLocalStorage(): Map<string, string> {
  const backing = new Map<string, string>();
  (globalThis as { localStorage?: unknown }).localStorage = {
    getItem: (key: string) => backing.get(key) ?? null,
    setItem: (key: string, value: string) => {
      backing.set(key, value);
    },
    removeItem: (key: string) => {
      backing.delete(key);
    },
  };
  return backing;
}

afterEach(() => {
  delete (globalThis as { localStorage?: unknown }).localStorage;
});

describe('stops config', () => {
  it('defaults to main phases plus opponent end step with stack stops on', () => {
    expect(defaultStops()).toEqual({
      my: ['main1', 'main2'],
      opponent: ['end_step'],
      stop_on_stack: true,
      auto_pass: true,
    });
  });

  it('sanitizes well-formed configs and rejects unknown steps', () => {
    const valid = {
      my: ['upkeep', 'main1'],
      opponent: [],
      stop_on_stack: false,
      auto_pass: true,
    };
    expect(sanitizeStops(valid)).toEqual(valid);

    expect(sanitizeStops(null)).toBeNull();
    expect(sanitizeStops({ my: ['untap'], opponent: [], stop_on_stack: true, auto_pass: true })).toBeNull();
    expect(sanitizeStops({ my: 'main1', opponent: [], stop_on_stack: true, auto_pass: true })).toBeNull();
    expect(sanitizeStops({ my: [], opponent: [], stop_on_stack: 'yes', auto_pass: true })).toBeNull();
  });

  it('accepts every advertised stop step', () => {
    const all = STOP_STEPS.map((step) => step.key);
    const config = { my: all, opponent: all, stop_on_stack: true, auto_pass: false };
    expect(sanitizeStops(config)).toEqual(config);
  });

  it('round-trips through localStorage and falls back on garbage', () => {
    const backing = stubLocalStorage();

    const custom = {
      my: ['end_step'],
      opponent: ['upkeep', 'draw'],
      stop_on_stack: false,
      auto_pass: true,
    };
    saveStoredStops(custom);
    expect(loadStoredStops()).toEqual(custom);

    backing.set(STOPS_STORAGE_KEY, 'not-json{');
    expect(loadStoredStops()).toEqual(defaultStops());

    backing.set(STOPS_STORAGE_KEY, JSON.stringify({ my: ['bogus_step'] }));
    expect(loadStoredStops()).toEqual(defaultStops());
  });

  it('falls back to defaults without localStorage', () => {
    expect(loadStoredStops()).toEqual(defaultStops());
  });
});
