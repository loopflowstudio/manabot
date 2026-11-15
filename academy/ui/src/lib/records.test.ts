import { afterEach, expect, it, vi } from 'vitest';
import { recordFetch } from './records';

function storage(initial: Record<string, string> = {}) {
  const values = new Map(Object.entries(initial));
  return {
    get length() { return values.size; },
    key: (index: number) => [...values.keys()][index] ?? null,
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value); },
  };
}
afterEach(() => vi.unstubAllGlobals());
it('includes this tab\'s Watch access without giving another tab its live credential', async () => {
  vi.stubGlobal('localStorage', storage({'etude.play.player-credential': 'player', 'etude.play.record-credentials': '["human-record"]'}));
  vi.stubGlobal('sessionStorage', storage({'etude.watch.match': 'watch-in-this-tab'}));
  const headers: Headers[] = [];
  vi.stubGlobal('fetch', vi.fn(async (_input: string, init: RequestInit) => { headers.push(new Headers(init.headers)); return Response.json([]); }));
  await recordFetch('/api/traces');
  expect(headers[0].get('x-etude-participant-tokens')).toBe('human-record,watch-in-this-tab');
  vi.stubGlobal('sessionStorage', storage());
  await recordFetch('/api/traces');
  expect(headers[1].get('x-etude-participant-tokens')).toBe('human-record');
  expect(headers[1].get('x-etude-participant-token')).toBe('human-record');
});
