import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { WatchStart } from './watch-start.svelte';

beforeEach(() => {
  vi.useFakeTimers();
  const saved = new Map<string, string>();
  vi.stubGlobal('sessionStorage', {getItem: (key: string) => saved.get(key) ?? null, setItem: (key: string, value: string) => saved.set(key, value), removeItem: (key: string) => saved.delete(key), length: 0});
  vi.stubGlobal('localStorage', {getItem: () => 'reader'});
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });
const created = {id: 'recording', token: 'producer-capability', finished: false};

it('retains a lost setup reply and checks that attempt instead of loading another pair', async () => {
  let posts = 0;
  vi.stubGlobal('fetch', vi.fn(async (_url: string, init: RequestInit) => {
    if (init.method === 'POST') { posts++; throw new Error('Lost start reply'); }
    return Response.json({work: null, created, error: null});
  }));
  const start = new WatchStart();
  await expect(start.start({players: ['a', 'b']})).rejects.toThrow('Lost start reply');
  const envelope = sessionStorage.getItem('academy.watch-setup');
  expect(envelope).not.toBeNull();
  // Even changed form state cannot replace the unresolved attempt.
  expect(await new WatchStart().start({players: ['changed', 'changed']})).toEqual(created);
  expect(posts).toBe(1);
  expect(sessionStorage.getItem('etude.watch.recording')).toBe('producer-capability');
  expect(sessionStorage.getItem('academy.watch-setup')).toBeNull();
});

it('shows loading seats, then retains exactly one producer result', async () => {
  let finish!: (result: Response) => void;
  const start = new WatchStart();
  const fetcher = vi.fn(async (_url: string, init: RequestInit) => {
    if (init.method === 'POST') return new Promise<Response>(resolve => finish = resolve);
    const {handle} = JSON.parse(sessionStorage.getItem('academy.watch-setup')!);
    return Response.json({work: {scope: 'setup', setup_id: handle, loading_seat: 1, operation_id: 'op', sequence: 2, kind: 'load', started_at: 1, match_id: null, revision: null, chooser: null, state: 'running'}, created: null, error: null});
  });
  vi.stubGlobal('fetch', fetcher);
  const pending = start.start({players: ['a', 'b']});
  expect(start.start({players: ['x', 'y']})).toBe(pending);
  await vi.advanceTimersByTimeAsync(1000);
  expect(start.work?.loading_seat).toBe(1);
  expect(start.work?.chooser).toBeNull();
  finish(Response.json({work: null, created, error: null}));
  expect(await pending).toEqual(created);
  expect(fetcher.mock.calls.filter(([, init]) => init.method === 'POST')).toHaveLength(1);
  expect(start.work).toBeNull();
});

it('reports missing setup after restart without silently starting it again', async () => {
  sessionStorage.setItem('academy.watch-setup', JSON.stringify({handle: 'missing-handle', token: 'setup-credential', request: {players: ['a', 'b']}}));
  const fetcher = vi.fn(async () => Response.json({detail: 'Watch setup unavailable. Start again explicitly.'}, {status: 404}));
  vi.stubGlobal('fetch', fetcher);
  await expect(new WatchStart().start({})).rejects.toThrow('Start again explicitly');
  expect(fetcher).toHaveBeenCalledTimes(1);
  expect(sessionStorage.getItem('academy.watch-setup')).toBeNull();
});
