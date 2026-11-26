import { afterEach, describe, expect, it, vi } from 'vitest';

import { createReplayStore } from './replay.svelte';
import type { Observation } from './types';

function makeObservation(turnNumber: number, gameOver = false): Observation {
  return {
    game_over: gameOver,
    won: false,
    turn: {
      turn_number: turnNumber,
      phase: 'PRECOMBAT_MAIN',
      step: 'PRIORITY',
      active_player_id: 10,
      agent_player_id: 10,
    },
    agent: {
      player_index: 0,
      id: 10,
      is_active: true,
      is_agent: true,
      life: 20,
      zone_counts: { HAND: 1, LIBRARY: 39, GRAVEYARD: 0, EXILE: 0, STACK: 0 },
      library_count: 39,
      hand_hidden_count: 0,
      hand: [],
      graveyard: [],
      exile: [],
      stack: [],
      battlefield: [],
    },
    opponent: {
      player_index: 1,
      id: 20,
      is_active: false,
      is_agent: false,
      life: 20,
      zone_counts: { HAND: 1, LIBRARY: 39, GRAVEYARD: 0, EXILE: 0, STACK: 0 },
      library_count: 39,
      hand_hidden_count: 1,
      hand: [],
      graveyard: [],
      exile: [],
      stack: [],
      battlefield: [],
    },
  };
}

import { orientBoard } from './recording';
import type { RecordedPosition, RecordingIndex } from './recording';

const positions: RecordingIndex['positions'] = [
  {key: 'exact-choice', kind: 'choice', turn: 1, phase: 'MAIN', step: 'PRIORITY', actor: 1, automatic: false, can_reveal: false},
  {key: 'after:0', kind: 'consequences', turn: 2, phase: 'COMBAT', step: 'DAMAGE', actor: null, automatic: false, can_reveal: false},
  {key: 'latest', kind: 'state', turn: 2, phase: 'COMBAT', step: 'DAMAGE', actor: null, automatic: false, can_reveal: false},
];
function index(canProduce = false): RecordingIndex {
  return {id: 'trace-1', positions: [...positions], total: 3, next_offset: null, revision: 91, status: canProduce ? 'active' : 'completed', can_produce: canProduce, can_study: true, winner: 0, players: [], source_kind: null, watch: true};
}
function current(key: string): RecordedPosition {
  return {description: null, note_key: key, ...positions.find(position => position.key === key)!, observation: makeObservation(2, key === 'latest'), decision: null, outputs: null, viewer: 0, revealed: false, can_reveal: false, history: [], history_labels: [], history_before: null, cursor: 0, transition_start: 0, consequences_recorded: true};
}
function installApi(live = false): string[] {
  const calls: string[] = [];
  vi.stubGlobal('sessionStorage', {getItem: () => null});
  vi.stubGlobal('localStorage', {getItem: () => null, setItem: () => {}});
  vi.stubGlobal('fetch', vi.fn(async (input: string, init: RequestInit) => {
    calls.push(`${init.method ?? 'GET'} ${input}`);
    if (input.includes('/positions/')) return Response.json(current(decodeURIComponent(input.split('/positions/')[1].split('?')[0])));
    return Response.json(index(live));
  }));
  return calls;
}
afterEach(() => vi.unstubAllGlobals());

describe('ReplayStore', () => {
  it('walks choice, consequences and terminal board using only recorded reads', async () => {
    const calls = installApi();
    const store = createReplayStore();
    await store.open('trace-1');
    expect(store.current?.key).toBe('exact-choice');
    await store.next();
    expect(store.current?.kind).toBe('consequences');
    await store.next();
    expect(store.current?.observation.game_over).toBe(true);
    expect(calls.every(call => call.startsWith('GET'))).toBe(true);
  });
  it('restores a nonzero key and rejects an unavailable saved position visibly', async () => {
    installApi();
    const store = createReplayStore();
    await store.open('trace-1', 'after:0');
    expect(store.selected).toBe(1);
    await store.open('trace-1', 'removed');
    expect(store.current).toBeNull();
    expect(store.error).toContain('unavailable');
  });
  it('ignores an earlier seek reply that arrives after the later seek', async () => {
    installApi();
    const store = createReplayStore();
    await store.open('trace-1');
    let finish: (response: Response) => void = () => {};
    vi.stubGlobal('fetch', vi.fn((input: string) => input.includes('after:0') || input.includes('after%3A0')
      ? new Promise<Response>(resolve => finish = resolve)
      : Promise.resolve(Response.json(current('latest')))));
    const earlier = store.seek(1);
    await store.seek(2);
    finish(Response.json(current('after:0')));
    await earlier;
    expect(store.current?.key).toBe('latest');
    expect(store.busy).toBe(false);
  });
  it('uses the persisted revision for explicit continuation and never retries a lost reply', async () => {
    const calls = installApi(true);
    const store = createReplayStore();
    await store.open('trace-1', 'latest');
    const bodies: unknown[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: string, init: RequestInit) => {
      calls.push(`${init.method ?? 'GET'} ${input}`);
      if (init.method === 'POST') { bodies.push(JSON.parse(init.body as string)); throw new Error('Lost reply'); }
      return Response.json(index(true));
    }));
    await store.produce();
    expect(bodies).toEqual([{revision: 91}]);
    expect(store.playing).toBe(false);
    expect(store.error).toContain('Refresh');
    expect(calls.filter(call => call.startsWith('POST'))).toHaveLength(1);
  });
  it('keeps physical seating independent of the acting information view', () => {
    const observation = makeObservation(1);
    const oriented = orientBoard(observation, 1, null);
    expect(oriented.agent).toBe(observation.opponent);
    expect(oriented.opponent).toBe(observation.agent);
    expect(observation.agent.player_index).toBe(0);
    expect(oriented.opponent.hand).toEqual(observation.agent.hand);
  });
  it('shows neither seat as the winner of a drawn game', () => {
    const over = {...makeObservation(1), game_over: true, won: false};
    expect(orientBoard(over, 1, null).won).toBe(false);
    expect(orientBoard(over, 1, 1).won).toBe(true);
    expect(orientBoard(over, 1, 0).won).toBe(false);
  });
});

it('loads every beat of a transition spanning several bounded history pages', async () => {
  installApi();
  const store = createReplayStore();
  await store.open('trace-1');
  const beforeValues: (string | null)[] = [];
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const before = new URL(input, 'https://academy.test').searchParams.get('before');
    beforeValues.push(before);
    const end = before === null ? 205 : Number(before);
    const start = Math.max(0, end - 100);
    return Response.json({...current('after:0'), cursor: 205, transition_start: 0,
      history: Array.from({length: end - start}, (_, i) => ({seq: start + i, from_revision: 90, to_revision: 91, caused_by: null, group: 0, importance: 'ambient', suggested_ms: 100, sound: null, kind: {kind: 'turn_started', player: 0, turn_number: 1}})),
      history_labels: Array.from({length: end - start}, () => ({objects: {}, players: {}, stacks: {}})), history_before: start || null,
    });
  }));
  await store.seek(1);
  expect(store.current?.history.map(event => event.seq)).toEqual(Array.from({length: 205}, (_, i) => i));
  expect(beforeValues).toEqual([null, '105', '5']);
});

it('opens at the first permitted choice while retaining earlier opponent consequences', async () => {
  installApi();
  vi.stubGlobal('fetch', vi.fn(async (input: string) => input.includes('/positions/')
    ? Response.json(current('exact-choice'))
    : Response.json({...index(), positions: [positions[1], positions[0], positions[2]]})));
  const store = createReplayStore();
  await store.open('trace-1');
  expect(store.selected).toBe(1);
  expect(store.current?.key).toBe('exact-choice');
  expect(store.index?.positions[0].kind).toBe('consequences');
});

it('pause holds the board when a delayed next-position read finishes', async () => {
  installApi();
  const store = createReplayStore();
  await store.open('trace-1');
  const held = store.current;
  let finish!: (response: Response) => void;
  vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(resolve => finish = resolve)));
  store.playing = true;
  const next = store.next(true);
  store.pause();
  expect(store.busy).toBe(false);
  finish(Response.json(current('after:0')));
  await next;
  expect(store.current).toBe(held);
  expect(store.selected).toBe(0);
  expect(store.playing).toBe(false);
  vi.stubGlobal('fetch', vi.fn(async () => Response.json(current('after:0'))));
  await store.next(true);
  expect(store.current?.key).toBe('after:0');
});

it.each([false, true])('pause holds the reader while a pending producer saves (lost reply: %s)', async lostReply => {
  installApi(true);
  const store = createReplayStore();
  await store.open('trace-1', 'latest');
  const held = store.current;
  let finish!: () => void;
  let commands = 0;
  const saved = {...index(true), revision: 92, positions: [...positions.slice(0, -1), {...positions[0], key: 'new-choice'}, {...positions[1], key: 'after:1'}, positions[2]]};
  vi.stubGlobal('fetch', vi.fn(async (input: string, init: RequestInit) => {
    if (init.method === 'POST') {
      commands++;
      await new Promise<void>(resolve => finish = resolve);
      if (lostReply) throw new Error('Lost reply');
      return Response.json({});
    }
    if (input.includes('/positions/')) return Response.json({...current('exact-choice'), key: 'new-choice'});
    return Response.json(saved);
  }));
  store.playing = true;
  const production = store.produce();
  store.pause();
  finish();
  await production;
  expect(store.index?.revision).toBe(92);
  expect(store.current).toBe(held);
  expect(store.selected).toBe(2);
  expect(store.playing).toBe(false);
  expect(store.producing).toBe(false);
  expect(commands).toBe(1);
  await store.next();
  expect(store.current?.key).toBe('new-choice');
  expect(commands).toBe(1);
});

it('stops playback at a failed frontier and produces only through explicit retry', async () => {
  const calls = installApi(true);
  const store = createReplayStore();
  await store.open('trace-1', 'latest');
  store.index = {...store.index!, mirror_retry: {code: 'policy_failed', message: 'Bot failed', match_id: 'trace-1', expected_revision: 91}};
  store.playing = true;
  await store.next(true);
  expect(store.playing).toBe(false);
  expect(calls.some(call => call.startsWith('POST'))).toBe(false);
  await store.seek(0);
  expect(store.current?.key).toBe('exact-choice');
  await store.seek(2);
  const bodies: unknown[] = [];
  vi.stubGlobal('fetch', vi.fn(async (input: string, init: RequestInit) => {
    if (init.method === 'POST') bodies.push(JSON.parse(init.body as string));
    return Response.json(input.includes('/positions/') ? current('latest') : index(true));
  }));
  await store.produce(true);
  expect(bodies).toEqual([{revision: 91, retry: true}]);
});

it('keeps an uncertain Watch head blocked until an explicit read succeeds', async () => {
  installApi(true);
  const store = createReplayStore();
  await store.open('trace-1', 'latest');
  let posts = 0;
  vi.stubGlobal('fetch', vi.fn(async (_input: string, init: RequestInit) => {
    if (init.method === 'POST') posts++;
    throw new Error('Connection unavailable');
  }));
  await store.produce();
  expect(store.headUncertain).toBe(true);
  await store.produce(true);
  expect(posts).toBe(1);
  vi.stubGlobal('fetch', vi.fn(async () => Response.json(index(true))));
  await store.refreshHead();
  expect(store.headUncertain).toBe(false);
  expect(store.error).toBe('');
});

it('keeps historical reading separate from blocked frontier work and retires late source updates', async () => {
  installApi(true);
  const store = createReplayStore();
  await store.open('trace-1', 'latest');
  vi.useFakeTimers();
  let finishStep!: () => void;
  let finishStatus!: (response: Response) => void;
  const status = {scope: 'match', setup_id: null, loading_seat: null, operation_id: 'frontier', sequence: 2, kind: 'choose', started_at: 1, match_id: 'trace-1', revision: 91, chooser: 1, state: 'running'};
  const fetcher = vi.fn(async (input: string, init: RequestInit) => {
    if (init.method === 'POST') { await new Promise<void>(resolve => finishStep = resolve); return Response.json({}); }
    if (input.endsWith('/work')) return new Promise<Response>(resolve => finishStatus = resolve);
    if (input.includes('/positions/')) return Response.json(current(decodeURIComponent(input.split('/positions/')[1].split('?')[0])));
    return Response.json({...index(true), id: input.includes('trace-2') ? 'trace-2' : 'trace-1'});
  });
  vi.stubGlobal('fetch', fetcher);
  try {
    const production = store.produce();
    await vi.advanceTimersByTimeAsync(1000);
    finishStatus(Response.json({work: status}));
    await vi.advanceTimersByTimeAsync(0);
    expect(store.work?.revision).toBe(91);
    await store.seek(0);
    expect(store.current?.key).toBe('exact-choice');
    expect(store.producing).toBe(true);
    expect(store.work?.match_id).toBe('trace-1');
    await store.next();
    expect(store.current?.key).toBe('after:0');
    await vi.advanceTimersByTimeAsync(1000);
    await store.open('trace-2');
    finishStatus(Response.json({work: {...status, sequence: 99}}));
    await vi.advanceTimersByTimeAsync(2000);
    expect(store.work).toBeNull();
    expect(store.index?.id).toBe('trace-2');
    expect(fetcher.mock.calls.filter(([input]) => input.endsWith('/work'))).toHaveLength(2);
    finishStep();
    await production;
    expect(store.index?.id).toBe('trace-2');
    expect(store.work).toBeNull();
  } finally { store.cancelReads(); vi.useRealTimers(); }
});

it('marks lost frontier status as disconnected without retrying the producer', async () => {
  installApi(true);
  const store = createReplayStore();
  await store.open('trace-1', 'latest');
  vi.useFakeTimers();
  let finish!: () => void;
  let disconnected = true;
  let commands = 0;
  vi.stubGlobal('fetch', vi.fn(async (input: string, init: RequestInit) => {
    if (init.method === 'POST') { commands++; await new Promise<void>(resolve => finish = resolve); return Response.json({}); }
    if (input.endsWith('/work')) {
      if (disconnected) throw new Error('Disconnected');
      return Response.json({work: {scope: 'match', setup_id: null, loading_seat: null, operation_id: 'current', sequence: 2, kind: 'choose', started_at: 1, match_id: 'trace-1', revision: 91, chooser: 1, state: 'running'}});
    }
    return Response.json(index(true));
  }));
  try {
    const production = store.produce();
    await vi.advanceTimersByTimeAsync(1000);
    expect(store.workDisconnected).toBe(true);
    disconnected = false;
    await vi.advanceTimersByTimeAsync(1000);
    expect(store.workDisconnected).toBe(false);
    expect(store.work?.operation_id).toBe('current');
    expect(commands).toBe(1);
    store.cancelReads();
    finish();
    await production;
  } finally { store.cancelReads(); vi.useRealTimers(); }
});
