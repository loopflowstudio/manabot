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
  return {id: 'trace-1', positions: [...positions], total: 3, next_offset: null, revision: 91, status: canProduce ? 'waiting' : 'finished', can_produce: canProduce, can_study: true, winner: 0, players: [], source_kind: null, watch: true, can_feedback: false};
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
    const oriented = orientBoard(observation, 1);
    expect(oriented.agent).toBe(observation.opponent);
    expect(oriented.opponent).toBe(observation.agent);
    expect(observation.agent.player_index).toBe(0);
    expect(oriented.opponent.hand).toEqual(observation.agent.hand);
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
