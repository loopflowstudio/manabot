import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import WorkStatus from './WorkStatus.svelte';
import type { WorkStatus as Status } from '../types';

const work: Status = { operation_id: 'op-1', sequence: 1, kind: 'choose', started_at: 1,
  match_id: 'match-1', revision: 4, chooser: 1, state: 'running' };

describe('runner work status', () => {
  it('names the authoritative operation and chooser without guessed progress', () => {
    const { body } = render(WorkStatus, {props: {work}});
    expect(body).toContain('Choosing bot action · Seat 2');
    expect(body).toContain('role="status"');
    expect(body).not.toContain('%');
  });
  it('suppresses remote progress when disconnected or work has finished', () => {
    expect(render(WorkStatus, {props: {work, connected: false}}).body).not.toContain('Choosing');
    expect(render(WorkStatus, {props: {work: {...work, state: 'failed'}}}).body).not.toContain('Choosing');
  });
});
