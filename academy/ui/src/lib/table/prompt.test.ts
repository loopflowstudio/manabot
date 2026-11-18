import { readdirSync, readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type { ExperienceFrame } from '$lib/types';
import { promptLine } from './prompt';
import { tableView } from './view';

const FIXTURES = new URL('../../../../../managym/academy/fixtures/', import.meta.url);

function framesIn(value: unknown, found: ExperienceFrame[] = []): ExperienceFrame[] {
  if (Array.isArray(value)) value.forEach(item => framesIn(item, found));
  else if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    if (Array.isArray(record.offers) && record.projection && 'revision' in record && record.prompt) found.push(record as unknown as ExperienceFrame);
    Object.values(record).forEach(item => framesIn(item, found));
  }
  return found;
}

const frames = readdirSync(FIXTURES).filter(name => name.endsWith('.json'))
  .flatMap(name => framesIn(JSON.parse(readFileSync(new URL(name, FIXTURES), 'utf8'))));

describe('promptLine', () => {
  it.each(frames.map(frame => [frame.action_space, frame] as const))('%s: one short line in plain words', (_, frame) => {
    const line = promptLine(frame, tableView(frame));
    expect(line.length).toBeGreaterThan(5);
    expect(line.length).toBeLessThan(110);
    expect(line).not.toMatch(/offer|commit|clause|nonparticipation/i);
  });

  it('names the cost and how Space pays it', () => {
    const pay = frames.find(frame => frame.action_space === 'PAY_OR_NOT' && frame.projection.decision_context?.mana_cost);
    expect(pay).toBeDefined();
    expect(promptLine(pay!, tableView(pay!))).toMatch(/^Pay \{.*Space pays automatically/);
  });

  it('invites a land or a sorcery-speed spell only in your own main phase', () => {
    const frame = structuredClone(frames[0]);
    frame.action_space = 'PRIORITY';
    frame.projection.stack_objects = [];
    frame.projection.turn.active_player_id = frame.projection.opponent.id;
    expect(promptLine(frame, tableView(frame))).not.toMatch(/Play a land/);
  });
});

describe('stack names in the prompt', () => {
  it('names an ability by its source, not by its rules text', () => {
    const frame = structuredClone(frames[0]);
    frame.action_space = 'PRIORITY';
    frame.projection.stack_objects = [{ id: 9, kind: 'triggered_ability', controller: 0,
      label: 'Fire Nation Cadets — Whenever this attacks (while your graveyard has at least 1 Lesson): Add {R}{R}.' }];
    expect(promptLine(frame, tableView(frame))).toBe("Fire Nation Cadets's trigger is on the stack. Respond, or Space to let it resolve");
  });
});
