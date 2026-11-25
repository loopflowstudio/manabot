import { readdirSync, readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type { ExperienceFrame } from '$lib/types';
import { committed, primaryLabel, promptHint, promptLine } from './prompt';
import { boardView } from './view';

const PLAYERS = { you: 'You', opponent: 'Heuristic' };

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
    const line = promptLine(frame, boardView(frame, PLAYERS));
    expect(line.length).toBeGreaterThan(5);
    expect(line.length).toBeLessThan(110);
    expect(line).not.toMatch(/offer|commit|clause|nonparticipation/i);
  });

  it('asks for the cost in the line and says how Space pays it in the hint', () => {
    const pay = frames.find(frame => frame.action_space === 'PAY_OR_NOT' && frame.projection.decision_context?.mana_cost);
    expect(pay).toBeDefined();
    const view = boardView(pay!, PLAYERS);
    expect(promptLine(pay!, view)).toMatch(/^Pay \{[^.]*$/);
    expect(promptHint(pay!, view)).toMatch(/^Space pays automatically/);
  });

  it('lists what is committed to the cost, in order, apart from the line', () => {
    const pay = frames.find(frame => frame.action_space === 'PAY_OR_NOT' && frame.projection.decision_context?.mana_cost)!;
    const paying = { ...pay, projection: { ...pay.projection, decision_context: { ...pay.projection.decision_context!,
      paid: [{ phrase: 'Gray Ogre, sacrificed when the payment completes' }, { phrase: 'tapped Hill Giant to crew' }] } } };
    expect(committed(paying)).toEqual(['Gray Ogre, sacrificed when the payment completes', 'tapped Hill Giant to crew']);
    expect(promptLine(paying, boardView(paying, PLAYERS))).toBe(promptLine(pay, boardView(pay, PLAYERS)));
    expect(committed(pay)).toEqual([]);
  });

  it('names a tap-pays choice, combat damage and replacement order by what they ask, other modal choices as modes', () => {
    const pay = frames.find(frame => frame.action_space === 'PAY_OR_NOT' && frame.projection.decision_context?.mana_cost)!;
    const modal = (stage: string): ExperienceFrame => ({ ...pay, action_space: 'MODAL', projection: { ...pay.projection,
      decision_context: { ...pay.projection.decision_context!, stage, source: { ...pay.projection.decision_context!.source!, label: 'Appeal to Eirdu' } } } });
    const line = (frame: ExperienceFrame): string => promptLine(frame, boardView(frame, PLAYERS));
    expect(line(modal('payment'))).toBe('Choose what the tap pays for Appeal to Eirdu');
    expect(line(modal('mode'))).toBe('Choose a mode for Appeal to Eirdu');
    expect(line(modal('assign_combat_damage'))).toBe('Assign combat damage');
    expect(line(modal('order_replacement'))).toBe('Choose which replacement applies next');
  });

  it('invites a land or a sorcery-speed spell only in your own main phase', () => {
    const frame = structuredClone(frames[0]);
    frame.action_space = 'PRIORITY';
    frame.projection.stack_objects = [];
    frame.projection.turn.active_player_id = frame.projection.opponent.id;
    expect(promptLine(frame, boardView(frame, PLAYERS))).not.toMatch(/Play a land/);
  });
});

describe('stack names in the prompt', () => {
  it('names an ability by its source, not by its rules text', () => {
    const frame = structuredClone(frames[0]);
    frame.action_space = 'PRIORITY';
    frame.projection.stack_objects = [{ id: 9, kind: 'triggered_ability', controller: 0,
      label: 'Fire Nation Cadets — Whenever this attacks (while your graveyard has at least 1 Lesson): Add {R}{R}.' }];
    expect(promptLine(frame, boardView(frame, PLAYERS))).toBe("Fire Nation Cadets' trigger is on the stack. Respond, or Space to let it resolve");
  });
});

describe('primaryLabel', () => {
  const PROTOCOL = new URL('../../../../../protocol/fixtures/', import.meta.url);
  const priority = readdirSync(PROTOCOL).filter(name => name.endsWith('.json'))
    .flatMap(name => framesIn(JSON.parse(readFileSync(new URL(name, PROTOCOL), 'utf8'))))
    .filter(frame => frame.offers.some(offer => offer.verb === 'pass_priority'));

  it('has priority frames to check', () => expect(priority.length).toBeGreaterThan(0));

  it.each(priority.map(frame => [frame.projection.turn.step, frame] as const))('%s: says what passing does next', (_, frame) => {
    const view = boardView(frame, PLAYERS);
    const label = primaryLabel(frame, view)!;
    if (view.stack.length) expect(label.text).toMatch(/^Resolve\b/);
    else expect(label.text).toMatch(/^To /);
    expect(label.description).toContain('Pass priority');
  });

  function emptyStack(): ExperienceFrame {
    const frame = structuredClone(priority[0]);
    frame.projection.stack_objects = [];
    frame.projection.agent.stack = [];
    frame.projection.opponent.stack = [];
    return frame;
  }

  it('names the next damage step while the stack is empty', () => {
    const frame = emptyStack();
    frame.projection.turn.step = 'COMBAT_FIRST_STRIKE_DAMAGE';
    expect(primaryLabel(frame, boardView(frame, PLAYERS))?.text).toBe('To regular damage');
    frame.projection.turn.step = 'COMBAT_DAMAGE';
    expect(primaryLabel(frame, boardView(frame, PLAYERS))?.text).toBe('To end of combat');
  });

  it('names the stack top while something waits to resolve', () => {
    const frame = emptyStack();
    frame.projection.stack_objects = [{ id: 9001, kind: 'spell', controller: 0, label: 'Lightning Bolt' }];
    expect(primaryLabel(frame, boardView(frame, PLAYERS))?.text).toBe('Resolve Lightning Bolt');
  });

  it('never cuts a long name: the button says the verb and the prompt line names it', () => {
    const frame = emptyStack();
    frame.projection.stack_objects = [{ id: 9001, kind: 'triggered_ability', controller: 0, label: 'Invasion Reinforcements — create a token' }];
    const view = boardView(frame, PLAYERS);
    const label = primaryLabel(frame, view)!;
    expect(label.text).toBe('Resolve');
    expect(label.description).toContain("Invasion Reinforcements' trigger");
    expect(promptLine(frame, view)).toContain("Invasion Reinforcements' trigger is on the stack");
  });

  it('skips blockers when nothing attacks', () => {
    const frame = emptyStack();
    frame.projection.turn.step = 'COMBAT_DECLARE_ATTACKERS';
    [frame.projection.agent, frame.projection.opponent].forEach(player => player.battlefield.forEach(permanent => { permanent.attacking = false; }));
    expect(primaryLabel(frame, boardView(frame, PLAYERS))?.text).toBe('To end of combat');
  });

  it('shows the cost a payment pays', () => {
    const pay = frames.find(frame => frame.offers.some(offer => offer.verb === 'pay') && frame.projection.decision_context?.mana_cost)!;
    expect(primaryLabel(pay, boardView(pay, PLAYERS))?.text).toBe(`Auto pay ${pay.projection.decision_context!.mana_cost}`);
  });

  it('says the cost on the button and the lands it taps in its accessible name', () => {
    const pay = frames.find(frame => frame.offers.some(offer => offer.verb === 'pay') && frame.projection.decision_context?.mana_cost)!;
    const lands = pay.projection.agent.battlefield.filter(permanent => permanent.types?.is_land).slice(0, 2);
    const preview = {
      state: 'exact' as const, explanation: '', cost: pay.projection.decision_context!.mana_cost!,
      sources: lands.map(land => ({ source: { kind: 'object' as const, id: { entity: land.id, incarnation: 0 } }, label: 'x', output: { amounts: [0, 1, 0, 0, 0, 0], classes: [] } })),
      pool_after_production: null, pool_after_spending: null,
    };
    const cost = pay.projection.decision_context!.mana_cost;
    const names = lands.map(land => land.name);
    const listed = names[0] === names[1] ? `${names[0]} ×2` : names.join(', ');
    const label = primaryLabel(pay, boardView(pay, PLAYERS), preview);
    expect(label?.text).toBe(`Auto pay ${cost}`);
    expect(label?.description).toBe(`${pay.offers.find(offer => offer.verb === 'pay')!.label}: ${cost} · ${listed}`);
    // A preview that is not exact draws nothing; the button keeps the cost.
    expect(primaryLabel(pay, boardView(pay, PLAYERS), { ...preview, state: 'needs_choice' })?.text).toBe(`Auto pay ${cost}`);
  });
});
