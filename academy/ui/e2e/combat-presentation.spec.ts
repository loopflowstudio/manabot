import { tableSnapshot } from './fixtures/table';
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

import combatSequence from '../src/lib/fixtures/curated-combat-to-turn.json' with { type: 'json' };
import boltFixtureJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type {
  Command,
  ExperienceFrame,
  FrameUpdate,
  InteractionOffer,
  Observation,
  PresentationEvent,
  RecoveryEnvelope,
} from '../src/lib/types';


const boltFixture = boltFixtureJson as { recovery: RecoveryEnvelope };

function combatObservation(): Observation {
  const observation = structuredClone(boltFixture.recovery.frame.projection);
  observation.turn = {
    turn_number: 4,
    phase: 'COMBAT',
    step: 'DECLARE_ATTACKERS_STEP',
    active_player_id: 0,
    agent_player_id: 0,
  };
  observation.agent.battlefield = [{
    id: 83,
    name: 'Otter-Penguin',
    controller_id: 0,
    tapped: false,
    damage: 0,
    summoning_sick: false,
    power: 2,
    toughness: 2,
    base_power: 2,
    base_toughness: 2,
    plus1_counters: 0,
  }];
  observation.opponent.battlefield = [{
    id: 84,
    name: 'Badgermole Cub',
    controller_id: 1,
    tapped: false,
    damage: 0,
    summoning_sick: false,
    power: 1,
    toughness: 1,
    base_power: 1,
    base_toughness: 1,
    plus1_counters: 0,
  }];
  return observation;
}

function subject(id: number, label: string): NonNullable<InteractionOffer['subjects']>[number]['object'] {
  return { entity_id: id, subject: { kind: 'object', id: { entity: id, incarnation: 0 } }, display_id: id, label };
}

/** A combat offer as the authority builds it: the attack names its creature as source, a block names the attacker as target. */
function offer(
  id: number,
  verb: 'declare_attackers' | 'declare_blockers' | 'pass_priority',
  label: string,
  actionType: string,
  focus: number[],
  subjects: InteractionOffer['subjects'] = undefined,
): InteractionOffer {
  return { id, actor: 0, verb, source: null, label, help: null, choices: [], confirm_label: label, action_type: actionType, focus, ...(subjects ? { subjects } : {}) };
}

const OTTER = subject(83, 'Otter-Penguin');
const CUB = subject(84, 'Badgermole Cub');

function frameAt(stage: 'attack' | 'block' | 'priority'): ExperienceFrame {
  const base = structuredClone(boltFixture.recovery.frame);
  const revision = stage === 'attack' ? 10 : stage === 'block' ? 11 : 12;
  const observation = combatObservation();
  const actionSpace = stage === 'attack' ? 'DECLARE_ATTACKER' : stage === 'block' ? 'DECLARE_BLOCKER' : 'PRIORITY';
  if (stage === 'block') {
    observation.turn = { ...observation.turn, active_player_id: 1, step: 'DECLARE_BLOCKERS_STEP' };
    observation.opponent.battlefield[0].attacking = true;
  }
  if (stage === 'priority') observation.turn = { ...observation.turn, active_player_id: 1, phase: 'COMBAT', step: 'COMBAT_END' };
  const offers = stage === 'attack'
    ? [
        offer(101, 'declare_attackers', 'Attack with Otter-Penguin', 'DECLARE_ATTACKER', [83], [{ role: 'source', object: OTTER }]),
        offer(102, 'declare_attackers', 'Do not attack with Otter-Penguin', 'DECLARE_ATTACKER', [83], [{ role: 'choice', object: OTTER }]),
      ]
    : stage === 'block'
      ? [
          offer(201, 'declare_blockers', 'Block Badgermole Cub with Otter-Penguin', 'DECLARE_BLOCKER', [83, 84],
            [{ role: 'choice', object: OTTER }, { role: 'target', object: CUB }]),
          offer(202, 'declare_blockers', 'Otter-Penguin: do not block', 'DECLARE_BLOCKER', [83], [{ role: 'choice', object: OTTER }]),
        ]
      : [offer(301, 'pass_priority', 'Pass priority', 'PRIORITY_PASS_PRIORITY', [])];
  return {
    ...base,
    match_id: 'combat-prompt-proof',
    revision,
    frame_hash: `combat-frame-${revision}`,
    prompt: { id: 100 + revision, actor: 0, kind: actionSpace.toLowerCase(), title: 'Choose a combat action', instruction: 'Choose a combat action' },
    projection: observation,
    offers,
    action_space: actionSpace,
    deck_names: { hero: 'UR Lessons', villain: 'GW Allies' },
  };
}

/** What the authority reports after the block: the block, damage both ways, nobody dies. */
function combatEvents(from: number): PresentationEvent[] {
  const event = (seq: number, kind: PresentationEvent['kind']): PresentationEvent => ({
    seq, from_revision: from, to_revision: from + 1, caused_by: null, group: 0, importance: 'normal', suggested_ms: 80, sound: null, kind,
  });
  return [
    event(0, { kind: 'blocked', attacker: { entity: 84, incarnation: 0 }, blockers: [{ entity: 83, incarnation: 0 }] }),
    event(1, { kind: 'damage', source: { kind: 'object', id: { entity: 84, incarnation: 0 } }, target: { kind: 'object', id: { entity: 83, incarnation: 0 } }, amount: 1 }),
    event(2, { kind: 'damage', source: { kind: 'object', id: { entity: 83, incarnation: 0 } }, target: { kind: 'object', id: { entity: 84, incarnation: 0 } }, amount: 2 }),
  ];
}

function acceptedUpdate(command: Command, frame: ExperienceFrame, presentation: PresentationEvent[] = []): FrameUpdate {
  return {
    base_revision: command.expected_revision,
    frame,
    presentation,
    receipt: {
      command_id: command.command_id, actor: 0, accepted_at: command.expected_revision,
      resulting_revision: frame.revision, resulting_frame_hash: frame.frame_hash,
    },
  };
}

/** Attack, then block, then the combat's end; each Command answers the frame it was made at. */
async function installCombatAuthority(page: Page, commands: Command[]): Promise<void> {
  await page.routeWebSocket('**/ws/play', (socket) => {
    socket.onMessage((raw) => {
      const message = JSON.parse(String(raw)) as { type?: string; command?: Command };
      if (message.type === 'new_game') {
        const frame = frameAt('attack');
        socket.send(JSON.stringify({
          type: 'observation', table: tableSnapshot(), data: frame.projection, actions: [],
          session_id: 'combat-session', resume_token: 'combat-resume-token',
          recovery: { ...structuredClone(boltFixture.recovery), reason: 'initial_connect', frame, presentation_tail: [],
            presentation_cursor: 0, accepted_commands: [], replay_cursor: frame.revision },
        }));
        return;
      }
      if (message.type !== 'command' || !message.command) return;
      const command = message.command;
      commands.push(command);
      const blocked = command.expected_revision === 11;
      const next = command.expected_revision === 10 ? frameAt('block') : frameAt('priority');
      if (blocked) next.projection.agent.battlefield[0].damage = 1;
      if (blocked) next.projection.opponent.battlefield[0].damage = 2;
      socket.send(JSON.stringify({ type: 'command_outcome', status: 'accepted',
        update: acceptedUpdate(command, next, blocked ? combatEvents(11) : []) }));
    });
  });
}

async function startCombat(page: Page, commands: Command[]): Promise<void> {
  await installCombatAuthority(page, commands);
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game' }).first().click();
  await expect(page.getByTestId('table')).toHaveAttribute('data-action-space', 'DECLARE_ATTACKER');
}

const card = (page: Page, id: number) => page.locator(`.board [data-testid=table-card][data-object-id="${id}"]`);

for (const mode of ['pointer', 'keyboard'] as const) {
  test(`combat is answered on the board by ${mode}: attack with the creature, block on the attacker`, async ({ page }) => {
    const commands: Command[] = [];
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await startCombat(page, commands);

    // The creature asked about is lifted and is the attack; declining is a separate button.
    await expect(card(page, 83)).toHaveClass(/asked/);
    await expect(card(page, 83)).toHaveClass(/legal/);
    await expect(card(page, 83)).toHaveAccessibleName(/Attack with Otter-Penguin \(1\)$/);
    await expect(page.getByTestId('prompt-action')).toHaveText(['2Do not attack with Otter-Penguin']);
    await expect(page.getByTestId('action-option')).toHaveCount(2);
    if (mode === 'pointer') await card(page, 83).click();
    else await page.keyboard.press('1');
    await expect.poll(() => commands.length).toBe(1);
    expect(commands[0]).toMatchObject({ match_id: 'combat-prompt-proof', expected_revision: 10, prompt_id: 110, offer_id: 101, answers: [] });

    // Blocking: the attacker is the control; the blocker asked about is lifted; not blocking is a button.
    await expect(page.getByTestId('table')).toHaveAttribute('data-action-space', 'DECLARE_BLOCKER');
    await expect(card(page, 83)).toHaveClass(/asked/);
    await expect(card(page, 84)).toHaveClass(/legal/);
    await expect(card(page, 84)).toHaveAccessibleName(/Block Badgermole Cub with Otter-Penguin \(1\)$/);
    await expect(page.getByTestId('prompt-action')).toHaveText(['2Otter-Penguin: do not block']);
    if (mode === 'pointer') await card(page, 84).click();
    else await page.keyboard.press('1');
    await expect.poll(() => commands.length).toBe(2);
    expect(commands[1]).toMatchObject({ expected_revision: 11, prompt_id: 111, offer_id: 201, answers: [] });
    expect(errors).toEqual([]);
  });
}

test('after combat, the block stays drawn and damage shows on the cards it landed on', async ({ page }, testInfo) => {
  const commands: Command[] = [];
  await startCombat(page, commands);
  await card(page, 83).click();
  await expect(page.getByTestId('table')).toHaveAttribute('data-action-space', 'DECLARE_BLOCKER');
  await card(page, 84).click();
  await expect(page.getByTestId('table')).toHaveAttribute('data-action-space', 'PRIORITY');
  // No frame holds the block any longer; the block events since the move draw it.
  await expect(page.locator('[data-testid=table-lines] [data-line-kind=block]')).toHaveCount(1);
  await expect(card(page, 84).getByTestId('damage-dealt')).toHaveText('−2');
  await expect(card(page, 83).getByTestId('damage-dealt')).toHaveText('−1');
  await expect(card(page, 84)).toHaveAccessibleName(/dealt 2 since your last move/);
  await expect(page.getByTestId('recent-fact')).toHaveText([
    'Otter-Penguin blocked Badgermole Cub',
    'Badgermole Cub dealt 1 to Otter-Penguin',
    'Otter-Penguin dealt 2 to Badgermole Cub',
  ]);
  await page.screenshot({ path: testInfo.outputPath('after-combat.png') });
  // Passing is the person's next move: the marks of the last one go.
  await page.keyboard.press('Space');
  await expect.poll(() => commands.length).toBe(3);
});

test('Replay seeks retain exactly the recorded prefix and label missing old consequences', async ({ page }) => {
  const observation = combatObservation();
  const trace = {
    id: 'recorded-history',
    config: { hero_deck: {}, villain_deck: {}, villain_type: 'passive', seed: 42 },
    events: [0, 1, 2].map(index => ({
      actor: 'hero', observation, actions: [], action: 0, action_description: `Recorded choice ${index}`,
      reward: 0, presentation: combatSequence.events.slice(index * 2, index * 2 + 2),
    })),
    final_observation: observation, winner: null, end_reason: 'fixture', timestamp: '2026-11-14T00:00:00Z',
  };
  const positions = Array.from({length: 7}, (_, index) => ({
    key: index === 6 ? 'latest' : index % 2 === 0 ? `legacy-choice:${index / 2}` : `after:${(index - 1) / 2}`,
    kind: index === 6 ? 'state' : index % 2 === 0 ? 'choice' : 'consequences',
    turn: 1, phase: 'COMBAT', step: 'DAMAGE', actor: null, automatic: false, can_reveal: false,
  }));
  await page.route('**/api/traces', route => route.fulfill({ json: [] }));
  await page.route('**/api/traces/recorded-history/recording', route => route.fulfill({json: {
    id: trace.id, positions, total: positions.length, next_offset: null, revision: 11,
    status: 'interrupted', can_produce: false, winner: null, players: [], source_kind: null, watch: false, can_feedback: false,
  }}));
  await page.route('**/api/traces/recorded-history/positions/**', route => {
    const key = decodeURIComponent(new URL(route.request().url()).pathname.split('/positions/')[1]);
    const index = positions.findIndex(position => position.key === key);
    const events = trace.events.slice(0, Math.ceil(index / 2)).flatMap(event => event.presentation);
    return route.fulfill({json: {...positions[index], observation, decision: null, outputs: null,
      description: null, note_key: key, viewer: 0, revealed: false, history: events,
      history_labels: events.map(() => combatSequence.labels), history_before: null,
      cursor: events.length, transition_start: Math.max(0, events.length - 2), consequences_recorded: events.length > 0,
    }});
  });
  await page.goto('/replay?trace=recorded-history');
  const history = page.getByTestId('presentation-history');
  await history.locator('summary').click();
  await expect(history).toContainText('Consequences not recorded');
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(history.getByRole('listitem')).toHaveCount(2);
  await page.getByRole('slider', { name: 'Replay position' }).fill('3');
  await expect(history.getByRole('listitem')).toHaveCount(4);
  await page.getByRole('button', { name: 'Previous', exact: true }).click();
  await expect(history.getByRole('listitem')).toHaveCount(2);
  await page.getByRole('slider', { name: 'Replay position' }).fill('5');
  await expect(history.getByRole('listitem')).toHaveCount(6);
  await expect(history.locator('[data-presentation-seq="1"]')).toHaveCount(1);
  await expect(history).toContainText('Otter-Penguin, Badgermole Cub die.');
  await page.getByRole('slider', { name: 'Replay position' }).fill('0');
  await expect(history.getByRole('listitem')).toHaveCount(0);
  await expect(history).toContainText('Consequences not recorded');
  // Older recordings remain readable without inventing causal events.
  for (const event of trace.events) event.presentation = [];
  await page.reload();
  await page.getByRole('slider', { name: 'Replay position' }).fill('3');
  await history.locator('summary').click();
  await expect(history).toContainText('Consequences not recorded');
});



test('history seek removes inspection and repositions to a preceding retained event', async ({ page }) => {
  const observation = combatObservation();
  // Two transitions of a legacy record: ten facts, then ninety-five more.
  const events: PresentationEvent[] = Array.from({ length: 105 }, (_, seq) => ({
    ...structuredClone(combatSequence.events[0]) as PresentationEvent, seq,
    from_revision: seq < 10 ? 1 : 2, to_revision: seq < 10 ? 2 : 3,
  }));
  const trace = {
    config: { hero_deck: {}, villain_deck: {}, villain_type: 'passive' },
    events: [events.slice(0, 10), events.slice(10)].map((presentation, index) => ({
      actor: 'hero', observation, actions: [], action: 0, action_description: `Choice ${index}`,
      reward: 0, presentation,
    })),
    final_observation: observation, winner: null, end_reason: 'fixture', timestamp: '2026-11-14T00:00:00Z',
  };
  // The real legacy reader serves the record from the release server's store.
  await writeFile(fileURLToPath(new URL('../test-results/release/traces/history-anchor-seek.json', import.meta.url)), JSON.stringify(trace));
  await page.goto('/replay?trace=history-anchor-seek&position=latest');
  const slider = page.getByRole('slider', { name: 'Replay position' });
  // Positions: choice 0, its consequences, choice 1, its consequences, latest state.
  await expect(slider).toHaveAttribute('max', '4');
  const panel = page.getByTestId('presentation-history');
  await panel.locator('summary').click();
  const region = panel.getByRole('list', { name: 'Recorded game events' });
  const subject = panel.locator('[data-presentation-seq="50"]').getByRole('button').first();
  await subject.focus();
  await subject.press('Enter');
  const inspection = panel.getByRole('region', { name: 'History card inspection' });
  await expect(inspection).toBeFocused();
  await region.evaluate(element => {
    const row = element.querySelector<HTMLElement>('[data-presentation-seq="50"]')!;
    element.scrollTop += row.getBoundingClientRect().top - element.getBoundingClientRect().top;
    element.dispatchEvent(new Event('scroll'));
  });
  await expect.poll(() => region.evaluate(element => Array.from(element.querySelectorAll<HTMLElement>('[data-presentation-seq]'))
    .find(row => row.getBoundingClientRect().bottom > element.getBoundingClientRect().top)?.dataset.presentationSeq)).toBe('50');
  // Select a recorded prefix without moving focus away from the removed inspection.
  await slider.evaluate((element: HTMLInputElement) => {
    element.value = '1';
    element.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await expect(inspection).toHaveCount(0);
  await expect(region).toBeFocused();
  await expect(region.locator('[data-presentation-seq="9"]')).toBeVisible();
  await expect(region.locator('[data-presentation-seq="10"]')).toHaveCount(0);
  await slider.fill('0');
  await expect(panel.locator('[data-presentation-seq]')).toHaveCount(0);
  await slider.fill('1');
  await expect(region.locator('[data-presentation-seq="0"]')).toHaveCount(1);
  await region.getByRole('button').first().click();
  await expect(inspection).toBeFocused();
  await slider.evaluate((element: HTMLInputElement) => {
    element.value = '0';
    element.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await expect(inspection).toHaveCount(0);
  await expect(panel.locator('summary')).toBeFocused();
});

test('Replay reader actions hold playback and reject a delayed position without pausing hover or restored scroll', async ({ page }) => {
  const observation = combatObservation();
  const events: PresentationEvent[] = Array.from({length: 150}, (_, seq) => ({
    ...structuredClone(combatSequence.events[0]) as PresentationEvent, seq,
  }));
  const positions = ['choice', 'consequences', 'state'].map((kind, i) => ({
    key: `held:${i}`, kind, turn: 4, phase: 'COMBAT', step: 'DAMAGE', actor: null, automatic: false, can_reveal: false,
  }));
  await page.route('**/api/traces', route => route.fulfill({json: []}));
  await page.route('**/api/traces/reader/recording', route => route.fulfill({json: {
    id: 'reader', positions, total: 3, next_offset: null, revision: 11,
    status: 'finished', can_produce: false, winner: null, players: [], source_kind: null, watch: false, can_feedback: false,
  }}));
  let delay = false;
  let release: (() => void) | null = null;
  await page.route('**/api/traces/reader/positions/**', async route => {
    const url = new URL(route.request().url());
    const key = decodeURIComponent(url.pathname.split('/positions/')[1]);
    if (delay && key === 'held:1') await new Promise<void>(resolve => release = resolve);
    const position = positions.find(position => position.key === key)!;
    await route.fulfill({json: {...position, observation, decision: null, outputs: null,
      description: null, note_key: key, viewer: 0, revealed: false, history: url.searchParams.has('before') ? [] : events,
      history_labels: url.searchParams.has('before') ? [] : events.map(() => combatSequence.labels), history_before: url.searchParams.has('before') ? null : 1,
      cursor: 150, transition_start: 150, consequences_recorded: true,
    }});
  });
  await page.setViewportSize({width: 1280, height: 800});
  await page.goto('/replay?trace=reader');
  const counter = page.getByTestId('position-counter');
  await expect(counter).toHaveText('Position 1 / 3');
  await page.clock.install({time: 0});
  await page.clock.pauseAt(1000);
  await page.getByRole('combobox', {name: 'Speed', exact: true}).selectOption('16');
  const play = page.getByRole('button', {name: 'Play', exact: true});
  const pause = page.getByRole('button', {name: 'Pause', exact: true});
  const card = page.getByTestId('game-board').locator('button[data-object-id]').first();
  const inspection = page.getByRole('region', {name: 'Inspected card', exact: true});
  for (const input of ['pointer', 'keyboard']) {
    await play.click();
    await card.hover();
    await expect(pause).toBeVisible();
    if (input === 'pointer') await card.click();
    else { await card.focus(); await page.keyboard.press('Enter'); }
    await expect(inspection).toBeVisible();
    await expect(play).toBeVisible();
    await page.clock.runFor(500);
    await expect(counter).toHaveText('Position 1 / 3');
    await page.getByRole('button', {name: 'Swap seats'}).click();
    await expect(inspection).toBeVisible();
    await page.getByRole('button', {name: 'Close inspection', exact: true}).click();
    await expect(play).toBeVisible();
  }
  // The delayed read is already in flight when inspection claims the board.
  delay = true;
  await play.click();
  await page.clock.runFor(100);
  await expect.poll(() => release !== null).toBe(true);
  await card.click();
  release!();
  await expect(page.getByText('Reading saved recording…', {exact: true})).toHaveCount(0);
  await page.clock.runFor(500);
  await expect(inspection).toBeVisible();
  await expect(counter).toHaveText('Position 1 / 3');
  await page.getByRole('button', {name: 'Close inspection', exact: true}).click();
  const history = page.getByTestId('presentation-history');
  await play.click();
  await history.locator('summary').click();
  await expect(play).toBeVisible();
  const region = history.getByRole('list', {name: 'Recorded game events'});
  await play.click();
  // Programmatic anchor restoration is not reader intent.
  await region.evaluate(element => { element.scrollTop = 100; });
  await expect(pause).toBeVisible();
  await region.focus();
  await page.keyboard.press('Tab');
  await expect(pause).toBeVisible();
  await region.focus();
  // Tab can reveal the first row. Page Up must have somewhere to scroll.
  await region.evaluate(element => { element.scrollTop = 100; });
  await expect.poll(() => region.evaluate(element => element.scrollTop)).toBe(100);
  await page.keyboard.press('PageUp');
  await page.clock.runFor(50);
  await expect(play).toBeVisible();
  // Page Up scrolls smoothly in real time; measure the anchor once it settles.
  let anchor = -1;
  await expect.poll(async () => {
    const previous = anchor;
    anchor = await region.evaluate(element => element.scrollTop);
    return anchor === previous;
  }).toBe(true);
  await page.clock.runFor(500);
  expect(await region.evaluate(element => element.scrollTop)).toBe(anchor);
  await history.getByRole('button', {name: 'Latest', exact: false}).click();
  await expect(play).toBeVisible();
  await play.click();
  await region.hover();
  await page.mouse.wheel(0, -200);
  await page.clock.runFor(50);
  await expect(play).toBeVisible();
  await history.getByRole('button', {name: 'Latest', exact: false}).click();
  await play.click();
  await history.getByRole('button', {name: 'Earlier events', exact: true}).click();
  await expect(play).toBeVisible();
  await play.click();
  await history.getByRole('button', {name: /^Inspect /}).first().click();
  await expect(play).toBeVisible();
  await page.clock.runFor(500);
  await expect(history.getByRole('region', {name: 'History card inspection'})).toBeVisible();
  await history.getByRole('button', {name: 'Close history inspection'}).click();
  await expect(play).toBeVisible();
  await play.click();
  await page.getByRole('button', {name: 'Load earlier recorded history'}).click();
  await expect(play).toBeVisible();
  await expect(counter).toHaveText('Position 1 / 3');
  delay = false;
  await play.click();
  await page.clock.runFor(100);
  await expect(counter).toHaveText('Position 2 / 3');
});

test('Replay holds inspection while a live producer saves and resumes at its saved choice', async ({ page }) => {
  const observation = combatObservation();
  const latest = {key: 'latest', kind: 'state', turn: 4, phase: 'COMBAT', step: 'DAMAGE', actor: null, automatic: false, can_reveal: false};
  const choice = {...latest, key: 'saved-choice', kind: 'choice'};
  const consequences = {...latest, key: 'saved-consequences', kind: 'consequences'};
  let produced = 0;
  let committed = false;
  let release: (() => void) | null = null;
  await page.route('**/api/traces', route => route.fulfill({json: []}));
  await page.route('**/api/watch/pending/step', async route => {
    produced++;
    await new Promise<void>(resolve => release = resolve);
    committed = true;
    await route.fulfill({json: {}});
  });
  await page.route('**/api/traces/pending/recording', route => route.fulfill({json: {
    id: 'pending', positions: committed ? [choice, consequences, latest] : [latest],
    total: committed ? 3 : 1, next_offset: null, revision: committed ? 12 : 11,
    status: 'waiting', can_produce: true, winner: null, players: [], source_kind: null, watch: true, can_feedback: false,
  }}));
  await page.route('**/api/traces/pending/positions/**', route => {
    const key = decodeURIComponent(new URL(route.request().url()).pathname.split('/positions/')[1]);
    const position = [choice, consequences, latest].find(position => position.key === key)!;
    return route.fulfill({json: {...position, observation, decision: null, outputs: null,
      description: position.kind === 'choice' ? 'Saved producer choice' : null, note_key: key, viewer: 0, revealed: false, history: [], history_labels: [], history_before: null,
      cursor: 0, transition_start: 0, consequences_recorded: true,
    }});
  });
  await page.goto('/replay?trace=pending');
  await expect(page.getByText('Waiting at the live edge', {exact: true})).toBeVisible();
  await page.clock.install({time: 0});
  await page.clock.pauseAt(1000);
  await page.getByRole('combobox', {name: 'Speed', exact: true}).selectOption('16');
  await page.getByRole('button', {name: 'Play', exact: true}).click();
  await page.clock.runFor(100);
  await expect.poll(() => release !== null).toBe(true);
  await expect(page.getByRole('button', {name: 'Pause', exact: true})).toBeEnabled();
  await page.getByTestId('game-board').locator('button[data-object-id]').first().click();
  release!();
  const counter = page.getByTestId('position-counter');
  await expect(counter).toHaveText('Position 1 / 3');
  await expect(page.getByRole('region', {name: 'Inspected card', exact: true})).toBeVisible();
  await expect(page.getByText('Live Watch · reading saved recording', {exact: true})).toBeVisible();
  await expect.poll(() => new URL(page.url()).searchParams.get('position')).toBe('saved-choice');
  await expect(page.getByText('Saved producer choice', {exact: true})).toHaveCount(0);
  await page.clock.runFor(500);
  expect(produced).toBe(1);
  await expect(counter).toHaveText('Position 1 / 3');
  await page.getByRole('button', {name: 'Close inspection', exact: true}).click();
  await page.getByRole('button', {name: 'Next', exact: true}).click();
  await expect(counter).toHaveText('Position 1 / 3');
  await expect(page.getByText('Saved producer choice', {exact: true})).toBeVisible();
  await expect(page.getByTestId('study-score').locator('[aria-current="true"]')).toContainText('Choice');
  await page.reload();
  await expect(counter).toHaveText('Position 1 / 3');
  expect(produced).toBe(1);
});
