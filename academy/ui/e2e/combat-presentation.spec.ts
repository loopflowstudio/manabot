import { tableSnapshot } from './fixtures/table';
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { expect, test, type Browser, type Page } from '@playwright/test';

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

import { finishBoardNarration } from './presentation-readiness';

type InputMode = 'keyboard' | 'pointer';

interface CombatPromptReceipt {
  attackerMode: InputMode;
  blockerMode: InputMode;
  commands: Command[];
}

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

function offer(
  id: number,
  verb: 'declare_attackers' | 'declare_blockers' | 'pass_priority',
  label: string,
  actionType: string,
  focus: number[],
): InteractionOffer {
  return {
    id,
    actor: 0,
    verb,
    source: null,
    label,
    help: null,
    choices: [],
    confirm_label: label,
    action_type: actionType,
    focus,
  };
}

function frameAt(stage: 'attack' | 'block' | 'priority'): ExperienceFrame {
  const base = structuredClone(boltFixture.recovery.frame);
  const revision = stage === 'attack' ? 10 : stage === 'block' ? 11 : 12;
  const promptId = stage === 'attack' ? 110 : stage === 'block' ? 111 : 112;
  const observation = combatObservation();
  const actionSpace = stage === 'attack'
    ? 'DECLARE_ATTACKER'
    : stage === 'block'
      ? 'DECLARE_BLOCKER'
      : 'PRIORITY';
  if (stage === 'block') {
    observation.turn.step = 'DECLARE_BLOCKERS_STEP';
  }
  const offers = stage === 'attack'
    ? [
        offer(101, 'declare_attackers', 'Attack with Otter-Penguin', 'DECLARE_ATTACKER', [83]),
        offer(102, 'declare_attackers', 'Do not attack with Otter-Penguin', 'DECLARE_ATTACKER', [83]),
      ]
    : stage === 'block'
      ? [
          offer(201, 'declare_blockers', 'Block Badgermole Cub with Otter-Penguin', 'DECLARE_BLOCKER', [84, 83]),
          offer(202, 'declare_blockers', 'Otter-Penguin: do not block', 'DECLARE_BLOCKER', [83]),
        ]
      : [offer(301, 'pass_priority', 'Pass priority', 'PRIORITY_PASS_PRIORITY', [])];

  return {
    ...base,
    match_id: 'combat-prompt-proof',
    revision,
    frame_hash: `combat-frame-${revision}`,
    prompt: {
      id: promptId,
      actor: 0,
      kind: actionSpace.toLowerCase(),
      title: 'Choose a combat action',
      instruction: 'Choose a combat action',
    },
    projection: observation,
    offers,
    action_space: actionSpace,
    deck_names: { hero: 'UR Lessons', villain: 'GW Allies' },
  };
}

function acceptedUpdate(command: Command, frame: ExperienceFrame): FrameUpdate {
  return {
    base_revision: command.expected_revision,
    frame,
    presentation: [],
    receipt: {
      command_id: command.command_id,
      actor: 0,
      accepted_at: command.expected_revision,
      resulting_revision: frame.revision,
      resulting_frame_hash: frame.frame_hash,
    },
  };
}

async function installCombatAuthority(
  page: Page,
  commands: Command[],
  events: PresentationEvent[] = [],
  terminal = false,
): Promise<void> {
  // This authority is entirely mocked, including its read-only preview API.
  await page.route('**/api/sessions/combat-session/payment-preview', async route => {
    await route.fulfill({ json: {
      ...route.request().postDataJSON(), viewer: 0,
      preview: { state: 'not_applicable', scope: null,
        continuation: { ability_output: false, outer_payment: false },
        explanation: 'Combat has no mana payment.', cost: null, sources: [],
        pool_after_production: null, pool_after_spending: null },
    } });
  });
  let starts = 0;
  await page.routeWebSocket('**/ws/play', (socket) => {
    socket.onMessage((raw) => {
      const message = JSON.parse(String(raw)) as {
        type?: string;
        command?: Command;
      };
      if (message.type === 'new_game') {
        const frame = frameAt('attack');
        if (starts++ > 0) frame.match_id += `-${starts}`;
        if (terminal) {
          frame.projection.game_over = true;
          frame.winner = 1;
          frame.prompt = null;
          frame.offers = [];
        }
        socket.send(JSON.stringify({
          type: 'observation', table: tableSnapshot(),
          data: frame.projection,
          actions: [],
          session_id: 'combat-session',
          resume_token: 'combat-resume-token',
          recovery: {
            ...structuredClone(boltFixture.recovery),
            reason: 'initial_connect',
            frame,
            presentation_tail: events,
            presentation_cursor: events[0]?.seq ?? 0,
            accepted_commands: [],
            replay_cursor: frame.revision,
          },
        }));
        return;
      }
      if (message.type !== 'command' || !message.command) {
        return;
      }

      const command = message.command;
      commands.push(command);
      const next = command.expected_revision === 10 ? frameAt('block') : frameAt('priority');
      socket.send(JSON.stringify({
        type: 'command_outcome',
        status: 'accepted',
        update: acceptedUpdate(command, next),
      }));
    });
  });
}

async function updateSequence(page: Page): Promise<number> {
  return Number(await page.locator('main').getAttribute('data-update-seq'));
}

async function assertPrompt(
  page: Page,
  family: 'DECLARE_ATTACKER' | 'DECLARE_BLOCKER',
  expectedLabels: string[],
): Promise<void> {
  const panel = page.getByTestId('action-panel');
  const options = page.getByTestId('action-option');
  await expect(panel).toHaveAttribute('data-action-space-kind', family);
  await expect(page.getByTestId('decision-prompt')).toHaveAttribute('data-kind', family);
  await expect(options).toHaveCount(expectedLabels.length);
  await expect(options.first()).toBeFocused();
  for (let index = 0; index < expectedLabels.length; index += 1) {
    await expect(options.nth(index)).toHaveAccessibleName(expectedLabels[index]);
    await expect(options.nth(index)).toBeEnabled();
  }
  await page.keyboard.press('Tab');
  await expect(options.nth(1)).toBeFocused();
  await expect(options.nth(1)).toHaveCSS('outline-style', 'solid');
  await page.keyboard.press('Shift+Tab');
  await expect(options.first()).toBeFocused();
  await expect(options.first()).toHaveCSS('outline-style', 'solid');
}

async function activate(
  page: Page,
  mode: InputMode,
  expected: { offerId: number; promptId: number; revision: number },
  commands: Command[],
): Promise<void> {
  const before = await updateSequence(page);
  const option = page.getByTestId('action-option').first();
  if (mode === 'keyboard') {
    await expect(option).toBeFocused();
    await expect(option).toHaveCSS('outline-style', 'solid');
    await page.keyboard.press('Enter');
  } else {
    await option.click();
  }
  await expect.poll(() => commands.length).toBeGreaterThan(0);
  expect(commands.at(-1)).toMatchObject({
    match_id: 'combat-prompt-proof',
    expected_revision: expected.revision,
    prompt_id: expected.promptId,
    offer_id: expected.offerId,
    answers: [],
  });
  expect(commands.at(-1)?.command_id).toMatch(/\S+/);
  await expect.poll(() => updateSequence(page)).toBeGreaterThan(before);
}

async function runPromptProof(
  browser: Browser,
  attackerMode: InputMode,
  blockerMode: InputMode,
): Promise<CombatPromptReceipt> {
  const context = await browser.newContext();
  const page = await context.newPage();
  const commands: Command[] = [];
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.text().includes('favicon')) {
      errors.push(message.text());
    }
  });
  await installCombatAuthority(page, commands);
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'New Game' }).first().click();

  await assertPrompt(page, 'DECLARE_ATTACKER', [
    'Attack with Otter-Penguin',
    'Do not attack with Otter-Penguin',
  ]);
  await activate(page, attackerMode, { offerId: 101, promptId: 110, revision: 10 }, commands);

  await assertPrompt(page, 'DECLARE_BLOCKER', [
    'Block Badgermole Cub with Otter-Penguin',
    'Otter-Penguin: do not block',
  ]);
  await activate(page, blockerMode, { offerId: 201, promptId: 111, revision: 11 }, commands);

  expect(errors).toEqual([]);
  await context.close();
  return { attackerMode, blockerMode, commands };
}

test('combat prompts submit only current offers by pointer and keyboard', async ({ browser }) => {
  const pointerThenKeyboard = await runPromptProof(browser, 'pointer', 'keyboard');
  const keyboardThenPointer = await runPromptProof(browser, 'keyboard', 'pointer');

  expect(pointerThenKeyboard.commands.map((command) => command.offer_id)).toEqual([101, 201]);
  expect(keyboardThenPointer.commands.map((command) => command.offer_id)).toEqual([101, 201]);
});

test('a taller decision sidebar does not stretch the board', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1200 });
  await installCombatAuthority(page, [], combatSequence.events as PresentationEvent[]);
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'New Game' }).first().click();
  const board = page.getByTestId('game-board');
  await expect(board).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  const initialHeight = (await board.boundingBox())!.height;
  await page.getByTestId('current-decision').evaluate((status) => {
    const content = document.createElement('div');
    content.style.height = '3000px';
    status.parentElement!.append(content);
  });
  expect((await board.boundingBox())!.height).toBe(initialHeight);
});

for (const reducedMotion of ['no-preference', 'reduce'] as const) {
  test(`consequences survive playback and Finish with ${reducedMotion}`, async ({ page }) => {
    await page.emulateMedia({ reducedMotion });
    await installCombatAuthority(page, [], combatSequence.events as PresentationEvent[]);
    await page.goto('/');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    await page.clock.install({ time: 0 });
    await page.clock.pauseAt(10_000);
    await page.getByRole('button', { name: 'New Game' }).first().click();
    const history = page.getByTestId('presentation-history');
    await page.getByText('History and settings', { exact: true }).click();
    await history.locator('summary').click();
    const stage = page.getByTestId('presentation-stage');
    await expect(history.getByRole('listitem')).toHaveCount(6);
    await expect(history).toContainText('Otter-Penguin deals 2 damage to Badgermole Cub.');
    await expect(page.getByTestId('current-decision')).toHaveText('Your decision');
    await page.getByRole('button', { name: 'Pause narration' }).click();
    await page.clock.runFor(5000);
    await expect(stage).toHaveAttribute('data-presentation-kind', 'damage');
    await page.getByRole('button', { name: 'Resume narration' }).click();
    await page.clock.runFor(10000);
    await expect(stage).toBeHidden();
    await expect(history).toContainText('Otter-Penguin, Badgermole Cub die.');
    await expect(history).toContainText("Hero's turn begins.");

    await page.getByRole('banner').getByRole('link', { name: 'New game', exact: true }).click();
    await page.clock.runFor(32); // Flush the navigation's animation frame while narration time is controlled.
    page.once('dialog', dialog => dialog.accept());
    await page.getByRole('button', { name: 'New Game', exact: true }).click();
    await expect(stage).toBeVisible();
    await page.getByRole('button', { name: 'Skip beat' }).click();
    await page.getByRole('button', { name: 'Fast-forward' }).click();
    await page.getByRole('button', { name: 'Finish', exact: true }).click();
    await expect(stage).toBeHidden();
    await expect(history.getByRole('listitem')).toHaveCount(6);
    await expect(history).toContainText('Otter-Penguin, Badgermole Cub die.');
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
  });
}


test('terminal capture finishes JavaScript narration and retains consequences', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await installCombatAuthority(page, [], combatSequence.events as PresentationEvent[], true);
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.clock.install({ time: 0 });
  await page.clock.pauseAt(10_000);
  await page.getByRole('button', { name: 'New Game' }).first().click();
  const board = page.getByTestId('game-board');
  const stage = page.getByTestId('presentation-stage');
  const resultAction = page.getByTestId('game-result-action');
  await expect(resultAction).toBeFocused();
  await expect(page.getByTestId('action-option')).toHaveCount(0);
  await page.evaluate(() => document.fonts.ready);

  // Semantic narration still advances under the terminal dialog. Finish must
  // settle it without losing the retained consequences.
  await expect(stage).toHaveAttribute('data-presentation-kind', 'damage');
  await page.clock.runFor(1000);
  await expect(stage).toHaveAttribute('data-presentation-kind', 'died');

  await finishBoardNarration(page);
  await expect(stage).toBeHidden();
  await expect(resultAction).toBeFocused();
  const history = page.getByTestId('presentation-history');
  // Inspect the retained disclosures without moving focus out of the terminal dialog.
  await page.locator('details.secondary-panels').evaluate((element: HTMLDetailsElement) => { element.open = true; });
  await history.locator('details').evaluate((element: HTMLDetailsElement) => { element.open = true; });
  await expect(history.getByRole('listitem')).toHaveCount(6);
  await expect(history).toContainText('Otter-Penguin, Badgermole Cub die.');
  const settled = await board.screenshot({ animations: 'disabled' });
  await page.clock.runFor(10000);
  expect((await board.screenshot({ animations: 'disabled' })).equals(settled)).toBe(true);
  await finishBoardNarration(page);
  await expect(resultAction).toBeFocused();
});


for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`history respects reading and decision access ${viewport.width} ${colorScheme}`, async ({ page }, testInfo) => {
      await page.setViewportSize(viewport);
      await page.emulateMedia({ colorScheme });
      const commands: Command[] = [];
      // Fixed semantic pagination fixture, not a simulated workload or game.
      const history: PresentationEvent[] = Array.from({ length: 105 }, (_, seq) => ({
        ...structuredClone(combatSequence.events[seq % 6]) as PresentationEvent,
        seq, from_revision: 8, to_revision: 9, suggested_ms: 80,
      }));
      let append = (): void => { throw new Error('Fixture not connected'); };
      await page.routeWebSocket('**/ws/play', socket => {
        socket.onMessage(raw => {
          const message = JSON.parse(String(raw));
          if (message.type === 'command') { commands.push(message.command); return; }
          if (message.type !== 'new_game') return;
          const frame = frameAt('attack');
          socket.send(JSON.stringify({ type: 'observation', table: tableSnapshot(), data: frame.projection, actions: [],
            session_id: 'history', resume_token: 'history', recovery: {
              ...structuredClone(boltFixture.recovery), frame,
              presentation_tail: history, presentation_cursor: 0,
            },
          }));
          append = () => {
            const next = structuredClone(frame);
            next.revision = 11;
            const presentation = [{ ...structuredClone(history[0]), seq: 105, from_revision: 10, to_revision: 11 }];
            socket.send(JSON.stringify({ type: 'command_outcome', status: 'accepted', update: {
              base_revision: 10, frame: next, presentation, receipt: null,
            }}));
          };
        });
      });
      await page.goto('/');
      await expect(page.getByTestId('connection-badge')).toHaveText('connected');
      await page.clock.install({ time: 0 });
      await page.clock.pauseAt(10000);
      await page.getByRole('button', { name: 'New Game', exact: true }).click();
      await page.getByText('History and settings', { exact: true }).click();
      const panel = page.getByTestId('presentation-history');
      await panel.locator('summary').click();
      await expect(panel.getByRole('listitem')).toHaveCount(100);
      await expect(panel).toContainText('6–105 of 105');
      await expect(page.getByTestId('action-option').first()).toBeEnabled();
      const geometry = await page.evaluate(() => {
        const rect = (testid: string): DOMRect => document.querySelector(`[data-testid="${testid}"]`)!.getBoundingClientRect();
        const board = rect('game-board');
        const stage = rect('presentation-stage');
        return { width: innerWidth, height: innerHeight, dpr: devicePixelRatio, scale: visualViewport!.scale, scroll: scrollY,
          board: board.toJSON(), stage: stage.toJSON(), intersects: stage.left < board.right && stage.right > board.left && stage.top < board.bottom && stage.bottom > board.top };
      });
      expect(geometry.intersects).toBe(false);
      expect(geometry.scroll).toBe(0);
      expect(geometry.scale).toBe(1);
      await testInfo.attach('history-geometry', { body: JSON.stringify(geometry), contentType: 'application/json' });
      await page.getByRole('button', { name: 'Pause narration' }).click();
      await panel.getByRole('button', { name: 'Earlier events' }).click();
      const region = panel.getByRole('list', { name: 'Recorded game events' });
      await region.focus();
      await region.evaluate(element => { element.scrollTop = 80; });
      const before = await region.evaluate(element => element.scrollTop);
      append();
      await expect(panel.getByRole('button', { name: 'Latest (1 unread)' })).toBeVisible();
      expect(await region.evaluate(element => element.scrollTop)).toBe(before);
      await expect(region).toBeFocused();
      await expect(panel.locator('[data-presentation-seq="0"]')).toHaveCount(1);
      await panel.getByRole('button', { name: /^Inspect Otter-Penguin/ }).first().click();
      await expect(panel.getByRole('region', { name: 'History card inspection' })).toContainText('Otter-Penguin');
      expect(commands).toEqual([]);
      expect(await panel.getByRole('button', { name: /^Inspect / }).evaluateAll(buttons => buttons.every(button => button.scrollHeight <= button.clientHeight + 1 && button.scrollWidth <= button.clientWidth + 1))).toBe(true);
      await panel.getByRole('button', { name: 'Close history inspection' }).click();
      await panel.getByRole('button', { name: 'Latest (1 unread)' }).click();
      await expect(panel).toContainText('7–106 of 106');
      await expect(panel.getByRole('button', { name: 'Latest', exact: true })).toBeVisible();
      await page.screenshot({ path: testInfo.outputPath('history-reading.png') });
      await page.getByRole('button', { name: 'Finish', exact: true }).click();
      await expect(page.getByTestId('presentation-stage')).toBeHidden();
      await page.getByTestId('action-option').first().click();
      await expect.poll(() => commands.length).toBe(1);
      expect(commands[0].expected_revision).toBe(11);
      expect(await page.evaluate(() => scrollY)).toBe(0);
      await page.screenshot({ path: testInfo.outputPath('read-history.png') });
    });
  }
}

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


test('player damage emphasizes its exact subject while choices stay focused', async ({ page }) => {
  const event: PresentationEvent = {
    ...combatSequence.events[0] as PresentationEvent, seq: 0,
    kind: { kind: 'damage', source: null, target: { kind: 'player', id: 0 }, amount: 1 },
  };
  await installCombatAuthority(page, [], [event]);
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.clock.install({ time: 0 });
  await page.clock.pauseAt(10000);
  await page.getByRole('button', { name: 'New Game', exact: true }).click();
  await expect(page.locator('[data-player-id="0"]')).toHaveAttribute('data-narrated-seq', '0');
  await expect(page.locator('[data-player-id="0"]')).toHaveCSS('outline-style', 'solid');
  await expect(page.locator('[data-player-id="1"]')).not.toHaveAttribute('data-narrated-seq');
  await expect(page.getByTestId('action-option').first()).toBeFocused();
  await page.clock.runFor(1000);
  await expect(page.locator('[data-player-id="0"]')).not.toHaveAttribute('data-narrated-seq');
});

test('partial history recovery preserves event pixels, focus and recovered unread identities', async ({ page }) => {
  const events: PresentationEvent[] = Array.from({ length: 105 }, (_, index) => ({
    ...structuredClone(combatSequence.events[0]) as PresentationEvent,
    seq: 900 + index, from_revision: 8, to_revision: 9, suggested_ms: 80,
  }));
  events[0].kind = { kind: 'turn_started', player: 0, turn_number: 1 };
  events[50].kind = { kind: 'turn_started', player: 0, turn_number: 2 };
  let recover = (_events: PresentationEvent[], _match = 'combat-prompt-proof'): void => { throw new Error('Not connected'); };
  await page.routeWebSocket('**/ws/play', socket => {
    recover = (tail, match = 'combat-prompt-proof') => {
      const frame = frameAt('attack');
      frame.match_id = match;
      if (match === 'different-viewer-scope') frame.projection.opponent.battlefield[0].name = 'Scope-only name';
      socket.send(JSON.stringify({ type: 'observation', table: tableSnapshot(), data: frame.projection, actions: [],
        session_id: 'partial-history', resume_token: 'partial-history', recovery: {
          ...structuredClone(boltFixture.recovery), frame, presentation_tail: tail, presentation_cursor: tail[0]?.seq ?? 0,
        },
      }));
    };
    socket.onMessage(raw => {
      if (JSON.parse(String(raw)).type === 'new_game') {
        // Each transport tail is contiguous; retained history can have intervals
        // between tails when reconnect misses earlier events.
        recover(events.slice(0, 1));
        recover(events.slice(3, 50));
        recover(events.slice(51));
      }
    });
  });
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.clock.install({ time: 0 });
  await page.clock.pauseAt(10000);
  await page.getByRole('button', { name: 'New Game', exact: true }).click();
  await page.getByText('History and settings', { exact: true }).click();
  const panel = page.getByTestId('presentation-history');
  await panel.locator('summary').click();
  const region = panel.getByRole('list', { name: 'Recorded game events' });
  await expect(panel).toContainText('Events 950–950 unavailable');
  await expect(region.getByRole('heading').first()).toHaveText('Turn unknown');
  await region.evaluate(element => { element.scrollTop = 200; });
  const anchor = await region.evaluate(element => {
    const row = Array.from(element.querySelectorAll<HTMLElement>('[data-presentation-seq]'))
      .find(row => row.getBoundingClientRect().bottom > element.getBoundingClientRect().top)!;
    return { seq: row.dataset.presentationSeq!, top: row.getBoundingClientRect().top };
  });
  const row = panel.locator(`[data-presentation-seq="${anchor.seq}"]`);
  await row.getByRole('button').first().focus();
  const top = (await row.boundingBox())!.y;
  recover(events.slice(1, 3));
  await expect(panel.getByRole('button', { name: 'Latest (2 unread)' })).toBeVisible();
  recover(events.slice(50, 51));
  await expect(panel.getByRole('button', { name: 'Latest (3 unread)' })).toBeVisible();
  expect(Math.abs((await row.boundingBox())!.y - top)).toBeLessThan(1);
  await expect(row.getByRole('button').first()).toBeFocused();
  await expect(panel.locator('[data-history-gap]')).toHaveCount(0);
  // The same recovery contains no newly available facts and cannot restart motion.
  await page.getByRole('button', { name: 'Finish', exact: true }).click();
  recover(events);
  await expect(page.getByTestId('presentation-stage')).toBeHidden();
  await expect(panel.getByRole('button', { name: 'Latest (3 unread)' })).toBeVisible();
  await panel.getByRole('button', { name: 'Latest (3 unread)' }).click();
  await expect(panel.getByRole('button', { name: 'Latest', exact: true })).toBeVisible();
  await panel.locator('[data-presentation-seq="1004"]').getByRole('button').first().click();
  await expect(panel.getByRole('region', { name: 'History card inspection' })).toBeFocused();
  recover([events[3]], 'different-viewer-scope');
  await expect(region.locator('[data-presentation-seq]')).toHaveCount(1);
  await expect(panel.getByRole('button', { name: 'Latest', exact: true })).toBeVisible();
  await expect(panel.getByRole('region', { name: 'History card inspection' })).toHaveCount(0);
  await expect(region).toBeFocused();
  await expect(panel).toContainText('Scope-only name');
  await expect(panel).not.toContainText('Badgermole Cub');
  recover([events[3]]);
  await expect(panel).toContainText('Badgermole Cub');
  await expect(panel).not.toContainText('Scope-only name');
  await expect(panel.getByRole('button', { name: 'Latest', exact: true })).toBeVisible();
  expect(await page.evaluate(() => scrollY)).toBe(0);
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
