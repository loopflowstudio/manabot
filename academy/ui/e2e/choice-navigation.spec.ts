import { gameReply } from './fixtures/game';
import { expect, test, type Page } from '@playwright/test';
import positionsJson from '../../../managym/academy/fixtures/choice-navigation-opening-v1.json' with { type: 'json' };
import boltJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type { Command, ExperienceFrame, RecoveryEnvelope } from '../src/lib/types';

/** Setup with the matchup named, so first-visit defaults do not choose one. */
const SETUP = '/?hero=ur_lessons&villain=gw_allies';

// Retained openings pay by hand; Auto-tap would submit their payments before the test reads them.
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('academy.auto-tap', 'off'));
});

test.beforeEach(({ page }) => { page.on('dialog', dialog => dialog.accept()); });

const positions = positionsJson.positions;
type Family = keyof typeof positions;

declare global {
  interface Window {
    __choiceNavigation: {
      socket: WebSocket | null;
      frame: ExperienceFrame | null;
      status: string;
      working: boolean;
    };
  }
}

async function installPosition(page: Page, family: Family): Promise<Command[]> {
  const frame = positions[family].frame as ExperienceFrame;
  const commands: Command[] = [];
  const recovery: RecoveryEnvelope = {
    ...(boltJson.recovery as RecoveryEnvelope),
    frame,
    presentation_tail: [],
    accepted_commands: [],
    replay_cursor: frame.revision,
  };
  const game = gameReply();
  await page.routeWebSocket('**/ws/play', socket => {
    socket.onMessage(raw => {
      const message = JSON.parse(String(raw));
      if (message.type === 'new_game' || message.type === 'resume') {
        socket.send(JSON.stringify({ type: 'observation', ...game, data: frame.projection,
          actions: [], recovery }));
      } else if (message.type === 'command') {
        commands.push(message.command);
        // Exercise the existing recovery path without claiming mocked acceptance.
        socket.send(JSON.stringify({ type: 'command_outcome', status: 'rejected',
          rejection: { command_id: message.command.command_id, code: 'stale_revision',
            message: 'Recovering the current decision.', current_revision: frame.revision,
            current_prompt: frame.prompt!.id }, recovery }));
      }
    });
  });
  await page.goto(SETUP);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game', exact: true }).first().click();
  await expect(page.getByTestId('action-panel')).toHaveAttribute('data-action-space-kind', family);
  return commands;
}

const FAMILIES = ['CHOOSE_TARGET', 'DISCARD', 'LEARN', 'DECLARE_BLOCKER', 'PAY_OR_NOT'] as const;

/** Where the board draws an offer besides its rail row: a card, a player, a prompt button or the tray. */
function drawn(page: Page, offer: { id: number; label: string }, number: number) {
  return page.locator([
    `[data-testid=board-card][aria-label*="${offer.label} (${number})"]`,
    `[data-testid=board-card][data-offer-count]:not([data-offer-count="1"])[aria-label*="${offer.label}"]`,
    `[data-testid=player-target][aria-label*="${offer.label} (${number})"]`,
    `[data-testid=primary-action][data-offer-id="${offer.id}"]`,
    `[data-testid=prompt-action][data-offer-id="${offer.id}"]`,
    `[data-testid=tray] button:has-text("${offer.label}")`,
  ].join(', '));
}

for (const family of FAMILIES) {
  test(`${family}: every offer is a numbered row and drawn on the board, and a click commits exactly it`, async ({ page }) => {
    const commands = await installPosition(page, family);
    const frame = positions[family].frame as ExperienceFrame;
    await expect(page.getByTestId('action-option')).toHaveCount(frame.offers.length);
    await expect(page.getByTestId('legal-choice-count')).toContainText(String(frame.offers.length));
    expect(await page.getByTestId('action-option').evaluateAll(rows => rows.map(row => Number(row.getAttribute('data-offer-id')))))
      .toEqual(frame.offers.map(offer => offer.id));
    for (const [index, offer] of frame.offers.entries()) {
      expect(await drawn(page, offer, index + 1).count(), offer.label).toBeGreaterThanOrEqual(1);
    }
    // Pointing and inspecting send nothing.
    await page.getByTestId('board-card').first().hover();
    expect(commands).toEqual([]);

    // A card with exactly one offer commits that offer; a number key commits its row.
    const card = page.locator('[data-testid=board-card].legal[data-offer-count="1"]').first();
    const number = Number((await card.getAttribute('aria-label'))!.match(/\((\d+)\)$/)![1]);
    const target = frame.offers[number - 1];
    const box = (await card.boundingBox())!;
    // Inside the card's own edge: an attacker's tilt puts its box's corners outside it.
    await card.click({ position: { x: box.width - 16, y: box.height / 2 } });
    await expect.poll(() => commands.length).toBe(1);
    expect(commands[0]).toMatchObject({ offer_id: target.id, expected_revision: frame.revision, prompt_id: frame.prompt!.id, answers: [] });
    await expect(page.getByTestId('action-option')).toHaveCount(frame.offers.length);
    await page.mouse.move(2, 2);
    await page.keyboard.press(String(Math.min(2, frame.offers.length)));
    await expect.poll(() => commands.length).toBe(2);
    expect(commands[1].offer_id).toBe(frame.offers[Math.min(2, frame.offers.length) - 1].id);
  });
}

test('Learn draws Lessons from outside the game in the tray, and declining stays a button', async ({ page }) => {
  const commands = await installPosition(page, 'LEARN');
  const frame = positions.LEARN.frame as ExperienceFrame;
  const lessons = frame.offers.filter(offer => offer.action_type === 'LEARN_TAKE_LESSON');
  await expect(page.getByTestId('tray').getByTestId('board-card')).toHaveCount(lessons.length);
  await expect(page.getByTestId('prompt-action').filter({ hasText: 'Decline Learn' })).toBeVisible();
  // A click on the Learn spell never declines it.
  await expect(page.getByTestId('stack-entry').locator('xpath=self::*[contains(@class, "legal")]')).toHaveCount(0);
  expect(commands).toEqual([]);
});

// The review path uses the real backend, the retained legal prefix and ordinary
// Commands. Set ACADEMY_REVIEW_CHOICE to a family and run headed to pause for play.
for (const family of ['CHOOSE_TARGET', 'DISCARD'] as const) {
  test(`live retained ${family} can execute an intended choice`, async ({ page }) => {
    test.skip(Boolean(process.env.ACADEMY_REVIEW_CHOICE && process.env.ACADEMY_REVIEW_CHOICE !== family));
    const position = positions[family];
    await page.addInitScript(config => {
      const state: Window['__choiceNavigation'] = {
        socket: null, frame: null, status: '', working: false,
      };
      const Native = window.WebSocket;
      class TrackedSocket extends Native {
        constructor(url: string | URL, protocols?: string | string[]) {
          super(url, protocols);
          state.socket = this;
          this.addEventListener('message', event => {
            const message = JSON.parse(event.data);
            if (message.work) state.working = message.work.state === 'running';
            if (message.type === 'command_outcome') state.status = message.status;
            const findFrame = (value: unknown): void => {
              if (!value || typeof value !== 'object') return;
              if ('frame_hash' in value && 'projection' in value) state.frame = value as ExperienceFrame;
              else for (const child of Object.values(value)) findFrame(child);
            };
            findFrame(message);
          });
        }
        send(data: string): void {
          const message = JSON.parse(data);
          if (message.type === 'new_game') message.config = config;
          super.send(JSON.stringify(message));
        }
      }
      window.__choiceNavigation = state;
      window.WebSocket = TrackedSocket;
    }, position.config);
    await page.goto(SETUP);
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    await page.getByRole('button', { name: 'Start game', exact: true }).first().click();
    await expect(page.getByTestId('game')).toBeVisible();
    for (const [ordinal, step] of position.prefix.entries()) {
      await expect.poll(() => page.evaluate(() => window.__choiceNavigation.working)).toBe(false);
      const revision = await page.evaluate(({ ordinal, step }) => {
        const state = window.__choiceNavigation;
        const frame = state.frame!;
        if (frame.offers.find(offer => offer.id === step.offer_id)?.label !== step.label) {
          throw new Error(`Retained prefix diverged at ${ordinal}`);
        }
        state.socket!.send(JSON.stringify({ type: 'command', command: {
          command_id: `navigation.live.${ordinal}`, match_id: frame.match_id,
          expected_revision: frame.revision, prompt_id: frame.prompt!.id,
          offer_id: step.offer_id, answers: [],
        } }));
        return frame.revision;
      }, { ordinal, step });
      await expect.poll(() => page.evaluate(() =>
        window.__choiceNavigation.frame!.revision,
      )).toBeGreaterThan(revision);
    }
    await expect(page.getByTestId('action-panel')).toHaveAttribute('data-action-space-kind', family);
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    const actual = await page.evaluate(() =>
      window.__choiceNavigation.frame!,
    );
    expect(actual.offers).toEqual(position.frame.offers);
    expect(actual.projection).toEqual(position.frame.projection);
    if (process.env.ACADEMY_REVIEW_CHOICE === family) {
      await page.pause();
      return;
    }
    // The intended choice, made on the card it acts on.
    const intended = position.frame.offers[0];
    const card = drawn(page, intended, 1).first();
    const box = (await card.boundingBox())!;
    // Inside the card's own edge: an attacker's tilt puts its box's corners outside it.
    await card.click({ position: { x: box.width - 16, y: box.height / 2 } });
    await expect.poll(() => page.evaluate(() =>
      window.__choiceNavigation.frame!.revision,
    )).toBeGreaterThan(actual.revision);
    expect(await page.evaluate(() =>
      window.__choiceNavigation.status,
    )).toBe('accepted');
  });
}
