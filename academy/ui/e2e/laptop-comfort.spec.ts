import { writeFile } from 'node:fs/promises';
import { finishBoardNarration } from './presentation-readiness';
import { expect, test, type Page } from '@playwright/test';
import boltJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type { Command, ExperienceFrame, RecoveryEnvelope } from '../src/lib/types';
import { denseFrame } from './fixtures/laptop-table';

async function installTable(page: Page): Promise<{ commands: Command[]; frame: ExperienceFrame }> {
  const frame = denseFrame();
  // A large published offer set over exact visible objects, including identical copies.
  frame.offers = frame.projection.agent.battlefield.map((card, index) => ({
    ...frame.offers[0], id: 1000 + index, label: `Inspect choice ${index + 1}: ${card.name}`,
    focus: [card.id],
  }));
  const commands: Command[] = [];
  await page.routeWebSocket('**/ws/play', socket => {
    const recover = (): RecoveryEnvelope => ({ ...boltJson.recovery as RecoveryEnvelope,
      frame, presentation_tail: [], accepted_commands: [], replay_cursor: frame.revision });
    socket.onMessage(raw => {
      const message = JSON.parse(String(raw));
      if (message.type === 'new_game') {
        socket.send(JSON.stringify({ type: 'observation', data: frame.projection, actions: [],
          session_id: 'laptop', resume_token: 'laptop', recovery: recover() }));
      } else if (message.type === 'command') {
        commands.push(message.command);
        frame.revision += 1;
        frame.frame_hash = `laptop-${frame.revision}`;
        socket.send(JSON.stringify({ type: 'command_outcome', status: 'rejected',
          rejection: { command_id: message.command.command_id, code: 'stale_revision',
            message: 'Inspect the current decision.', current_revision: frame.revision,
            current_prompt: frame.prompt!.id }, recovery: recover() }));
      }
    });
  });
  await page.route('**/api/card-art/**', route => route.fulfill({ status: 404 }));
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'New Game', exact: true }).click();
  await expect(page.getByTestId('action-option')).toHaveCount(frame.offers.length);
  return { commands, frame };
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`laptop choices, inspection and recovery ${viewport.width} ${colorScheme}`, async ({ browser }, testInfo) => {
      const context = await browser.newContext({ viewport, colorScheme, deviceScaleFactor: 1 });
      const page = await context.newPage();
      const { commands, frame } = await installTable(page);
      await page.evaluate(() => document.fonts.ready);
      const geometry = await page.evaluate(() => {
        const bounds = (selector: string): DOMRect => document.querySelector(selector)!.getBoundingClientRect();
        return { width: innerWidth, height: innerHeight, dpr: devicePixelRatio,
          scale: visualViewport!.scale, scroll: scrollY, documentWidth: document.documentElement.scrollWidth,
          workspace: bounds('[data-testid="play-workspace"]').toJSON(),
          hand: bounds('[aria-label="You hand"]').toJSON(),
          choices: bounds('#table-choices').toJSON() };
      });
      expect(geometry.scale).toBe(1);
      expect(geometry.scroll).toBe(0);
      expect(geometry.documentWidth).toBe(viewport.width);
      for (const box of [geometry.workspace, geometry.hand, geometry.choices]) {
        expect(box.top).toBeGreaterThanOrEqual(0);
        expect(box.bottom).toBeLessThanOrEqual(viewport.height);
      }
      expect(geometry.choices.left).toBeGreaterThan(geometry.hand.right);
      await writeFile(testInfo.outputPath('geometry.json'), JSON.stringify(geometry, null, 2));
      await page.screenshot({ path: testInfo.outputPath('initial-table.png') });

      const filter = page.getByRole('searchbox', { name: 'Find an action' });
      for (const offer of frame.offers) {
        await filter.fill(offer.label);
        const option = page.getByTestId('action-option').filter({ hasText: offer.label });
        await expect(option).toHaveCount(1);
        await option.focus();
        // Every exact copy remains reachable, even on another battlefield page.
        await expect(page.locator(`[data-object-id="${offer.focus[0]}"]`)).toBeVisible();
      }
      await filter.fill('nothing matches');
      await filter.press('Escape');
      await expect(page.getByTestId('action-option')).toHaveCount(frame.offers.length);
      expect(commands).toEqual([]);
      // Keyboard card inspection changes navigation only.
      const hand = page.getByRole('region', { name: 'You hand', exact: true });
      await hand.getByRole('button').last().focus();
      const handBounds = (await hand.boundingBox())!;
      const lastCard = (await hand.getByRole('button').last().boundingBox())!;
      expect(lastCard.x + lastCard.width).toBeLessThanOrEqual(handBounds.x + handBounds.width);
      await hand.getByRole('button').first().focus();
      await hand.getByRole('button').first().press('Enter');
      await expect(page.getByRole('region', { name: 'Inspected card' })).toBeVisible();
      expect(commands).toEqual([]);
      await page.getByRole('button', { name: 'Show all', exact: false }).click();
      const revision = frame.revision;
      const first = frame.offers[0];
      await page.getByTestId('action-option').first().click();
      await expect.poll(() => commands.length).toBe(1);
      expect(commands[0]).toMatchObject({ expected_revision: revision, offer_id: first.id,
        match_id: frame.match_id, prompt_id: frame.prompt!.id });
      await expect(page.getByTestId('decision-surface')).toHaveAttribute('data-revision', String(revision + 1));
      expect(await page.evaluate(() => scrollY)).toBe(0);
      const recoveredHand = await hand.boundingBox();
      expect(recoveredHand!.y + recoveredHand!.height).toBeLessThanOrEqual(viewport.height);
      await page.screenshot({ path: testInfo.outputPath('laptop-table.png') });
      await context.close();
    });
  }
}

// Actual server Commands supplement the deterministic access fixtures.
for (const [seed, viewport] of [
  [0, { width: 1440, height: 900 }],
  [1, { width: 1280, height: 800 }],
] as const) {
  test(`ordinary seeded laptop game ${seed}`, async ({ page }, testInfo) => {
    test.setTimeout(120_000);
    await page.setViewportSize(viewport);
    await page.routeWebSocket('**/ws/play', socket => {
      const server = socket.connectToServer();
      socket.onMessage(raw => {
        const message = JSON.parse(String(raw));
        if (message.type === 'new_game') message.config.seed = seed;
        server.send(JSON.stringify(message));
      });
    });
    await page.goto('/');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    await page.getByTestId('opponent-select').selectOption('random');
    await page.getByRole('button', { name: 'New Game', exact: true }).click();
    const receipts: { kind: string | null; label: string | null; scroll: number }[] = [];
    try {
      for (let input = 0; input < 120; input += 1) {
        const options = page.getByTestId('action-option');
        const result = page.getByTestId('game-result');
        await expect(page.locator('[data-testid="action-option"], [data-testid="learn-mode"]:not(:disabled), [data-testid="game-result"]').first()).toBeVisible();
        if (await result.isVisible()) return;
        await finishBoardNarration(page);
        if (await options.count() === 0) {
          await page.getByTestId('learn-mode').first().click();
          continue;
        }
        expect(await page.evaluate(() => scrollY)).toBe(0);
        const before = await page.getByTestId('decision-surface').getAttribute('data-revision');
        const labels = await options.allTextContents();
        let pick = labels.findIndex(label => /Play (Island|Mountain)|Cast /.test(label));
        if (pick < 0) pick = input % labels.length;
        const chosen = options.nth(pick);
        receipts.push({ kind: await page.getByTestId('action-panel').getAttribute('data-action-space-kind'),
          label: await chosen.textContent(), scroll: await page.evaluate(() => scrollY) });
        await chosen.focus();
        await chosen.press('Enter');
        await expect.poll(() => page.getByTestId('decision-surface').getAttribute('data-revision')).not.toBe(before);
      }
      throw new Error('Seeded game did not finish within 120 deliberate inputs');
    } finally {
      await writeFile(testInfo.outputPath('game-receipt.json'), JSON.stringify({ seed, viewport, receipts }, null, 2));
      await finishBoardNarration(page);
      await page.screenshot({ path: testInfo.outputPath('game-final.png') });
    }
  });
}

test('narrow table keeps explicit board and choices navigation', async ({ page }) => {
  await page.setViewportSize({ width: 900, height: 800 });
  const { commands } = await installTable(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(900);
  await page.getByRole('link', { name: 'Choose a play', exact: true }).click();
  await expect(page.locator('#table-choices')).toBeInViewport();
  await page.getByRole('link', { name: 'Inspect board', exact: true }).click();
  await expect(page.locator('#table-board')).toBeInViewport();
  expect(commands).toEqual([]);
});
