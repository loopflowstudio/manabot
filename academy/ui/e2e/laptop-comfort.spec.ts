import { writeFile } from 'node:fs/promises';
import { finishBoardNarration } from './presentation-readiness';
import { expect, test, type Locator, type Page } from '@playwright/test';
import boltJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type { Command, ExperienceFrame, RecoveryEnvelope } from '../src/lib/types';
import { denseFrame } from './fixtures/laptop-table';

interface TableOptions {
  sparse?: boolean;
  offerCount?: number;
  longInstruction?: boolean;
  holdCommand?: boolean;
}

async function installTable(page: Page, options: TableOptions = {}): Promise<{
  commands: Command[]; frame: ExperienceFrame; rejectPending: () => void;
}> {
  const frame = denseFrame();
  if (options.sparse) {
    frame.projection.agent.battlefield = frame.projection.agent.battlefield.filter(card => card.types?.is_land).slice(0, 1);
    frame.projection.opponent.battlefield = [];
    frame.projection.agent.hand = frame.projection.agent.hand.slice(0, 7);
  }
  // A large published offer set over exact visible objects, including identical copies.
  frame.offers = Array.from({ length: options.offerCount ?? frame.projection.agent.battlefield.length }, (_, index) => {
    const card = frame.projection.agent.battlefield[index % frame.projection.agent.battlefield.length];
    return {
      ...frame.offers[0], id: 1000 + index, label: `Inspect choice ${index + 1}: ${card.name}`,
      focus: [card.id],
    };
  });
  if (options.longInstruction) {
    frame.prompt!.instruction = `${'Read this instruction before choosing. '.repeat(60)}End of the complete instruction.`;
  }
  let rejectPending = (): void => { throw new Error('No pending command'); };
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
        rejectPending = () => {
          frame.revision += 1;
          frame.frame_hash = `laptop-${frame.revision}`;
          socket.send(JSON.stringify({ type: 'command_outcome', status: 'rejected',
            rejection: { command_id: message.command.command_id, code: 'stale_revision',
              message: 'Inspect the current decision.', current_revision: frame.revision,
              current_prompt: frame.prompt!.id }, recovery: recover() }));
        };
        if (!options.holdCommand) rejectPending();
      }
    });
  });
  await page.route('**/api/card-art/**', route => route.fulfill({ status: 404 }));
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'New Game', exact: true }).click();
  await expect(page.getByTestId('action-option')).toHaveCount(frame.offers.length);
  return { commands, frame, rejectPending: () => rejectPending() };
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

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const sparse of [true, false]) {
    test(`battlefield lane access ${sparse ? 'sparse' : 'dense'} ${viewport.width}`, async ({ page }, testInfo) => {
      await page.setViewportSize(viewport);
      const { commands, frame } = await installTable(page, { sparse });
      await page.evaluate(() => document.fonts.ready);
      const zones = page.getByRole('region', { name: 'You battlefield and zones', exact: true });
      const zoneBounds = (await zones.boundingBox())!;
      if (sparse) {
        const land = (await zones.locator('[data-object-id]').first().boundingBox())!;
        expect(land.y).toBeGreaterThanOrEqual(zoneBounds.y);
        expect(land.y + land.height).toBeLessThanOrEqual(zoneBounds.y + zoneBounds.height);
        expect(await zones.evaluate(element => element.scrollTop)).toBe(0);
      }
      const lanes = zones.locator('[data-battlefield-lane].occupied');
      for (const lane of await lanes.all()) {
        const heading = (await lane.locator('h3').boundingBox())!;
        expect(heading.y).toBeGreaterThanOrEqual(zoneBounds.y);
        expect(heading.y + heading.height).toBeLessThanOrEqual(zoneBounds.y + zoneBounds.height);
        const seen = new Set<string>();
        const next = lane.getByRole('button', { name: /^Next / });
        do {
          for (const card of await lane.locator('[data-object-id]').all()) {
            seen.add((await card.getAttribute('data-object-id'))!);
            await card.focus();
            const cardBounds = (await card.boundingBox())!;
            const visibleZone = (await zones.boundingBox())!;
            expect(cardBounds.y).toBeGreaterThanOrEqual(visibleZone.y);
            expect(cardBounds.y + cardBounds.height).toBeLessThanOrEqual(visibleZone.y + visibleZone.height);
          }
          if (!await next.count() || !await next.isEnabled()) break;
          await next.focus();
          await expect(next).toBeInViewport();
          await next.press('Enter');
        } while (true);
        const kind = await lane.getAttribute('data-battlefield-lane');
        const expected = frame.projection.agent.battlefield.filter(card => kind === 'lands' ? card.types?.is_land : card.types?.is_creature);
        expect([...seen].sort()).toEqual(expected.map(card => String(card.id)).sort());
      }
      if (sparse) {
        await expect(page.getByRole('region', { name: 'Search 64 battlefield and zones', exact: true }).getByText('Creatures · 0', { exact: true })).toBeVisible();
      }
      expect(await page.evaluate(() => scrollY)).toBe(0);
      expect(commands).toEqual([]);
      await page.screenshot({ path: testInfo.outputPath('battlefield.png') });
      await writeFile(testInfo.outputPath('battlefield-geometry.json'), JSON.stringify(await zones.evaluate(element => ({
        bounds: element.getBoundingClientRect().toJSON(), scrollTop: element.scrollTop, scrollHeight: element.scrollHeight,
        lanes: Array.from(element.querySelectorAll('[data-battlefield-lane]'), lane => ({ label: lane.getAttribute('aria-label'), bounds: lane.getBoundingClientRect().toJSON() })),
      })), null, 2));
    });
  }
}

async function expectInsideViewport(locator: Locator): Promise<void> {
  await expect(locator).toBeVisible();
  const bounds = (await locator.boundingBox())!;
  const viewport = locator.page().viewportSize()!;
  expect(bounds.y).toBeGreaterThanOrEqual(0);
  expect(bounds.y + bounds.height).toBeLessThanOrEqual(viewport.height);
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`persistent context through unfiltered browsing ${viewport.width} ${colorScheme}`, async ({ page }, testInfo) => {
      await page.setViewportSize(viewport);
      await page.emulateMedia({ colorScheme });
      const { commands, frame, rejectPending } = await installTable(page, {
        offerCount: 100, longInstruction: true, holdCommand: true,
      });
      const offers = page.getByTestId('action-option');
      const list = page.getByRole('group', { name: 'Legal choices', exact: true });
      const prompt = page.getByTestId('decision-prompt');
      const filter = page.getByRole('searchbox', { name: 'Find an action' });
      const count = page.getByTestId('legal-choice-count');
      const pass = page.getByTestId('pass-turn');
      const automation = page.getByTestId('automation-status');
      const assertRegions = async (): Promise<void> => {
        const boxes = await page.evaluate(() => {
          const box = (selector: string): DOMRect => document.querySelector(selector)!.getBoundingClientRect();
          return { status: box('[data-testid="current-decision"]').toJSON(),
            decision: box('[data-testid="decision-surface"]').toJSON(),
            automation: box('[data-testid="automation-status"]').toJSON(),
            secondary: box('.secondary-panels').toJSON(),
            rail: box('#table-choices').toJSON(), board: box('#table-board').toJSON() };
        });
        expect(boxes.status.bottom).toBeLessThanOrEqual(boxes.decision.top);
        expect(boxes.decision.bottom).toBeLessThanOrEqual(boxes.automation.top);
        expect(boxes.automation.bottom).toBeLessThanOrEqual(boxes.secondary.top);
        expect(boxes.secondary.bottom).toBeLessThanOrEqual(boxes.rail.bottom);
        expect(boxes.board.top).toBeGreaterThanOrEqual(0);
        expect(boxes.board.bottom).toBeLessThanOrEqual(viewport.height);
        expect(await page.evaluate(() => scrollY)).toBe(0);
      };
      // Browse the full list with the keyboard; never reduce it to one search result.
      await offers.first().focus();
      for (let index = 1; index < frame.offers.length; index += 1) await page.keyboard.press('Tab');
      await expect(offers.last()).toBeFocused();
      for (const control of [prompt, filter, count, pass, automation, offers.last()]) await expectInsideViewport(control);
      expect(await list.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
      await expect(count).toHaveText('100 of 100 legal choices');
      // All of a long instruction remains independently keyboard-readable.
      await prompt.focus();
      await prompt.press('End');
      await expect.poll(() => prompt.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
      await expectInsideViewport(offers.last());
      expect(commands).toEqual([]);

      // Several offers for an exact copy: Show all must survive last-offer browsing.
      const subject = frame.offers.at(-1)!.focus[0];
      await page.locator(`[data-object-id="${subject}"]`).click();
      const showAll = page.getByRole('button', { name: 'Show all', exact: true });
      await offers.last().focus();
      await expectInsideViewport(showAll);
      await showAll.press('Enter');
      await expect(offers).toHaveCount(100);
      await filter.fill('nothing matches');
      await expect(offers).toHaveCount(0);
      await expectInsideViewport(page.getByRole('button', { name: 'Clear filter', exact: true }));
      await filter.press('Escape');
      await expect(offers).toHaveCount(100);

      const settings = page.getByText('History and settings', { exact: true });
      await settings.click();
      await page.getByTestId('stops-panel').locator('summary').click();
      const secondary = page.getByRole('region', { name: 'History and settings', exact: true });
      await secondary.focus();
      await secondary.press('End');
      await offers.last().focus();
      for (const control of [prompt, count, filter, automation, offers.last()]) await expectInsideViewport(control);
      await assertRegions();
      await page.screenshot({ path: testInfo.outputPath('last-offer-context.png') });
      const revision = frame.revision;
      const last = frame.offers.at(-1)!;
      await offers.last().press('Enter');
      await expect.poll(() => commands.length).toBe(1);
      expect(commands[0]).toMatchObject({ expected_revision: revision, offer_id: last.id,
        match_id: frame.match_id, prompt_id: frame.prompt!.id });
      await expect(offers.last()).toBeDisabled();
      await expectInsideViewport(page.getByText('Actions unavailable', { exact: true }));
      // Recover a different pending frame; navigation cannot retain an old offer or filter.
      frame.prompt!.id += 1;
      frame.prompt!.instruction = 'Recovered pending decision';
      frame.offers = frame.offers.slice(0, 12).map(offer => ({ ...offer, id: offer.id + 2000 }));
      rejectPending();
      await expect(page.getByTestId('decision-surface')).toHaveAttribute('data-revision', String(revision + 1));
      await expect(prompt).toHaveText('Recovered pending decision');
      await expect(filter).toHaveValue('');
      await expect(offers).toHaveCount(12);
      await expect(offers.first()).toBeFocused();
      await expectInsideViewport(offers.first());
      await expectInsideViewport(count);
      expect(commands).toHaveLength(1);
      expect(await page.evaluate(() => scrollY)).toBe(0);
      await writeFile(testInfo.outputPath('context-geometry.json'), JSON.stringify(await page.evaluate(() => ({
        viewport: { width: innerWidth, height: innerHeight, scale: visualViewport!.scale, dpr: devicePixelRatio },
        controls: [...document.querySelectorAll('[data-testid="decision-prompt"], [data-testid="legal-choice-count"], .offer-list')]
          .map(node => ({ bounds: node.getBoundingClientRect().toJSON(), scroll: node.scrollTop })),
      })), null, 2));
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
