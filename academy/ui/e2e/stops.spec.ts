import { selectDeck } from './deck-selection';
import { expect, test, type Page } from '@playwright/test';

// Priority stops e2e: the same small deal (seed 0, passive villain,
// scripted deterministic hero) is played twice —
// first with auto-pass disabled (the old surface-every-window behavior),
// then with stops on main phases only. The stops run must need dramatically
// fewer deliberate inputs to reach the same turn boundary, while the log narrates what
// happened inside the fast-forwarded stretches.

const MAX_ACTIONS = 120;
const COMPARISON_TURN = 7;

test.beforeEach(async ({ page }) => {
  // Ordinary play generates a new seed; this comparison needs a fixed deal.
  await page.routeWebSocket('**/ws/play', socket => {
    const server = socket.connectToServer();
    socket.onMessage(message => {
      if (typeof message !== 'string') { server.send(message); return; }
      const payload = JSON.parse(message);
      if (payload.type === 'new_game') payload.config = {
        ...payload.config, seed: 0,
        hero_deck: { Island: 4, Counterspell: 4 },
        villain_deck: { Plains: 8 }, villain_type: 'passive',
      };
      server.send(JSON.stringify(payload));
    });
  });
});

async function setCheckbox(page: Page, testId: string, checked: boolean): Promise<void> {
  const box = page.getByTestId(testId);
  if ((await box.isChecked()) !== checked) {
    await box.click();
    if (await page.getByTestId('automation-status').count()) {
      await expect(page.getByTestId('automation-status')).not.toContainText('updating…');
    }
  }
}

async function openStopsPanel(page: Page): Promise<void> {
  const secondary = page.locator('details.secondary-panels');
  if (await secondary.count() && !await secondary.evaluate((node: HTMLDetailsElement) => node.open)) {
    await secondary.locator(':scope > summary').click();
  }
  const panel = page.getByTestId('stops-panel');
  if (!(await panel.getAttribute('open'))) {
    await panel.locator('summary').first().click();
  }
  await expect(page.getByTestId('auto-pass-toggle').first()).toBeVisible();
}

async function updateSeq(page: Page): Promise<number> {
  return Number(await page.locator('main').getAttribute('data-update-seq'));
}

// Deterministic scripted hero: develop mana, otherwise pass. The same choice
// rule applies whether passes are submitted or automated. Each
// keyboard activation waits for the server response (data-update-seq increments per
// applied observation) before choosing again, so the action list is never
// mid-swap and the input count is exact. Keyboard play avoids repeated pointer
// scrolling/layout waits; pointer submission is covered by the release suite.
async function playUntilComparisonTurn(page: Page): Promise<number> {
  const actionButtons = page.getByTestId('action-option');
  const gameOverOverlay = page.getByText('Game Over', { exact: true });
  let inputs = 0;

  for (let i = 0; i < MAX_ACTIONS; i += 1) {
    await expect(actionButtons.first().or(gameOverOverlay)).toBeVisible({ timeout: 30_000 });
    await expect(gameOverOverlay).not.toBeVisible();
    await expect(page.locator('[data-testid="action-option"]:enabled').first().or(gameOverOverlay)).toBeVisible();
    await expect(gameOverOverlay).not.toBeVisible();
    const board = await page.getByTestId('game-board').textContent();
    if (board?.includes(`Turn ${COMPARISON_TURN} · Your turn · Precombat Main`)) return inputs;

    const seqBefore = await updateSeq(page);
    const labels = await actionButtons.allTextContents();
    let pick = labels.findIndex((label) => /Play (Island|Mountain)/.test(label));
    if (pick === -1) {
      pick = labels.findIndex((label) => label.includes('Pass priority'));
    }
    if (pick === -1) {
      pick = 0;
    }
    await actionButtons.nth(pick).focus();
    await page.keyboard.press('Enter');
    inputs += 1;

    await expect
      .poll(() => updateSeq(page), {
        timeout: 30_000,
        message: `no server response after input ${inputs}`,
      })
      .toBeGreaterThan(seqBefore);
  }

  throw new Error(`game did not reach turn ${COMPARISON_TURN} within ${MAX_ACTIONS} actions`);
}

test('stops cut inputs to the same turn boundary, log keeps narrating', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await page.getByTestId('opponent-select').selectOption('random');
  // Counterspell keeps priority available; tiny decks finish by exhaustion
  // independently of a random opponent's spell and mana choices.
  await selectDeck(page, 'hero', 'interactive');
  await selectDeck(page, 'villain', 'interactive');

  // ---- Run 1: no-stops baseline (auto-pass off = pre-stops behavior). ----
  await openStopsPanel(page);
  await setCheckbox(page, 'auto-pass-toggle', false);

  await page.getByRole('button', { name: 'New Game' }).first().click();
  await expect(page.getByTestId('game-board')).toBeVisible({ timeout: 15_000 });
  const baselineInputs = await playUntilComparisonTurn(page);

  // ---- Run 2: same seed, stops on main phases only. ----
  await openStopsPanel(page);
  await setCheckbox(page, 'auto-pass-toggle', true);
  await setCheckbox(page, 'stop-my-main1', true);
  await setCheckbox(page, 'stop-my-main2', true);
  await setCheckbox(page, 'stop-opponent-end_step', false);

  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: 'Rematch', exact: true }).first().click();
  // The unfinished baseline remains mounted until the replacement is accepted.
  await expect(page.getByRole('region', { name: 'Play session' })).toContainText('Game 2');
  await expect(page.getByTestId('action-option').first()).toBeVisible({ timeout: 15_000 });
  const stopsInputs = await playUntilComparisonTurn(page);

  // Held Counterspells expose priority at every step of both turns.
  // Main-phase stops remove most of those otherwise deliberate inputs.
  console.log(
    `inputs to turn ${COMPARISON_TURN}, same seed: baseline=${baselineInputs} stops=${stopsInputs}`,
  );
  expect(baselineInputs).toBeGreaterThan(stopsInputs * 2);
  expect(stopsInputs).toBeGreaterThan(0);

  // The narrative survives the fast-forward: villain actions inside skipped
  // stretches are logged, and the auto-passed windows are called out.
  const logTexts = await page.getByTestId('log-entry').allTextContents();
  expect(logTexts.some((text) => text.trimStart().startsWith('Villain'))).toBe(true);
  expect(logTexts.some((text) => /Auto-passed \d+ priority window/.test(text))).toBe(true);
});

test('F6 passes the turn from the keyboard', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await page.getByTestId('opponent-select').selectOption('random');
  await selectDeck(page, 'hero', 'interactive');
  await selectDeck(page, 'villain', 'interactive');

  await page.getByRole('button', { name: 'New Game' }).first().click();
  const board = page.getByTestId('game-board');
  await expect(board).toBeVisible({ timeout: 15_000 });

  // Default stops: the game opens at my main1 on turn 1.
  await expect(board).toContainText('Turn 1');
  await expect(page.getByTestId('action-option').first()).toBeVisible({ timeout: 15_000 });

  await page.keyboard.press('F6');

  // The rest of turn 1 (including the my-main2 stop) is yielded server-side;
  // the next surfaced decision is on a later turn.
  await expect(board).toContainText(/Turn (?!1 )\d+/, { timeout: 30_000 });
  const logTexts = await page.getByTestId('log-entry').allTextContents();
  expect(
    logTexts.some((text) => /Hero\s*Pass turn/.test(text.replace(/\s+/g, ' '))),
  ).toBe(true);
});
