import { expect, test, type Page } from '@playwright/test';

// Priority stops e2e: the same deterministic game (server seed defaults to 0,
// random villain seeded, scripted deterministic hero) is played twice —
// first with auto-pass disabled (the old surface-every-window behavior),
// then with stops on main phases only. The stops run must need dramatically
// fewer clicks to reach game over, while the log keeps narrating what
// happened inside the fast-forwarded stretches.

const MAX_ACTIONS = 1200;

const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==',
  'base64',
);

async function setCheckbox(page: Page, testId: string, checked: boolean): Promise<void> {
  const box = page.getByTestId(testId);
  if ((await box.isChecked()) !== checked) {
    await box.click();
  }
}

async function openStopsPanel(page: Page): Promise<void> {
  const panel = page.getByTestId('stops-panel');
  if (!(await panel.getAttribute('open'))) {
    await panel.locator('summary').first().click();
  }
  await expect(page.getByTestId('auto-pass-toggle').first()).toBeVisible();
}

// Deterministic scripted hero: develop mana, otherwise pass. Identical
// engine trajectory whether the passes are clicked or auto-passed.
async function playUntilGameOver(page: Page): Promise<number> {
  const actionButtons = page.getByTestId('action-option');
  const gameOverOverlay = page.getByText('Game Over', { exact: true });
  let clicks = 0;

  for (let i = 0; i < MAX_ACTIONS; i += 1) {
    await expect(actionButtons.first().or(gameOverOverlay)).toBeVisible({ timeout: 30_000 });
    if (await gameOverOverlay.isVisible()) {
      return clicks;
    }

    // The action list is replaced when a server response lands, so a sampled
    // button can vanish between locate and click — retry with a fresh pick.
    let clicked = false;
    for (let attempt = 0; attempt < 5 && !clicked; attempt += 1) {
      if (await gameOverOverlay.isVisible()) {
        return clicks;
      }
      const playLand = actionButtons.filter({ hasText: /Play (Island|Mountain)/ }).first();
      const passPriority = actionButtons.filter({ hasText: 'Pass priority' }).first();
      const choice = (await playLand.count()) > 0
        ? playLand
        : (await passPriority.count()) > 0
          ? passPriority
          : actionButtons.first();
      try {
        await choice.click({ timeout: 2_000 });
        clicked = true;
      } catch {
        // list changed under us; resample
      }
    }
    if (clicked) {
      clicks += 1;
    }
  }

  throw new Error(`game did not finish within ${MAX_ACTIONS} actions`);
}

test('stops cut the clicks needed to finish a game, log keeps narrating', async ({ page }) => {
  test.setTimeout(300_000);

  await page.route('https://api.scryfall.com/**', (route) =>
    route.fulfill({ contentType: 'image/png', body: TINY_PNG }),
  );

  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await page.locator('select').first().selectOption('random');

  // ---- Run 1: no-stops baseline (auto-pass off = pre-stops behavior). ----
  await openStopsPanel(page);
  await setCheckbox(page, 'auto-pass-toggle', false);

  await page.getByRole('button', { name: 'New Game' }).first().click();
  await expect(page.getByTestId('game-board')).toBeVisible({ timeout: 15_000 });
  const baselineClicks = await playUntilGameOver(page);

  // ---- Run 2: same seed, stops on main phases only. ----
  await openStopsPanel(page);
  await setCheckbox(page, 'auto-pass-toggle', true);
  await setCheckbox(page, 'stop-my-main1', true);
  await setCheckbox(page, 'stop-my-main2', true);
  await setCheckbox(page, 'stop-opponent-end_step', false);

  await page.getByRole('button', { name: 'Play Again' }).click();
  await expect(page.getByTestId('game-board')).toBeVisible({ timeout: 15_000 });
  const stopsClicks = await playUntilGameOver(page);

  // The hero holds instants (INTERACTIVE_DECK), so the baseline surfaces
  // priority at every step of both turns; main-phase stops must remove the
  // bulk of them. Identical seeds and a deterministic policy make the two
  // runs the same game.
  console.log(
    `clicks to finish, same seed: baseline=${baselineClicks} stops=${stopsClicks}`,
  );
  expect(baselineClicks).toBeGreaterThan(stopsClicks * 2);
  expect(stopsClicks).toBeGreaterThan(0);

  // The narrative survives the fast-forward: villain actions inside skipped
  // stretches are logged, and the auto-passed windows are called out.
  const logTexts = await page.getByTestId('log-entry').allTextContents();
  expect(logTexts.some((text) => text.startsWith('Villain: '))).toBe(true);
  expect(logTexts.some((text) => /Auto-passed \d+ priority window/.test(text))).toBe(true);
});
