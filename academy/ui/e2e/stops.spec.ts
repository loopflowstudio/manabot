import { expect, test, type Page } from '@playwright/test';

// Priority stops e2e: the same deterministic game (server seed defaults to 0,
// random villain seeded, scripted deterministic hero) is played twice —
// first with auto-pass disabled (the old surface-every-window behavior),
// then with stops on main phases only. The stops run must need dramatically
// fewer clicks to reach game over, while the log keeps narrating what
// happened inside the fast-forwarded stretches.

const MAX_ACTIONS = 1200;

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

async function updateSeq(page: Page): Promise<number> {
  return Number(await page.locator('main').getAttribute('data-update-seq'));
}

// Deterministic scripted hero: develop mana, otherwise pass. Identical
// engine trajectory whether the passes are clicked or auto-passed. Each
// click waits for the server response (data-update-seq increments per
// applied observation) before choosing again, so the action list is never
// mid-swap and the click count is exact.
async function playUntilGameOver(page: Page): Promise<number> {
  const actionButtons = page.getByTestId('action-option');
  const gameOverOverlay = page.getByText('Game Over', { exact: true });
  let clicks = 0;

  for (let i = 0; i < MAX_ACTIONS; i += 1) {
    await expect(actionButtons.first().or(gameOverOverlay)).toBeVisible({ timeout: 30_000 });
    if (await gameOverOverlay.isVisible()) {
      return clicks;
    }

    const seqBefore = await updateSeq(page);
    const labels = await actionButtons.allTextContents();
    let pick = labels.findIndex((label) => /Play (Island|Mountain)/.test(label));
    if (pick === -1) {
      pick = labels.findIndex((label) => label.includes('Pass priority'));
    }
    if (pick === -1) {
      pick = 0;
    }
    await actionButtons.nth(pick).click();
    clicks += 1;

    await expect
      .poll(() => updateSeq(page), {
        timeout: 30_000,
        message: `no server response after click ${clicks}`,
      })
      .toBeGreaterThan(seqBefore);
  }

  throw new Error(`game did not finish within ${MAX_ACTIONS} actions`);
}

test('stops cut the clicks needed to finish a game, log keeps narrating', async ({ page }) => {
  test.setTimeout(300_000);

  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await page.getByTestId('opponent-select').selectOption('random');
  // This spec's click-count contract was written for the INTERACTIVE_DECK
  // mirror (hero holds Counterspell/Bolt, so the baseline surfaces priority
  // at every step); pin the decks rather than inherit the UR/GW default.
  await page.getByTestId('deck-select-hero').selectOption('interactive');
  await page.getByTestId('deck-select-villain').selectOption('interactive');

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
  expect(logTexts.some((text) => text.trimStart().startsWith('Villain'))).toBe(true);
  expect(logTexts.some((text) => /Auto-passed \d+ priority window/.test(text))).toBe(true);
});

test('F6 passes the turn from the keyboard', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await page.getByTestId('opponent-select').selectOption('random');
  await page.getByTestId('deck-select-hero').selectOption('interactive');
  await page.getByTestId('deck-select-villain').selectOption('interactive');

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
