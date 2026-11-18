import { selectDeck } from './deck-selection';
import { expect, test, type Page } from '@playwright/test';

// Priority stops e2e: the same deal (seed 0, seeded random villain,
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
      if (payload.type === 'new_game') payload.config.seed = 0;
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

/** On the table, stops and the session live in the top bar's Menu. */
async function openMenu(page: Page): Promise<void> {
  const menu = page.locator('details.table-menu');
  if (await menu.count() && !await menu.evaluate((node: HTMLDetailsElement) => node.open)) {
    await page.getByTestId('table-menu').click();
  }
}

async function openStopsPanel(page: Page): Promise<void> {
  await openMenu(page);
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
  const gameOverOverlay = page.getByTestId('game-result');
  const table = page.getByTestId('table');
  let inputs = 0;

  for (let i = 0; i < MAX_ACTIONS; i += 1) {
    await expect(actionButtons.first().or(gameOverOverlay)).toBeVisible({ timeout: 30_000 });
    await expect(gameOverOverlay).not.toBeVisible();
    await expect(page.locator('[data-testid="action-option"]:enabled').first().or(gameOverOverlay)).toBeVisible();
    await expect(gameOverOverlay).not.toBeVisible();
    if (await table.getAttribute('data-turn') === String(COMPARISON_TURN)
      && await table.getAttribute('data-turn-owner') === 'you'
      && (await table.getAttribute('data-step'))?.startsWith('PRECOMBAT_MAIN')) return inputs;

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
  // This spec's input-count contract was written for the INTERACTIVE_DECK
  // mirror (hero holds Counterspell/Bolt, so the baseline surfaces priority
  // at every step); pin the decks rather than inherit the UR/GW default.
  await selectDeck(page, 'hero', 'interactive');
  await selectDeck(page, 'villain', 'interactive');

  // ---- Run 1: no-stops baseline (auto-pass off = pre-stops behavior). ----
  await openStopsPanel(page);
  await setCheckbox(page, 'auto-pass-toggle', false);

  await page.getByRole('button', { name: 'Start game' }).first().click();
  await expect(page.getByTestId('table')).toBeVisible({ timeout: 15_000 });
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

  // The hero holds instants (INTERACTIVE_DECK), so the baseline surfaces
  // priority at every step of both turns; main-phase stops must remove the
  // bulk of them. The deal is fixed, though priority automation can change
  // the random opponent's sampling sequence.
  console.log(
    `inputs to turn ${COMPARISON_TURN}, same seed: baseline=${baselineInputs} stops=${stopsInputs}`,
  );
  expect(baselineInputs).toBeGreaterThan(stopsInputs * 2);
  expect(stopsInputs).toBeGreaterThan(0);

  // The narrative survives the fast-forward: the opponent's moves inside
  // skipped stretches are in the move list, and automated passes are marked
  // as automation, never as the person's moves; passes fold into quiet runs.
  await expect(page.locator('[data-testid=log-entry][data-actor=opponent]').first()).toBeAttached();
  await expect(page.locator('[data-testid=pass-run] [data-testid=log-entry][data-actor=auto]').first()).toBeAttached();
});

test('F6 passes the turn from the keyboard', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await page.getByTestId('opponent-select').selectOption('random');
  await selectDeck(page, 'hero', 'interactive');
  await selectDeck(page, 'villain', 'interactive');

  await page.getByRole('button', { name: 'Start game' }).first().click();
  const table = page.getByTestId('table');
  await expect(table).toBeVisible({ timeout: 15_000 });

  // Default stops: the game opens at my main1 on turn 1.
  await expect(table).toHaveAttribute('data-turn', '1');
  await expect(page.getByTestId('action-option').first()).toBeVisible({ timeout: 15_000 });

  await page.keyboard.press('F6');

  // The rest of turn 1 (including the my-main2 stop) is yielded server-side;
  // the next surfaced decision is on a later turn.
  await expect(table).not.toHaveAttribute('data-turn', '1', { timeout: 30_000 });
  await expect(page.locator('[data-testid=log-entry][data-actor=you]').filter({ hasText: 'Pass turn' })).toHaveCount(1);
});
