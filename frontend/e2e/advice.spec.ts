import { expect, test, type Page } from '@playwright/test';

// First unified belief-input and strategy-comparison surface. The live page
// renders the fixture's pinned decision beside the ActionPanel; the replay
// page renders the same component in study mode beside the board. Both fetch
// through the same POST /api/advice seam. This spec runs before play.spec.ts
// alphabetically, so the study-mode test plays its own game to create a trace.

function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() !== 'error' || message.text().includes('favicon')) {
      return;
    }
    errors.push(message.text());
  });
  page.on('pageerror', (error) => {
    errors.push(`pageerror: ${error.message}`);
  });
  return errors;
}

async function startGame(page: Page): Promise<void> {
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', { timeout: 15_000 });
  await page.getByTestId('opponent-select').selectOption('random');
  await page.getByRole('button', { name: 'New Game' }).first().click();
  await expect(page.getByTestId('game-board')).toBeVisible({ timeout: 15_000 });
}

async function openAdvice(page: Page): Promise<void> {
  const advice = page.getByTestId('decision-advice');
  await expect(advice).toBeVisible({ timeout: 15_000 });
  if ((await advice.getAttribute('open')) === null) {
    await advice.locator('summary').click();
  }
  await expect(page.getByTestId('advice-beliefs')).toBeVisible({ timeout: 15_000 });
}

test('live page renders the shared decision-advice surface with four regions', async ({ page }) => {
  const consoleErrors = collectConsoleErrors(page);
  await startGame(page);
  await openAdvice(page);

  await expect(page.getByTestId('advice-facts')).toBeVisible();
  await expect(page.getByTestId('advice-advice')).toBeVisible();
  await expect(page.getByTestId('advice-deltas')).toBeVisible();
  await expect(page.getByTestId('advice-footer')).toBeVisible();
  expect(await page.getByTestId('advice-scenario-option').count()).toBe(2);
  expect(await page.getByTestId('advice-action-row').count()).toBe(2);
  // The shared action vocabulary is the frame's legacy offers.
  await expect(page.getByTestId('advice-advice')).toContainText('Play Mountain');
  await expect(page.getByTestId('advice-advice')).toContainText('Pass priority');
  // The pinned advisor identity is visible.
  await expect(page.getByTestId('advice-footer')).toContainText('flat-mc-search-v1');
  // Truthful conditional-vs-unconditional wording: the strategy distribution is
  // framed as conditional on the selected belief, not an unconditional verdict.
  await expect(page.getByTestId('advice-advice')).toContainText('Advice given this belief');
  await expect(page.getByTestId('advice-advice')).toContainText('conditional on the selected belief');
  await expect(page.getByTestId('advice-footer')).toContainText('conditional on the selected belief');
  // Viewer-safety: the opponent hand is hidden, never rendered as identities.
  await expect(page.getByTestId('advice-facts')).toContainText('(hidden)');

  expect(consoleErrors, `console errors: ${consoleErrors.join('\n')}`).toEqual([]);
});

test('switching belief scenario updates advice and deltas (pointer and keyboard)', async ({ page }) => {
  const consoleErrors = collectConsoleErrors(page);
  await startGame(page);
  await openAdvice(page);

  const advice = page.getByTestId('advice-advice');
  const firstRow = page.getByTestId('advice-action-row').first();
  const before = (await firstRow.textContent()) ?? '';

  // Pointer: select scenario B.
  await page.getByTestId('advice-scenario-option').filter({ hasText: 'Opponent holding interaction' }).click();
  const scenarioB = page.getByTestId('advice-scenario-option').filter({ hasText: 'Opponent holding interaction' }).locator('input');
  await expect(scenarioB).toBeChecked();
  const afterPointer = (await firstRow.textContent()) ?? '';
  expect(afterPointer, 'advice did not change on scenario switch').not.toEqual(before);

  // Keyboard: from scenario B, arrow up to scenario A (native radio-group nav).
  await scenarioB.focus();
  await page.keyboard.press('ArrowUp');
  const scenarioA = page.getByTestId('advice-scenario-option').filter({ hasText: 'Opponent curving out' }).locator('input');
  await expect(scenarioA).toBeChecked();
  const afterKeyboard = (await firstRow.textContent()) ?? '';
  expect(afterKeyboard, 'advice did not change on keyboard switch').not.toEqual(afterPointer);

  expect(consoleErrors, `console errors: ${consoleErrors.join('\n')}`).toEqual([]);
});

test('reduced motion and narrow viewport keep the advice surface legible', async ({ browser }) => {
  const context = await browser.newContext({
    reducedMotion: 'reduce',
    viewport: { width: 390, height: 844 },
  });
  const page = await context.newPage();
  const consoleErrors = collectConsoleErrors(page);
  try {
    await startGame(page);
    await openAdvice(page);
    const advice = page.getByTestId('decision-advice');
    await expect(advice).toHaveAttribute('data-reduced-motion', 'true');
    await expect(advice.getAttribute('data-mode')).resolves.toBe('live');
    // The component remains visible and legible under the mobile breakpoint.
    await expect(page.getByTestId('advice-action-row').first()).toBeVisible();
  } finally {
    await context.close();
  }
  expect(consoleErrors, `console errors: ${consoleErrors.join('\n')}`).toEqual([]);
});

test('POST /api/advice fails closed with a typed unavailable state on identity mismatch', async ({ page }) => {
  const consoleErrors = collectConsoleErrors(page);
  await page.goto('/');
  const meta = await page.evaluate(async () => {
    const response = await fetch('/api/advice');
    return response.json() as Promise<{
      address: string;
      identity: { source_replay_id: string; match_id: string; advisor_id: string; compute_id: string };
    }>;
  });
  const closed = await page.evaluate(async (meta) => {
    const response = await fetch('/api/advice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        address: meta.address,
        scenario_id: 'advice-scenario-a',
        identity: { ...meta.identity, advisor_id: 'wrong' },
      }),
    });
    return response.json() as Promise<{ status: string; reason: string | null; evidence: unknown }>;
  }, meta);
  expect(closed.status).toBe('unavailable');
  expect(closed.reason).toBe('identity_mismatch');
  expect(closed.evidence).toBeNull();
  expect(consoleErrors, `console errors: ${consoleErrors.join('\n')}`).toEqual([]);
});

test('replay page renders the same surface in study mode at the pinned decision', async ({ page }) => {
  const consoleErrors = collectConsoleErrors(page);
  await startGame(page);

  // Play the game to completion so a trace lands for the replay page. The
  // action list is swapped out whenever a server response lands, so a sampled
  // button can vanish between count() and click() — retry with a fresh sample,
  // mirroring the established play.spec.ts loop.
  const actionButtons = page.getByTestId('action-option');
  const gameOver = page.getByText('Game Over', { exact: true });
  for (let i = 0; i < 1000; i += 1) {
    await expect(actionButtons.first().or(gameOver)).toBeVisible({ timeout: 30_000 });
    if (await gameOver.isVisible()) {
      break;
    }
    let clicked = false;
    for (let attempt = 0; attempt < 5 && !clicked; attempt += 1) {
      const count = await actionButtons.count();
      if (count === 0) {
        break; // response in flight or game over — re-enter the outer wait
      }
      try {
        await actionButtons.nth(0).click({ timeout: 2_000 });
        clicked = true;
      } catch {
        // list changed under us; resample
      }
    }
    if (!clicked) {
      continue;
    }
  }
  await expect(gameOver).toBeVisible({ timeout: 30_000 });

  await page.goto('/replay');
  const traceSelect = page.getByTestId('trace-select');
  await expect(traceSelect).toBeVisible({ timeout: 15_000 });
  await traceSelect.selectOption({ index: 1 });
  await expect(page.getByTestId('game-board')).toBeVisible({ timeout: 15_000 });

  const advice = page.getByTestId('decision-advice');
  await expect(advice).toBeVisible({ timeout: 15_000 });
  await expect(advice).toHaveAttribute('data-mode', 'study');
  if ((await advice.getAttribute('open')) === null) {
    await advice.locator('summary').click();
  }
  await expect(page.getByTestId('advice-beliefs')).toBeVisible({ timeout: 15_000 });
  await expect(page.getByTestId('advice-advice')).toBeVisible();
  await expect(page.getByTestId('advice-deltas')).toBeVisible();
  // The same shared action vocabulary as live mode.
  await expect(page.getByTestId('advice-advice')).toContainText('Play Mountain');

  expect(consoleErrors, `console errors: ${consoleErrors.join('\n')}`).toEqual([]);
});
