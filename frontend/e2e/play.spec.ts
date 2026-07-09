import { expect, test, type Page } from '@playwright/test';

// Regression test for the frozen-DOM bug: the play page was compiled in Svelte
// legacy mode, so every read of the runes-based gameStore was untracked and the
// UI never re-rendered after the initial paint. These assertions are all about
// observing DOM *mutation* — a frozen page fails them even though the
// WebSocket protocol underneath works fine.

const DECISION_POINTS = 10;

// 1x1 transparent PNG so card art requests resolve without hitting Scryfall
// (offline CI) and without logging resource-load console errors.
const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==',
  'base64',
);

function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() !== 'error') {
      return;
    }
    const text = message.text();
    if (text.includes('favicon')) {
      return; // harmless dev-server 404
    }
    errors.push(text);
  });
  page.on('pageerror', (error) => {
    errors.push(`pageerror: ${error.message}`);
  });
  return errors;
}

test('play page reacts to the full game loop', async ({ page }) => {
  const consoleErrors = collectConsoleErrors(page);

  await page.route('https://api.scryfall.com/**', (route) =>
    route.fulfill({ contentType: 'image/png', body: TINY_PNG }),
  );

  await page.goto('/');

  // 1. Connection badge must leave its initial "connecting"/"disconnected"
  // state. With the frozen DOM it stayed "disconnected" forever.
  const badge = page.getByTestId('connection-badge');
  await expect(badge).toHaveText('connected', { timeout: 15_000 });

  // 2. Play against the random villain (fast, no search/checkpoint needed).
  await page.locator('select').first().selectOption('random');

  // 3. New Game must actually render a board (observation applied to DOM).
  await page.getByRole('button', { name: 'New Game' }).first().click();
  const board = page.getByTestId('game-board');
  await expect(board).toBeVisible({ timeout: 15_000 });
  await expect(board).toContainText('Battlefield');
  await expect(board).toContainText('Turn');

  const logEntries = page.getByTestId('log-entry');
  const actionButtons = page.getByTestId('action-option');
  const gameOverOverlay = page.getByText('Game Over', { exact: true });

  const boardSnapshots = new Set<string>();
  let actionsTaken = 0;

  while (actionsTaken < DECISION_POINTS) {
    // Wait for a decision point: either legal actions are offered or the game
    // ended (vs random the hero occasionally dies fast — start another game).
    await expect(actionButtons.first().or(gameOverOverlay)).toBeVisible({ timeout: 30_000 });

    if (await gameOverOverlay.isVisible()) {
      await page.getByRole('button', { name: 'Play Again' }).click();
      await expect(board).toBeVisible({ timeout: 15_000 });
      continue;
    }

    const logCountBefore = await logEntries.count();
    boardSnapshots.add(await board.innerHTML());

    const first = actionButtons.first();
    await expect(first).toBeEnabled();
    await first.click();
    actionsTaken += 1;

    // THE regression assertion: every action must visibly mutate the DOM.
    // The store appends a `Hero: ...` log line for each action, so the log
    // must grow; a frozen DOM keeps the count constant forever.
    await expect
      .poll(async () => logEntries.count(), {
        timeout: 30_000,
        message: `log did not grow after action ${actionsTaken} (frozen DOM?)`,
      })
      .toBeGreaterThan(logCountBefore);
  }

  // The board itself must have re-rendered across the game: turn/phase header,
  // cards, and life totals change as the game advances. A frozen board yields
  // a single snapshot.
  boardSnapshots.add(await board.innerHTML());
  expect(
    boardSnapshots.size,
    'board DOM never changed across actions (frozen DOM?)',
  ).toBeGreaterThan(2);

  expect(consoleErrors, `console errors: ${consoleErrors.join('\n')}`).toEqual([]);
});
