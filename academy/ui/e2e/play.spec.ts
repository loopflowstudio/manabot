import { expect, test, type Page } from '@playwright/test';

// Regression test for the frozen-DOM bug: the play page was compiled in Svelte
// legacy mode, so every read of the runes-based gameStore was untracked and the
// UI never re-rendered after the initial paint. These assertions are all about
// observing DOM *mutation* — a frozen page fails them even though the
// WebSocket protocol underneath works fine.

const DECISION_POINTS = 10;
// Hard cap so a pathological game cannot spin forever; scripted random games
// in tests/gui/test_play_modes.py finish well under this.
const MAX_ACTIONS = 1000;

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

  await page.goto('/');

  // 1. Connection badge must leave its initial "connecting"/"disconnected"
  // state. With the frozen DOM it stayed "disconnected" forever.
  const badge = page.getByTestId('connection-badge');
  await expect(badge).toHaveText('connected', { timeout: 15_000 });

  // 2. Play against the random villain (fast, no search/checkpoint needed).
  await page.getByTestId('opponent-select').selectOption('random');

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
  let gameOver = false;

  while (actionsTaken < MAX_ACTIONS) {
    // Wait for a decision point: either legal actions are offered or the game
    // ended. Play the game to completion so a trace lands for the replay
    // suite, but require at least DECISION_POINTS decisions overall (vs
    // random the hero occasionally dies fast — start another game).
    await expect(actionButtons.first().or(gameOverOverlay)).toBeVisible({ timeout: 30_000 });

    if (await gameOverOverlay.isVisible()) {
      if (actionsTaken >= DECISION_POINTS) {
        gameOver = true;
        break;
      }
      await page.getByRole('button', { name: 'Play Again' }).click();
      await expect(board).toBeVisible({ timeout: 15_000 });
      continue;
    }

    const logCountBefore = await logEntries.count();
    boardSnapshots.add(await board.innerHTML());

    // Pick a random legal action, like the scripted WebSocket tests do. The
    // action list is swapped out whenever a server response lands, so the
    // sampled button can vanish between count() and click() — retry with a
    // fresh sample.
    let clicked = false;
    for (let attempt = 0; attempt < 5 && !clicked; attempt += 1) {
      const count = await actionButtons.count();
      if (count === 0) {
        break; // response in flight or game over — re-enter the outer wait
      }
      const choice = actionButtons.nth(Math.floor(Math.random() * count));
      try {
        await choice.click({ timeout: 2_000 });
        clicked = true;
      } catch {
        // list changed under us; resample
      }
    }
    if (!clicked) {
      continue;
    }
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

  expect(gameOver, `game did not finish within ${MAX_ACTIONS} actions`).toBe(true);
  expect(actionsTaken).toBeGreaterThanOrEqual(DECISION_POINTS);

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
