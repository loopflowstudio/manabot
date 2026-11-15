import { expect, test } from '@playwright/test';

// The replay route had the same latent frozen-DOM bug as the play page (legacy
// component reading a runes store): the trace list, board, and position counter
// never updated after the initial render. Runs after play.spec.ts (single
// worker, alphabetical order), which plays a game to completion so at least
// one trace exists.

test('replay page lists traces and steps through positions', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto('/replay');

  // The trace picker is fetched after mount; a frozen DOM would show the
  // initial "No traces yet." state forever.
  const traceSelect = page.getByTestId('trace-select');
  await expect(traceSelect).toBeVisible({ timeout: 15_000 });

  // Index 0 is the disabled placeholder; pick the newest real trace.
  await traceSelect.selectOption({ index: 1 });
  const board = page.getByTestId('game-board');
  await expect(board).toBeVisible({ timeout: 15_000 });

  const positionCounter = page.getByText(/^Position \d+ \/ \d+$/);
  await expect(positionCounter).toHaveText(/Position \d+ \/ \d+/);

  // Stepping must move the position counter and re-render the board. Adjacent
  // positions occasionally share an identical board (e.g. back-to-back priority
  // passes), so require the counter to track every click and the board to
  // change at least once across a handful of steps.
  const counterText = (await positionCounter.textContent()) ?? '';
  const totalPositions = Number(/\/ (\d+)$/.exec(counterText)?.[1] ?? '0');
  const initialPosition = Number(/Position (\d+)/.exec(counterText)?.[1] ?? '1');
  expect(totalPositions).toBeGreaterThan(1);
  const lastStep = Math.min(initialPosition + 5, totalPositions);

  const boardSnapshots = new Set<string>([await board.innerHTML()]);
  const next = page.getByRole('button', { name: 'Next', exact: true });
  for (let step = initialPosition + 1; step <= lastStep; step += 1) {
    await next.click();
    await expect(positionCounter).toHaveText(new RegExp(`Position ${step} / \\d+`));
    boardSnapshots.add(await board.innerHTML());
  }
  expect(
    boardSnapshots.size,
    'replay board never changed across positions (frozen DOM?)',
  ).toBeGreaterThan(1);

  expect(pageErrors, `page errors: ${pageErrors.join('\n')}`).toEqual([]);
});
