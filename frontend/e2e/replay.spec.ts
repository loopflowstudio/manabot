import { expect, test } from '@playwright/test';

// The replay route had the same latent frozen-DOM bug as the play page (legacy
// component reading a runes store): the trace list, board, and frame counter
// never updated after the initial render. Runs after play.spec.ts (single
// worker, alphabetical order), which plays a game to completion so at least
// one trace exists.

test('replay page lists traces and steps through frames', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto('/replay');

  // The trace list is fetched after mount; a frozen DOM would show the
  // initial "No traces yet." state forever.
  const traceButton = page.locator('button', { hasText: 'Winner:' }).first();
  await expect(traceButton).toBeVisible({ timeout: 15_000 });

  await traceButton.click();
  const board = page.getByTestId('game-board');
  await expect(board).toBeVisible({ timeout: 15_000 });

  const frameCounter = page.getByText(/^Frame \d+ \/ \d+$/);
  await expect(frameCounter).toHaveText(/Frame 1 \/ \d+/);

  // Stepping must move the frame counter and re-render the board. Adjacent
  // frames occasionally share an identical board (e.g. back-to-back priority
  // passes), so require the counter to track every click and the board to
  // change at least once across a handful of steps.
  const counterText = (await frameCounter.textContent()) ?? '';
  const totalFrames = Number(/\/ (\d+)$/.exec(counterText)?.[1] ?? '0');
  expect(totalFrames).toBeGreaterThan(1);
  const lastStep = Math.min(6, totalFrames);

  const boardSnapshots = new Set<string>([await board.innerHTML()]);
  const next = page.getByRole('button', { name: 'Next', exact: true });
  for (let step = 2; step <= lastStep; step += 1) {
    await next.click();
    await expect(frameCounter).toHaveText(new RegExp(`Frame ${step} / \\d+`));
    boardSnapshots.add(await board.innerHTML());
  }
  expect(
    boardSnapshots.size,
    'replay board never changed across frames (frozen DOM?)',
  ).toBeGreaterThan(1);

  expect(pageErrors, `page errors: ${pageErrors.join('\n')}`).toEqual([]);
});
