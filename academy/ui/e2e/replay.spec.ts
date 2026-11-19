import { expect, test } from '@playwright/test';

// The replay route once had a frozen-DOM bug (a legacy component reading a
// runes store): the recording list, board and position counter never updated
// after the initial render. Runs after play.spec.ts (single
// worker, alphabetical order), which plays a game to completion so at least
// one trace exists.

test('replay lists recordings and steps through positions on the board', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto('/replay');

  // The recording list is fetched after mount; a frozen DOM would show the
  // initial empty state forever. The newest readable recording opens on the
  // table (a Watch needs its participant's credential).
  const links = page.getByTestId('recording-link');
  await expect(links.first()).toBeVisible({ timeout: 15_000 });
  const hrefs = await links.evaluateAll(anchors => anchors.map(anchor => anchor.getAttribute('href')!));
  const board = page.getByTestId('game-board');
  for (const href of hrefs.slice(0, 10)) {
    await page.goto(href);
    await expect(board.or(page.getByRole('alert'))).toBeVisible({ timeout: 15_000 });
    if (await board.isVisible()) break;
  }
  await expect(board).toBeVisible();

  const positionCounter = page.getByTestId('position-counter');
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
