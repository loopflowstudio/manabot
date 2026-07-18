import { expect, test } from '@playwright/test';

import {
  collectStudyConsoleErrors,
  createCompletedStudyTrace,
  openFirstStudyDecision,
} from './study-helpers';

test('normal Study runtime keeps evidence sealed and returns exactly', async ({ page }) => {
  test.setTimeout(180_000);
  const consoleErrors = collectStudyConsoleErrors(page);
  const traceId = await createCompletedStudyTrace(page);
  const decision = await openFirstStudyDecision(page, traceId);
  const recordedBoard = await page.getByTestId('game-board').innerHTML();

  await expect(page.getByTestId('study-return')).toBeVisible();
  await page.getByTestId('study-return').click();
  await expect(decision).toBeFocused();
  await expect.poll(() => page.getByTestId('game-board').innerHTML()).toBe(recordedBoard);

  await decision.click();
  const retry = page.getByTestId('study-retry-offer').filter({
    hasText: 'Pass priority',
  });
  await expect(retry).toBeVisible();
  await retry.click();
  const reveal = page.getByTestId('study-reveal');
  await expect(reveal).toBeFocused();
  await reveal.click();
  await expect(page.getByTestId('study-error')).toHaveText(
    'Study evidence is unavailable for this recording.',
  );
  await expect(page.getByTestId('study-plan-played')).toHaveCount(0);

  await page.getByTestId('study-return').click();
  await expect(decision).toBeFocused();
  await expect.poll(() => page.getByTestId('game-board').innerHTML()).toBe(recordedBoard);
  expect(consoleErrors.some((message) => message.includes('status of 409'))).toBe(true);
  expect(
    consoleErrors.filter((message) => !message.includes('status of 409')),
    `unexpected console errors: ${consoleErrors.join('\n')}`,
  ).toEqual([]);
});
