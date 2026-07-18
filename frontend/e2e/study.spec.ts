import { expect, test } from '@playwright/test';

import {
  collectStudyConsoleErrors,
  createCompletedStudyTrace,
  openFirstStudyDecision,
} from './study-helpers';

const FORBIDDEN_BEFORE_REVEAL = [
  'policy_mass',
  'search_value',
  'visits',
  'uncertainty',
  'analysis_budget',
  'provenance',
];

test('Retry before reveal compares bounded plans and returns to the Score', async ({
  page,
}) => {
  test.setTimeout(180_000);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const consoleErrors = collectStudyConsoleErrors(page);
  const traceId = await createCompletedStudyTrace(page);

  const decision = await openFirstStudyDecision(page, traceId);
  const recordedBoard = await page.getByTestId('game-board').innerHTML();
  const retryResponse = page.waitForResponse(
    (response) => response.url().includes('/retry') && response.request().method() === 'POST',
  );
  await page.getByTestId('study-retry-offer').filter({
    hasText: 'Pass priority',
  }).click();
  const retryPayload = await (await retryResponse).text();
  for (const field of FORBIDDEN_BEFORE_REVEAL) expect(retryPayload).not.toContain(field);

  const reveal = page.getByTestId('study-reveal');
  await expect(reveal).toBeFocused();
  const playedPreview = page.waitForResponse((response) => {
    if (!response.url().includes('/preview')) return false;
    return (response.request().postData() ?? '').includes('"played"');
  });
  await reveal.click();
  await playedPreview;

  const played = page.getByTestId('study-plan-played');
  const policy = page.getByTestId('study-plan-policy');
  const search = page.getByTestId('study-plan-search');
  await expect(played).toBeFocused();
  await expect(policy).toContainText('Policy probability');
  await expect(search).toContainText('Search value');
  await expect(search).toContainText('Visits');
  await expect(search).toContainText('Robustness');
  await expect(search).toContainText('Uncertainty');

  const policyPreview = page.waitForResponse((response) =>
    response.url().includes('/preview')
      && (response.request().postData() ?? '').includes('"policy"'),
  );
  await played.press('ArrowRight');
  await policyPreview;
  await expect(policy).toBeFocused();
  await expect(policy).toHaveAttribute('aria-checked', 'true');

  const searchPreview = page.waitForResponse((response) =>
    response.url().includes('/preview')
      && (response.request().postData() ?? '').includes('"search"'),
  );
  await policy.press('ArrowRight');
  await searchPreview;
  await expect(search).toBeFocused();
  await expect(search).toHaveAttribute('aria-checked', 'true');

  await search.press('Escape');
  await expect(decision).toBeFocused();
  await expect.poll(() => page.getByTestId('game-board').innerHTML()).toBe(recordedBoard);

  await page.setViewportSize({ width: 390, height: 844 });
  await decision.click();
  await page.getByTestId('study-retry-offer').filter({
    hasText: 'Pass priority',
  }).click();
  await page.getByTestId('study-reveal').click();
  await expect(page.getByTestId('study-plan-search')).toBeVisible();
  const planBoxes = await Promise.all(
    [played, policy, search].map((plan) => plan.boundingBox()),
  );
  expect(planBoxes.every((box) => (box?.height ?? 0) >= 44)).toBe(true);
  expect((planBoxes[1]?.y ?? 0)).toBeGreaterThan(planBoxes[0]?.y ?? 0);
  expect((planBoxes[2]?.y ?? 0)).toBeGreaterThan(planBoxes[1]?.y ?? 0);
  const hasHorizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(hasHorizontalOverflow).toBe(false);

  await page.getByTestId('study-return').click();
  await expect(decision).toBeFocused();
  await expect.poll(() => page.getByTestId('game-board').innerHTML()).toBe(recordedBoard);
  expect(consoleErrors, `console errors: ${consoleErrors.join('\n')}`).toEqual([]);
});
