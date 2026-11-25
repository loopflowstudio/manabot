import { expect, test } from '@playwright/test';

test.use({ actionTimeout: 10_000 });

test('swapping sides mid-game flips seat, hand, and deck names, keeps move log, and accepts the next Command', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));

  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('passive');
  await page.getByRole('combobox', { name: 'Deal mode', exact: true }).selectOption('repeat');
  await page.getByLabel('Seed', { exact: true }).fill('11');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();

  // Keep hand so the move list records our first deliberate decision.
  const keep = page.getByTestId('action-option').filter({ hasText: 'Keep hand' });
  await expect(keep).toBeEnabled({ timeout: 15_000 });
  const revAfterStart = await page.locator('main').getAttribute('data-frame-revision');
  await keep.click();
  await expect(page.locator('main')).not.toHaveAttribute('data-frame-revision', revAfterStart!, { timeout: 15_000 });
  await expect(page.getByTestId('action-option').first()).toBeEnabled({ timeout: 15_000 });

  // Verify we start in seat 0 with UR Lessons under You and Passive · scripted as opponent.
  const nearPlate = page.locator('[data-testid="player-plate"][data-plate-side="you"]');
  await expect(nearPlate).toHaveAttribute('data-player-index', '0');
  await expect(page.getByTestId('deck-names')).toContainText('You · UR Lessons');
  await expect(page.getByTestId('deck-names')).toContainText('Passive');
  await expect(page.getByTestId('deck-names')).toContainText('GW Allies');

  // Make one deliberate move on turn 1 so the move list has an earlier move to retain.
  const revBeforeMove = await page.locator('main').getAttribute('data-frame-revision');
  await page.getByTestId('action-option').first().click();
  await expect(page.locator('main')).not.toHaveAttribute('data-frame-revision', revBeforeMove!, { timeout: 15_000 });
  await expect(page.getByTestId('action-option').first()).toBeEnabled({ timeout: 15_000 });
  const logBeforeSwap = await page.getByTestId('game-log').innerText();
  expect(logBeforeSwap.trim().length).toBeGreaterThan(0);

  // Capture seat 0's hand before swapping.
  const handBefore = await page.getByRole('region', { name: /Your hand/ }).innerText();

  // Open Settings → Game, verify This game / Next game grouping and Swap play/draw rename, then press Swap sides.
  await page.getByTestId('game-menu').click();
  const thisGame = page.getByRole('region', { name: 'This game', exact: true });
  await expect(thisGame).toBeVisible();
  const nextGame = page.getByRole('region', { name: 'Next game', exact: true });
  await expect(nextGame.getByRole('button', { name: 'Swap play/draw', exact: true })).toBeVisible();

  const revBeforeSwap = await page.locator('main').getAttribute('data-frame-revision');
  await thisGame.getByTestId('swap-sides').click();

  // Board flips to seat 1: GW Allies under You, UR Lessons under Passive, different hand,
  // live move list retains earlier moves and adds "Swapped sides".
  await expect(page.locator('main')).not.toHaveAttribute('data-frame-revision', revBeforeSwap!, { timeout: 15_000 });
  await expect(nearPlate).toHaveAttribute('data-player-index', '1');
  await expect(page.getByTestId('deck-names')).toContainText('You · GW Allies');
  await expect(page.getByTestId('deck-names')).toContainText('Passive');
  await expect(page.getByTestId('deck-names')).toContainText('UR Lessons');
  await expect(page.getByTestId('game-log')).toContainText('Swapped sides');
  const handAfter = await page.getByRole('region', { name: /Your hand/ }).innerText();
  expect(handAfter).not.toEqual(handBefore);

  // Next Command in the new seat is accepted.
  const revAfterSwap = await page.locator('main').getAttribute('data-frame-revision');
  await expect(page.getByTestId('action-option').first()).toBeEnabled({ timeout: 15_000 });
  await page.getByTestId('action-option').first().click();
  await expect(page.locator('main')).not.toHaveAttribute('data-frame-revision', revAfterSwap!, { timeout: 15_000 });

  expect(errors).toEqual([]);
});
