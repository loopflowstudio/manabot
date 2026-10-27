import { expect, test } from '@playwright/test';

test('watch steps, pauses, reopens a decision, and retries without advancing the match', async ({ page }) => {
  await page.goto('/replay?watch=1');
  await page.getByRole('button', { name: 'Start watching' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Decision 1 of 1' })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('decision-advice')).toContainText('Recorded at the acting bot');
  await expect(page.getByRole('table')).toContainText('✓');
  await expect(page.getByRole('checkbox', { name: 'Show both hands' })).not.toBeChecked();
  await expect(page.getByText('Both hands · spectator view', {exact: true})).toHaveCount(0);
  await page.getByRole('checkbox', { name: 'Show both hands' }).check();
  await expect(page.getByText('Both hands · spectator view', {exact: true})).toBeVisible();
  await page.getByRole('button', { name: 'Step decision' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Decision 2 of 2' })).toBeVisible();
  await page.getByRole('button', { name: 'Previous decision' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Decision 1 of 2' })).toBeVisible();
  await page.getByRole('button', { name: 'Try another line' }).click();
  await expect(page.getByTestId('study-panel')).toBeVisible();
  await page.getByTestId('action-option').first().click();
  await expect(page.getByText('Your line is committed')).toBeVisible();
  await page.getByRole('button', { name: 'Return to score' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Decision 1 of 2' })).toBeVisible();
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Pause', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await page.reload();
  await expect(page.getByRole('status').filter({ hasText: 'Decision 1 of 2' })).toBeVisible();
});

test('opens formatted arena JSON as one game', async ({ page }) => {
  const game = { match_id: 'recorded-match', decisions: [] };
  let imported: unknown;
  await page.route('**/api/watch/import', async route => {
    imported = route.request().postDataJSON();
    await route.fulfill({ status: 400, json: { detail: 'arena_import_invalid' } });
  });
  await page.goto('/replay?watch=1');
  await expect(page.getByRole('button', { name: 'Start watching' })).toBeEnabled();
  await page.getByLabel('Open arena game').setInputFiles({
    name: 'game.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(game, null, 2)),
  });
  await expect(page.getByRole('alert')).toHaveText('arena import invalid');
  expect(imported).toEqual(game);
});

test('loads replay history when navigating from Watch to Replay', async ({ page }) => {
  await page.route('**/api/traces', route => route.fulfill({ json: [] }));
  await page.goto('/replay?watch=1');
  await expect(page.getByRole('button', { name: 'Start watching' })).toBeEnabled();
  const history = page.waitForResponse(response => new URL(response.url()).pathname === '/api/traces');
  await page.getByRole('banner').getByRole('link', { name: 'Replay', exact: true }).click();
  expect((await history).ok()).toBe(true);
  await expect(page.getByRole('heading', { name: 'Replay', exact: true })).toBeAttached();
});
