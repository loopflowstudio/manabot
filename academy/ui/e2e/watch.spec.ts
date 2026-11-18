import { expect, test } from '@playwright/test';

test('watch reads choices and consequences on the table, then returns from Study', async ({ page }) => {
  await page.goto('/replay?watch=1');
  await page.getByRole('button', { name: 'Start watching' }).click();
  await expect(page.getByRole('button', {name: 'Continue', exact: true})).toBeVisible();
  await page.getByRole('button', {name: 'Continue', exact: true}).click();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 1 / 3');
  // The recorded choice: every offer is a row, the played one marked, evidence beside the prompt.
  const rows = page.getByTestId('action-option');
  await expect(rows.first()).toBeDisabled();
  await expect(page.locator('[data-testid="action-option"][data-chosen="true"]')).toHaveCount(1);
  await expect(page.getByTestId('recorded-decision-evidence')).toBeVisible();
  await page.getByRole('checkbox', {name: 'Show both hands'}).check();
  await expect(page.getByTestId('replay-perspective')).toContainText('Both hands · spectator view');
  await page.getByRole('button', {name: 'Next', exact: true}).click();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 2 / 3');
  await expect(page.getByTestId('replay-perspective')).toContainText('Recorded seat-');
  await expect(rows).toHaveCount(0);
  await page.getByRole('button', {name: 'Previous', exact: true}).click();
  await page.getByRole('button', {name: 'Try another line', exact: true}).click();
  await page.getByTestId('action-option').first().click();
  await expect(page.getByText('Your line is committed')).toBeVisible();
  await page.getByTestId('study-return').click();
  await expect(page.getByRole('button', {name: 'Try another line', exact: true})).toBeFocused();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 1 / 3');
  await expect(page.getByRole('checkbox', {name: 'Show both hands'})).not.toBeChecked();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByTestId('position-counter')).toHaveText('Position 2 / 3');
  await page.reload();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 2 / 3');
});

test('opens formatted arena JSON as one game', async ({ page }) => {
  const game = { match_id: 'recorded-match', decisions: [] };
  let imported: unknown;
  await page.route('**/api/watch/import', async route => {
    imported = route.request().postDataJSON();
    await route.fulfill({ status: 400, json: { detail: 'arena_import_invalid' } });
  });
  await page.goto('/games');
  await page.getByText('Open a recorded game file',{exact:true}).click();
  await expect(page.getByLabel('Open a game file')).toHaveAccessibleDescription('An exported training game or an evaluation trace');
  await page.getByLabel('Open a game file').setInputFiles({
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
  await page.getByRole('banner').getByRole('link', { name: 'Games', exact: true }).click();
  expect((await history).ok()).toBe(true);
  await expect(page.getByRole('heading', { name: 'Games', exact: true })).toBeAttached();
});
