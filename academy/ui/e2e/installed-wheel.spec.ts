import { expect, test } from '@playwright/test';

test.skip(!process.env.ACADEMY_WHEEL_COMMAND, 'Installed-wheel proof only');

test('installed Academy plays and opens its recorded decisions', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  page.on('requestfailed', request => errors.push(`${request.url()}: ${request.failure()?.errorText}`));
  const catalog = await (await page.request.get('/api/opponents')).json();
  expect(catalog.extension).toBeNull();
  expect(catalog.watch_players).toEqual(['random', 'search']);
  const outcomes: { status?: string }[] = [];
  page.on('websocket', socket => socket.on('framereceived', event => {
    const payload = JSON.parse(String(event.payload));
    if (payload.type === 'command_outcome') outcomes.push(payload);
  }));
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await expect(page.getByTestId('opponent-select')).toHaveValue('search-64');
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  const action = page.getByTestId('action-option').first();
  await expect(action).toBeEnabled();
  await action.click();
  await expect.poll(() => outcomes.some(row => row.status === 'accepted'), { timeout: 90_000 }).toBe(true);
  await page.goto('/replay');
  const select = page.getByTestId('trace-select');
  await expect(select).toBeVisible();
  await select.selectOption({ index: 1 });
  await expect(page.getByTestId('game-board')).toBeVisible();
  await expect(page.getByText(/^Frame 1 \/ \d+$/)).toBeVisible();
  await page.getByRole('button', { name: 'Next frame', exact: true }).click();
  await expect(page.getByText(/^Frame 2 \/ \d+$/)).toBeVisible();
  await page.goto('/replay?watch=1');
  await expect(page.getByRole('button', { name: 'Start watching' })).toBeEnabled();
  for (const seat of ['Seat 1', 'Seat 2']) {
    await expect(page.getByRole('combobox', { name: seat, exact: true }).locator('option')).toHaveText(['random', 'search']);
  }
  await expect(page.getByText('Trained bots need manabot installed on this server.')).toBeVisible();
  await expect(page.getByLabel('Open a game file')).toHaveCount(0);
  expect(errors).toEqual([]);
});
