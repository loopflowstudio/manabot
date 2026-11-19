import { expect, test } from '@playwright/test';

test.skip(!process.env.ACADEMY_WHEEL_COMMAND, 'Installed-wheel proof only');

test('installed Academy plays and opens its recorded decisions', async ({ page }) => {
  // Keep one real image request pending across navigation so document reloads
  // deterministically fail the request-error assertion instead of racing the cache.
  let releaseArt!: () => void;
  const artGate = new Promise<void>(resolve => { releaseArt = resolve; });
  let artRequested = false;
  let artFinished = false;
  let heldArtUrl = '';
  page.on('close', () => releaseArt());
  page.on('requestfinished', request => { if (request.url() === heldArtUrl) artFinished = true; });
  await page.route('**/api/card-art/art/*.jpg', async route => {
    heldArtUrl = route.request().url();
    artRequested = true;
    await artGate;
    await route.continue();
  }, { times: 1 });
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  page.on('requestfailed', request => {
    // The board aborts a payment preview its next frame supersedes; that read
    // commits nothing, so only its abort is expected here.
    const text = request.failure()?.errorText;
    if (text === 'net::ERR_ABORTED' && new URL(request.url()).pathname.endsWith('/payment-preview')) return;
    errors.push(`${request.url()}: ${text}`);
  });
  const catalog = await (await page.request.get('/api/opponents')).json();
  expect(catalog.extension).toBeNull();
  expect(catalog.watch_players).toEqual(['random', 'search']);
  const outcomes: { status?: string }[] = [];
  page.on('websocket', socket => socket.on('framereceived', event => {
    const payload = JSON.parse(String(event.payload));
    if (payload.type === 'command_outcome') outcomes.push(payload);
  }));
  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await expect(page.getByTestId('opponent-select')).toHaveValue('search-64');
  await page.getByRole('button', { name: 'Start game', exact: true }).first().click();
  const action = page.getByTestId('action-option').first();
  await expect(action).toBeEnabled();
  await action.click();
  await expect.poll(() => outcomes.some(row => row.status === 'accepted'), { timeout: 90_000 }).toBe(true);
  // Use the app's navigation: hard document loads cancel in-flight Image requests.
  await expect.poll(() => artRequested).toBe(true);
  // The board fills the window; its top bar is the way back to the games list.
  await page.getByRole('link', { name: 'Academy · Games', exact: true }).click();
  releaseArt();
  await expect.poll(() => artFinished).toBe(true);
  await page.locator('a[aria-label^="Replay game"]').first().click();
  await expect(page.getByTestId('game-board')).toBeVisible();
  const counter = page.getByTestId('position-counter');
  await expect(counter).toBeVisible();
  const initialPosition = Number(/Position (\d+)/.exec(await counter.innerText())![1]);
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(counter).toHaveText(new RegExp(`Position ${initialPosition + 1} / \\d+`));
  await page.getByRole('link', { name: 'Academy · Games', exact: true }).click();
  await page.getByRole('banner').getByRole('link', { name: 'New game', exact: true }).click();
  await page.getByRole('combobox', { name: 'Seat 1 controller', exact: true }).selectOption('bot');
  await expect(page.getByRole('button', { name: 'Start watching' })).toBeEnabled();
  for (const seat of ['Seat 1', 'Seat 2']) {
    await expect(page.getByRole('combobox', { name: `${seat} bot`, exact: true }).locator('option')).toHaveText([
      'Search 16 · search', 'Search 64 · search', 'Search 256 · search', 'Random', 'Passive · scripted', 'Registered policy',
    ]);
  }
  await expect(page.getByText('No policies registered on this server. Supply a Python policy factory or a saved specification.').first()).toBeVisible();
  await expect(page.getByLabel('Open a game file')).toHaveCount(0);
  expect(errors).toEqual([]);
});
