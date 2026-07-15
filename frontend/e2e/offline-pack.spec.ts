import { expect, test, type Page } from '@playwright/test';

const MAX_ACTIONS = 1500;
const PUBLIC_PROTOCOLS = new Set(['http:', 'https:', 'ws:', 'wss:']);
const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);

function isPublicRequest(rawUrl: string): boolean {
  const url = new URL(rawUrl);
  return PUBLIC_PROTOCOLS.has(url.protocol) && !LOOPBACK_HOSTS.has(url.hostname);
}

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.text().includes('favicon')) {
      errors.push(message.text());
    }
  });
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  return errors;
}

async function updateSeq(page: Page): Promise<number> {
  return Number(await page.locator('main').getAttribute('data-update-seq'));
}

async function finishGame(page: Page): Promise<void> {
  const actionButtons = page.getByTestId('action-option');
  const gameOver = page.getByText('Game Over', { exact: true });

  for (let step = 0; step < MAX_ACTIONS; step += 1) {
    await expect(actionButtons.first().or(gameOver)).toBeVisible({ timeout: 30_000 });
    if (await gameOver.isVisible()) {
      return;
    }

    const before = await updateSeq(page);
    const labels = await actionButtons.allTextContents();
    let choice = labels.findIndex((label) => label.startsWith('Cast '));
    if (choice === -1) {
      choice = labels.findIndex((label) => label.startsWith('Play '));
    }
    if (choice === -1) {
      choice = Math.floor(Math.random() * labels.length);
    }
    try {
      await actionButtons.nth(choice).click({ timeout: 2_000 });
    } catch {
      continue;
    }
    await expect
      .poll(() => updateSeq(page), { timeout: 30_000 })
      .toBeGreaterThan(before);
  }
  throw new Error(`offline pack game did not finish within ${MAX_ACTIONS} actions`);
}

test('fresh-cache play, reload, and replay need no public network', async ({ page }) => {
  test.setTimeout(300_000);
  const publicRequests: string[] = [];
  const errors = collectErrors(page);

  await page.route('**/*', async (route) => {
    const url = route.request().url();
    if (isPublicRequest(url)) {
      publicRequests.push(url);
      await route.abort('internetdisconnected');
      return;
    }
    await route.continue();
  });
  await page.routeWebSocket(
    (url) => isPublicRequest(url.toString()),
    async (socket) => {
      publicRequests.push(socket.url());
      await socket.close({ code: 1008, reason: 'offline verification' });
    },
  );
  page.on('websocket', (socket) => {
    if (isPublicRequest(socket.url())) {
      publicRequests.push(socket.url());
    }
  });

  const startedAt = Date.now();
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await page.getByTestId('opponent-select').selectOption('random');
  await page.getByRole('button', { name: 'New Game' }).first().click();

  const board = page.getByTestId('game-board');
  await expect(board).toBeVisible({ timeout: 15_000 });
  expect(Date.now() - startedAt).toBeLessThan(60_000);
  await expect(page.getByTestId('deck-names')).toHaveText('UR Lessons vs GW Allies');

  const initialTreatments = board.getByTestId('card-treatment');
  const initialTreatmentCount = await initialTreatments.count();
  expect(initialTreatmentCount).toBeGreaterThan(0);
  await expect(board.locator('[data-asset-source="pack"]')).toHaveCount(
    initialTreatmentCount,
  );
  await expect(board.locator('[data-asset-source="fallback"]')).toHaveCount(0);

  await page.reload();
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await expect(board).toBeVisible({ timeout: 15_000 });
  await expect(board.getByTestId('card-treatment').first()).toHaveAttribute(
    'data-asset-source',
    'pack',
  );

  await finishGame(page);
  await page.goto('/replay');
  const traceButton = page.locator('button', { hasText: 'Winner:' }).first();
  await expect(traceButton).toBeVisible({ timeout: 15_000 });
  await traceButton.click();
  await expect(board).toBeVisible({ timeout: 15_000 });
  await expect(board.getByTestId('card-treatment').first()).toHaveAttribute(
    'data-asset-source',
    'pack',
  );

  expect(publicRequests).toEqual([]);
  expect(errors, `console errors: ${errors.join('\n')}`).toEqual([]);
});
