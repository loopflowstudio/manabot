import { expect, test, type Page } from '@playwright/test';

test.use({ actionTimeout: 10_000 });

/** Setup with the matchup named, so first-visit defaults do not choose one. */
const SETUP = '/?hero=ur_lessons&villain=gw_allies';
const listing = (url: URL): boolean => url.pathname === '/api/traces';

/** The game on the board, as the page names it. */
async function gameOnBoard(page: Page): Promise<string> {
  await expect(page.locator('main')).not.toHaveAttribute('data-game', '');
  return (await page.locator('main').getAttribute('data-game'))!;
}

/** Make one choice (a pass when one is offered) and wait for the board to move. */
async function decide(page: Page): Promise<void> {
  const offers = page.getByTestId('action-option');
  await expect(page.locator('[data-testid="action-option"]:enabled').first()).toBeVisible({ timeout: 30_000 });
  const seq = await page.locator('main').getAttribute('data-update-seq');
  const pass = page.locator('[data-testid="action-option"][data-action-type="PRIORITY_PASS_PRIORITY"]');
  await (await pass.count() ? pass.first() : offers.first()).click();
  await expect(page.locator('main')).not.toHaveAttribute('data-update-seq', seq!);
}

/** Eight-card land decks end by native deck-out in a few turns of passing. */
async function playToEnd(page: Page): Promise<void> {
  for (let decision = 0; decision < 100; decision++) {
    await expect(page.locator('[data-testid="action-option"]:enabled').first().or(page.getByTestId('game-result'))).toBeVisible({ timeout: 30_000 });
    if (await page.getByTestId('game-result').count()) return;
    await decide(page);
  }
  throw new Error('Bounded native game did not complete');
}

async function openSettings(page: Page): Promise<void> {
  const menu = page.locator('details.game-menu');
  if (!await menu.evaluate((node: HTMLDetailsElement) => node.open)) await page.getByTestId('game-menu').click();
}

const historyRow = (page: Page, game: string) =>
  page.getByTestId('game-history-row').filter({ has: page.locator(`a[aria-label="Replay game ${game}"]`) });

test('an unfinished game stays in History, replays without a credential and resumes where it stood', async ({ page, browser }) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  const refused: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.status() === 403) refused.push(response.url()); });
  await page.goto(SETUP);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('passive');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await decide(page);
  await decide(page);
  const game = await gameOnBoard(page);
  // The address names the game on the board.
  expect(new URL(page.url()).searchParams.get('game')).toBe(game);
  const revision = await page.locator('main').getAttribute('data-frame-revision');

  // New game opens setup in place and ends nothing.
  await openSettings(page);
  await page.getByTestId('new-game').click();
  await expect(page.getByTestId('home')).toBeVisible();
  expect(new URL(page.url()).search).toBe('');
  await expect(page.getByTestId('opponent-select')).toHaveValue('passive');

  await page.getByRole('banner').getByRole('link', { name: 'History', exact: true }).click();
  const row = historyRow(page, game);
  await expect(row.getByTestId('game-history-status')).toHaveText('In progress');
  await expect(row.getByTestId('game-history-resume')).toHaveAttribute('href', `/?game=${encodeURIComponent(game)}`);
  // The status filter finds it among unfinished games and not among finished ones.
  await page.goto('/games?status=active');
  await expect(row).toHaveCount(1);
  await page.goto('/games?status=completed');
  await expect(page.getByRole('status').filter({ hasText: /on this page/ })).toBeVisible();
  await expect(row).toHaveCount(0);

  // Replay reads the recorded prefix of an unfinished game, in a browser that never played it.
  const visitor = await browser.newContext();
  try {
    const reader = await visitor.newPage();
    reader.on('response', response => { if (response.status() === 403) refused.push(response.url()); });
    await reader.goto(new URL(`/replay?trace=${encodeURIComponent(game)}`, page.url()).href);
    await expect(reader.getByTestId('game-board')).toBeVisible({ timeout: 30_000 });
    await expect(reader.getByTestId('position-counter')).toHaveText(/Position \d+ \/ \d+/);
    await expect(reader.getByRole('alert')).toHaveCount(0);
  } finally { await visitor.close(); }

  // Resume is a link: the same game, at the decision that was pending.
  await page.goto('/games?status=active');
  await row.getByTestId('game-history-resume').click();
  await expect(page.locator('[data-testid="action-option"]:enabled').first()).toBeVisible({ timeout: 30_000 });
  expect(await gameOnBoard(page)).toBe(game);
  await expect(page.locator('main')).toHaveAttribute('data-frame-revision', revision!);
  await decide(page);
  // Play alone returns to it, and so does a reload.
  await page.goto('/');
  expect(await gameOnBoard(page)).toBe(game);
  await page.reload();
  expect(await gameOnBoard(page)).toBe(game);
  await expect(page.locator('[data-testid="action-option"]:enabled').first()).toBeVisible({ timeout: 30_000 });
  expect(refused).toEqual([]);
  expect(errors).toEqual([]);
});

test('a finished game offers Rematch and Swap sides, and its address opens its Replay', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(SETUP);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('passive');
  await page.getByText('Import decklist files', { exact: true }).click();
  await page.getByLabel('Use imported decks for the next game').check();
  await page.getByLabel('Your deck source', { exact: true }).fill('8 Island');
  await page.getByLabel('Opponent deck source', { exact: true }).fill('8 Mountain');
  await page.getByRole('combobox', { name: 'Deal mode', exact: true }).selectOption('repeat');
  await page.getByLabel('Seed', { exact: true }).fill('18446744073709551615');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  const first = await gameOnBoard(page);
  await playToEnd(page);
  await page.screenshot({ path: testInfo.outputPath('result.png') });

  // Rematch plays the same setup again as a new game.
  await page.getByRole('button', { name: 'Rematch', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('main')).not.toHaveAttribute('data-game', first);
  const second = await gameOnBoard(page);
  await expect(page.getByTestId('game-result')).toHaveCount(0);
  await playToEnd(page);
  await page.getByRole('button', { name: 'Swap sides', exact: true }).click();
  await expect(page.locator('main')).not.toHaveAttribute('data-game', second);
  const third = await gameOnBoard(page);
  // The swapped game is the one a reload returns to.
  await page.reload();
  expect(await gameOnBoard(page)).toBe(third);
  await expect(page.locator('[data-testid="action-option"]:enabled').first()).toBeVisible({ timeout: 30_000 });

  await page.goto('/games');
  await expect(historyRow(page, first).getByTestId('game-history-resume')).toHaveCount(0);
  await expect(historyRow(page, first).getByTestId('game-history-status')).not.toHaveText('In progress');
  await expect(historyRow(page, third).getByTestId('game-history-resume')).toHaveCount(1);
  await page.screenshot({ path: testInfo.outputPath('history.png') });
  // A finished game's Play address opens its Replay.
  await page.goto(`/?game=${encodeURIComponent(first)}`);
  await expect(page).toHaveURL(new RegExp(`/replay\\?trace=${first}`));
  await expect(page.getByTestId('game-board')).toBeVisible({ timeout: 30_000 });
  // An unknown game opens setup, silently.
  await page.goto('/?game=no-such-game');
  await expect(page.getByTestId('home')).toBeVisible();
  await expect(page.getByTestId('resume-refused')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('History sends its filters and says when nothing matches', async ({ page }) => {
  const queries: URLSearchParams[] = [];
  await page.route(listing, route => {
    queries.push(new URL(route.request().url()).searchParams);
    return route.fulfill({ json: [] });
  });
  await page.goto('/games?status=active&pairing=human-bot&q=Passive');
  await expect(page.getByTestId('history-empty')).toHaveText('No games match these filters.');
  const sent = queries.at(-1)!;
  expect([sent.get('status'), sent.get('pairing'), sent.get('q')]).toEqual(['active', 'human-bot', 'Passive']);
  await expect(page.getByRole('combobox', { name: 'Status' })).toHaveValue('active');
  await expect(page.getByRole('combobox', { name: 'Status' }).locator('option')).toHaveText(['Any status', 'In progress', 'Completed', 'Stopped']);
  await expect(page.getByRole('combobox', { name: 'Players' }).locator('option')).toHaveText(['Any pairing', 'Human vs Bot', 'Bot vs Bot']);
  await page.getByRole('link', { name: 'Clear all', exact: true }).click();
  await expect(page.getByTestId('history-empty')).toHaveText('No games yet. Play one, or open a recorded game file above.');
  // No name, scope or record: History is the list.
  await expect(page.getByLabel('Display name')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Session results' })).toHaveCount(0);
});

test('long recorded identities fit History and preserve full filter values', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  const name = 'ci-fix-528/consumer/release-smoke/checkpoints/learner_transitions-8';
  const version = 'a'.repeat(64);
  await page.route(listing, route => route.fulfill({ json: [{ id: 'long-identity', timestamp: '2026-11-14T00:00:00Z',
    status: 'completed', winner: 1, end_reason: 'game_over', origin: 'automated_validation', num_events: 5,
    resumable: false, players: [
      { seat: 0, name: 'H'.repeat(60), kind: 'human', player_id: 'human-identity', deck: 'ur_lessons', version: null },
      { seat: 1, name, kind: 'bot', player_id: `checkpoint:${version}`, deck: 'gw_allies', version },
    ] }] }));
  await page.goto('/games');
  const row = page.getByTestId('game-history-row');
  await expect(row).toContainText('ci-fix-528/…/learner_transitions-8');
  await expect(row).toContainText('aaaaaaaaaa');
  await expect(row.locator('td').nth(3)).toContainText('ci-fix-528/…/learner_transitions-8 won');
  await expect(row.locator('td').nth(3)).not.toContainText('aaaaaaaaaa');
  const modelName = row.locator('.player-name .truncate').last();
  expect(await modelName.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(1280);
  expect(await row.locator('..').evaluate(el => el.scrollWidth)).toBeLessThanOrEqual(1280);
  await expect(row.locator('a[href^="/games?bot="]')).toHaveAttribute('href', `/games?bot=${encodeURIComponent(`checkpoint:${version}`)}`);
  await expect(row.locator(`[title="${name} · ${version}"]`)).toHaveCount(1);
  await expect(row.getByTestId('game-history-resume')).toHaveCount(0);
  await expect(row.getByTestId('game-history-replay')).toHaveAttribute('href', /^\/replay\?trace=long-identity/);
});
