import { selectDeck } from './deck-selection';
import { expect, test, type Page } from '@playwright/test';

test('unauthorized session history explains access and offers to leave the session filter', async ({ page }) => {
  await page.route('**/api/play-sessions/private-session', route => route.fulfill({ status: 403, json: { detail: 'Session not owned' } }));
  await page.route('**/api/traces?**', route => route.fulfill({ json: [] }));
  await page.goto('/games?session=private-session&q=Passive');
  const alert = page.getByRole('alert');
  await expect(alert).toContainText('browser and address (including port) where you played');
  await expect(alert).toContainText('credentials were cleared');
  await expect(alert.getByRole('button', { name: 'Try again' })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Session results' })).toHaveCount(0);
  await alert.getByRole('link', { name: 'Browse games outside this session' }).click();
  await expect(page).toHaveURL(/\/games\?q=Passive$/);
  await expect(alert).toHaveCount(0);
});

test('temporary session history failure retains retry and recovers authorized results', async ({ page }) => {
  let unavailable = true;
  await page.route('**/api/play-sessions/retry-session', route => route.fulfill(unavailable
    ? { status: 503, json: { detail: 'Unavailable' } }
    : { json: { results: [], totals: [{ origin: 'automated_validation', status: 'completed', count: 2 }], next_result_offset: null } }));
  await page.route('**/api/traces?**', route => route.fulfill({ json: [] }));
  await page.goto('/games?session=retry-session');
  await expect(page.getByRole('alert')).toContainText('Session history could not be loaded.');
  await expect(page.getByRole('alert')).not.toContainText('credentials');
  await expect(page.getByRole('region', { name: 'Session results' })).toHaveCount(0);
  unavailable = false;
  await page.getByRole('button', { name: 'Try again', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Session results' })).toContainText('2 completed');
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('session history pagination denial gives the same access guidance', async ({ page }) => {
  await page.route('**/api/play-sessions/paged-session*', route => route.fulfill(new URL(route.request().url()).searchParams.has('result_offset')
    ? { status: 403, json: { detail: 'Session not owned' } }
    : { json: { results: [], totals: [{ origin: 'human', status: 'completed', count: 3 }], next_result_offset: 100 } }));
  await page.route('**/api/traces?**', route => route.fulfill({ json: [] }));
  await page.goto('/games?session=paged-session');
  await page.getByRole('button', { name: 'More result configurations' }).click();
  await expect(page.getByRole('alert')).toContainText('Copying the link does not copy access');
  await expect(page.getByRole('region', { name: 'Session results' })).toContainText('human: 3 completed');
  await expect(page.getByRole('alert').getByRole('button', { name: 'Try again' })).toHaveCount(0);
});

test('session result pagination ignores responses from a prior history view', async ({ page }) => {
  let releasePage!: () => void;
  let requestedPage!: () => void;
  const released = new Promise<void>(resolve => { releasePage = resolve; });
  const requested = new Promise<void>(resolve => { requestedPage = resolve; });
  const result = (name: string) => ({
    opponent_name: name, opponent_id: 'control:passive', version: null,
    origin: 'automated_validation', human_seat: 0, human_deck: 'ur_lessons',
    opponent_deck: 'gw_allies', decklists: 'open', completed: 1, wins: 1,
    draws: 0, losses: 0, active: 0, stopped: 0, interrupted: 0,
    wall_elapsed_seconds: null, elapsed_samples: 0,
  });
  await page.route('**/api/traces?**', route => route.fulfill({ json: [] }));
  await page.route('**/api/play-sessions/session-a*', async route => {
    const more = new URL(route.request().url()).searchParams.has('result_offset');
    if (more) { requestedPage(); await released; }
    await route.fulfill({ json: {
      results: [result(more ? 'Stale opponent' : 'Current opponent')],
      totals: [], next_result_offset: more ? null : 100,
    } });
  });
  await page.goto('/games?session=session-a');
  await expect(page.getByRole('region', { name: 'Session results' })).toContainText('Current opponent');
  await page.getByRole('button', { name: 'More result configurations' }).click();
  await requested;
  await page.getByRole('link', { name: 'Remove Session filter' }).click();
  await expect(page.getByRole('region', { name: 'Session results' })).toHaveCount(0);
  await page.goBack();
  const results = page.getByRole('region', { name: 'Session results' });
  await expect(results).toContainText('Current opponent');
  const response = page.waitForResponse('**/api/play-sessions/session-a?result_offset=100');
  releasePage();
  await (await response).finished();
  // Let the browser apply the delivered fetch before inspecting the current view.
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  await expect(results).not.toContainText('Stale opponent');
  await expect(page.getByRole('button', { name: 'More result configurations' })).toBeEnabled();
});

test('shared history identifies both players and preserves private prefixes and notes', async ({ browser }) => {
  test.setTimeout(180_000);
  const contexts = await Promise.all([browser.newContext(), browser.newContext(), browser.newContext()]);
  const [alice, bob, visitor] = await Promise.all(contexts.map((context) => context.newPage()));
  const failures: string[] = [];
  for (const page of [alice, bob, visitor]) {
    page.on('pageerror', (error) => failures.push(error.message));
    await page.emulateMedia({ reducedMotion: 'reduce' });
    // Keep this history/privacy proof on one ordinary, reproducible deal.
    await page.routeWebSocket('**/ws/play', socket => {
      const server = socket.connectToServer();
      socket.onMessage(message => {
        if (typeof message !== 'string') { server.send(message); return; }
        const payload = JSON.parse(message);
        if (payload.type === 'new_game') payload.config.seed = 1255757736;
        server.send(JSON.stringify(payload));
      });
    });
  }
  const suffix = Date.now();
  const aliceName = `History Alice ${suffix}`;
  const bobName = `History Bob ${suffix}`;
  async function start(page: Page, name: string) {
    await page.goto('/games');
    await page.getByText('Your player name', { exact: true }).click();
    await expect(page.getByLabel('Display name')).not.toHaveValue('');
    await page.getByLabel('Display name').fill(name);
    await page.getByRole('button', { name: 'Save name', exact: true }).click();
    await expect(page.getByRole('status').filter({ hasText: 'Saved for your next game.' })).toBeVisible();
    await page.getByRole('link', { name: 'Play a game', exact: true }).click();
    await page.getByTestId('home-setup').click();
    await page.getByTestId('opponent-select').selectOption('passive');
    await selectDeck(page, 'hero', 'gw_allies');
    await selectDeck(page, 'villain', 'ur_lessons');
    await page.getByRole('button', { name: 'Start game', exact: true }).click();
    await expect(page.getByTestId('game-board')).toBeVisible();
  }
  try {
    await start(alice, aliceName);
    const over = alice.getByTestId('result-card');
    const actions = alice.getByTestId('action-option');
    for (let step = 0; step < 600 && !(await over.isVisible()); step++) {
      await expect(actions.first().or(over)).toBeVisible();
      if (await over.isVisible()) break;
      await expect(actions.first()).toBeEnabled();
      const before = Number(await alice.locator('main').getAttribute('data-update-seq'));
      const labels = await actions.allTextContents();
      let pick = labels.findIndex((label) => /^Play |^Cast |^Attack with/.test(label));
      if (pick < 0) pick = labels.findIndex((label) => /^Pass priority/.test(label));
      if (pick < 0) pick = 0;
      await actions.nth(pick).focus();
      await alice.keyboard.press('Enter');
      await expect.poll(async () => Number(await alice.locator('main').getAttribute('data-update-seq'))).toBeGreaterThan(before);
    }
    await expect(over).toBeVisible();
    await alice.getByTestId('table-menu').click();
    await alice.getByRole('textbox', { name: /How did this game feel/ }).fill('Automated history check: private note, not human feedback.');
    await alice.getByRole('button', { name: 'Save note', exact: true }).click();
    await expect(alice.getByRole('status').filter({ hasText: 'Feedback saved.' })).toBeVisible();
    await start(bob, bobName);
    await visitor.goto('/games');
    await expect(visitor.getByLabel('Display name')).not.toHaveValue('');
    await visitor.getByLabel('Search games').fill(String(suffix));
    await visitor.getByRole('button', { name: 'Find games' }).focus();
    await visitor.keyboard.press('Enter');
    const aliceRow = visitor.getByTestId('game-history-row').filter({ hasText: aliceName });
    const bobRow = visitor.getByTestId('game-history-row').filter({ hasText: bobName });
    await expect(aliceRow).toContainText('Human');
    await expect(aliceRow).toContainText('Bot');
    await expect(aliceRow).toContainText('Passive');
    await expect(bobRow).toContainText('In progress');
    await expect(bobRow.getByRole('link', { name: /^Replay game/ })).toHaveCount(0);
    await expect(visitor.getByText('Automated history check:', { exact: false })).toHaveCount(0);
    const filtered = visitor.url();
    await aliceRow.getByRole('link', { name: /^Replay game/ }).click();
    await expect(visitor.getByTestId('game-board')).toBeVisible();
    // Trivial opening windows are skipped natively; the first frame is a real decision.
    await expect(visitor.getByTestId('replay-perspective')).toContainText('Acting view · seat 1');
    await expect(visitor.getByRole('textbox', { name: /How did this game feel/ })).toHaveCount(0);
    await visitor.getByTestId('game-log').locator('button[data-move]').first().click();
    await expect(visitor.getByTestId('replay-perspective')).toContainText('Acting view · seat 1');
    await expect(visitor.getByTestId('game-board')).toBeVisible();
    await visitor.getByTestId('table').getByRole('link', { name: 'Games', exact: true }).focus();
    await visitor.keyboard.press('Enter');
    expect(failures).toEqual([]);
    await expect(visitor).toHaveURL(filtered);
    await aliceRow.getByRole('link', { name: 'Passive', exact: true }).click();
    await expect(visitor).toHaveURL(/player=control%3Apassive/);
    await expect(aliceRow).toBeVisible();
    await expect(bobRow).toBeVisible();
    await bobRow.getByRole('link', { name: 'Filter this pairing' }).click();
    await expect(visitor.getByTestId('game-history-row')).toHaveCount(1);
    await expect(visitor.getByTestId('game-history-row')).toContainText(bobName);
    await visitor.reload();
    await expect(visitor.getByTestId('game-history-row')).toHaveCount(1);
    await alice.goto('/games?scope=mine');
    await expect(alice.getByTestId('game-history-row')).toHaveCount(1);
    await expect(alice.getByTestId('game-history-row')).toContainText(aliceName);
    await expect(alice.getByTestId('game-history-row')).toContainText('private note');
    await bob.goto('/games?scope=mine');
    await expect(bob.getByTestId('game-history-row')).toHaveCount(1);
    await expect(bob.getByTestId('game-history-row')).toContainText(bobName);
    await bob.getByTestId('game-history-row').getByRole('link', { name: /^Replay game/ }).click();
    await expect(bob.getByTestId('game-board')).toBeVisible();
    expect(failures).toEqual([]);
  } finally {
    await Promise.all(contexts.map((context) => context.close()));
  }
});

test('long recorded identities fit Games and preserve full filter values', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  const name = 'ci-fix-528/consumer/release-smoke/checkpoints/learner_transitions-8';
  const version = 'a'.repeat(64);
  await page.route('**/api/traces?**', route => route.fulfill({ json: [{ id: 'long-identity', timestamp: '2026-11-14T00:00:00Z',
    status: 'completed', winner: 1, end_reason: 'game_over', origin: 'automated_validation', num_events: 5,
    replay_available: true, feedback: [], players: [
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
  await expect(row.getByRole('link', { name: 'aaaaaaaaaa', exact: true })).toHaveAttribute('href', `/games?version=${version}`);
  await expect(row.locator(`[title="${name} · ${version}"]`)).toHaveCount(1);
});
