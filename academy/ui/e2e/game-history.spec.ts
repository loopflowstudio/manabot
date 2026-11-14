import { selectDeck } from './deck-selection';
import { expect, test, type Page } from '@playwright/test';

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
    await page.getByTestId('opponent-select').selectOption('passive');
    await selectDeck(page, 'hero', 'gw_allies');
    await selectDeck(page, 'villain', 'ur_lessons');
    await page.getByRole('button', { name: 'New Game', exact: true }).click();
    await expect(page.getByTestId('game-board')).toBeVisible();
  }
  try {
    await start(alice, aliceName);
    const over = alice.getByText('Game Over', { exact: true });
    const actions = alice.getByTestId('action-option');
    for (let step = 0; step < 600 && !(await over.isVisible()); step++) {
      await expect(actions.first().or(over)).toBeVisible();
      if (await over.isVisible()) break;
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
    await expect(visitor.getByText('Decision before · Chosen action · Consequences after', { exact: true })).toBeVisible();
    await expect(visitor.getByRole('textbox', { name: /How did this game feel/ })).toHaveCount(0);
    await visitor.getByTestId('study-decision').first().click();
    await expect(visitor.getByText("Shared replay from Player 1's perspective.", { exact: false })).toBeVisible();
    await expect(visitor.getByTestId('game-board')).toBeVisible();
    await visitor.getByRole('link', { name: 'Back to games' }).focus();
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
