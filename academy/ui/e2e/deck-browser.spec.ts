import { expect, test } from '@playwright/test';
import manifest from '../src/lib/packs/elemental-world/v2/manifest.json' with { type: 'json' };
import { selectDeck } from './deck-selection';

test('Watch uses the same browser to assign both bot seats', async ({ page }) => {
  await page.goto('/replay?watch=1');
  await expect(page.getByTestId('deck-browser')).toHaveCount(1);
  await selectDeck(page, 'hero', 'elemental_bg');
  await selectDeck(page, 'villain', 'elemental_ug');
  await expect(page.getByTestId('selected-deck-hero')).toHaveAttribute('data-deck-id', 'elemental_bg');
  await expect(page.getByTestId('selected-deck-villain')).toHaveAttribute('data-deck-id', 'elemental_ug');
  const started = page.waitForRequest(request => request.url().endsWith('/api/watch') && request.method() === 'POST');
  await page.getByRole('button', { name: 'Start watching', exact: true }).click();
  expect((await started).postDataJSON()).toMatchObject({ hero_deck: 'elemental_bg', villain_deck: 'elemental_ug' });
  await expect(page.getByTestId('game-board')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Step decision', exact: true })).toBeEnabled();
});

test('browse every exact deck and Lesson pool, choose both seats, and keep a running game intact', async ({ page }, testInfo) => {
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  const browser = page.getByTestId('deck-browser');
  const detail = page.getByTestId('deck-detail');
  await expect(page.getByTestId('deck-browser')).toHaveCount(1);
  await expect(page.getByTestId('game-header').locator('select')).toHaveCount(1);
  for (const deck of manifest.roster) {
    await browser.getByTestId(`browse-${deck.deck_id}`).click();
    await expect(detail.getByRole('heading', { level: 3 })).toContainText(deck.display_name.replace('Elemental ', ''));
    for (const name of Object.keys(deck.cards)) await expect(detail.getByLabel('Main deck').locator('span').getByText(name, { exact: true })).toBeVisible();
    const lessons = detail.getByRole('list');
    await expect(lessons.getByRole('listitem')).toHaveCount(Object.keys(deck.sideboard).length);
    for (const name of Object.keys(deck.sideboard)) await expect(lessons).toContainText(name);
  }
  await browser.getByTestId('browse-elemental_ug').click();
  await detail.getByRole('button', { name: 'Use for me', exact: true }).click();
  await detail.getByRole('button', { name: 'Use for opponent', exact: true }).click();
  await expect(page.getByTestId('selected-deck-hero')).toHaveAttribute('data-deck-id', 'elemental_ug');
  await expect(page.getByTestId('selected-deck-villain')).toHaveAttribute('data-deck-id', 'elemental_ug');
  await page.screenshot({ path: testInfo.outputPath('deck-browser-desktop.png'), fullPage: true });
  const viewport = page.viewportSize()!;
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('deck-browser-mobile.png'), fullPage: true });
  await page.setViewportSize(viewport);
  await page.getByTestId('opponent-select').selectOption('passive');
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  await expect(page.getByTestId('action-option').first()).toBeVisible();
  const revision = await page.locator('main').getAttribute('data-frame-revision');
  await page.getByRole('button', { name: 'Choose decks', exact: true }).click();
  await browser.getByTestId('browse-elemental_bg').click();
  await detail.getByRole('button', { name: 'Use for opponent', exact: true }).click();
  await expect(page.locator('main')).toHaveAttribute('data-frame-revision', revision!);
  await expect(page.getByTestId('deck-names')).toContainText('Temur Lessons');
  await page.getByRole('button', { name: 'Return to game', exact: true }).click();
  await expect(browser).not.toBeVisible();
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  await expect(page.getByTestId('deck-names')).toContainText('Landfall and sacrifice');
});

test('cached artwork covers the Elemental lists and opens a readable printed card', async ({ page }, testInfo) => {
  test.skip(process.env.ACADEMY_REQUIRE_CARD_ART !== '1', 'Requires the local Scryfall art cache.');
  const remote: string[] = [];
  page.on('request', request => { if (request.url().startsWith('https://')) remote.push(request.url()); });
  await page.goto('/?hero=elemental_ur&villain=elemental_gw');
  for (const deck of manifest.roster) {
    await page.getByTestId(`browse-${deck.deck_id}`).click();
    const cards = page.getByTestId('deck-detail').getByTestId('card-treatment');
    await expect(cards).toHaveCount(Object.keys(deck.cards).length + Object.keys(deck.sideboard).length);
    await expect(page.getByTestId('deck-detail').locator('[data-art-source="procedural"]')).toHaveCount(0);
  }
  await page.getByTestId('browse-elemental_ur').click();
  await page.screenshot({ path: testInfo.outputPath('unified-decks-art.png'), fullPage: true });
  await page.getByLabel('Main deck').getByRole('button', { name: 'Inspect Firebending Lesson', exact: true }).click();
  const preview = page.getByRole('dialog', { name: 'Card preview: Firebending Lesson' });
  await expect(preview).toBeVisible();
  await expect.poll(() => preview.locator('img').evaluate((image: HTMLImageElement) => image.naturalWidth)).toBeGreaterThan(0);
  await page.screenshot({ path: testInfo.outputPath('printed-card-preview.png') });
  await page.keyboard.press('Escape');
  await expect(preview).not.toBeVisible();
  expect(remote).toEqual([]);
});

test('a pinned local model link selects the model and both Elemental decks', async ({ page, request }) => {
  const catalog = await (await request.get('/api/opponents')).json();
  const model = catalog.opponents.find((entry: { name: string; available: boolean }) => entry.available && entry.name.includes('MTG-183'));
  test.skip(!model, 'Run with ACADEMY_PLAY_CHECKPOINTS pointing at the retained MTG-183 model.');
  await page.goto(`/?hero=elemental_ur&villain=elemental_gw&opponent=${model.sha256}`);
  await expect(page.getByTestId('opponent-select')).toHaveValue(`checkpoint:${model.sha256}`);
  await expect(page.getByTestId('selected-deck-hero')).toHaveAttribute('data-deck-id', 'elemental_ur');
  await expect(page.getByTestId('selected-deck-villain')).toHaveAttribute('data-deck-id', 'elemental_gw');
  await page.setViewportSize({ width: 390, height: 844 });
  const bounds = await page.getByTestId('opponent-select').boundingBox();
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390);
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  await expect(page.getByTestId('action-option').first()).toBeVisible();
  await expect(page.getByTestId('deck-names')).toContainText('MTG-183');
  await page.reload();
  await expect(page.getByTestId('deck-names')).toContainText('MTG-183');
  await expect(page.getByTestId('action-option').first()).toBeVisible();
});

test('priority automation advances into a real choice and remains visible in the game log', async ({ page }) => {
  // Seed 0 gives the hero castable instants once lands are down, so priority
  // windows outside the configured stops become real choices that automation
  // passes; earlier windows hold only Pass priority and native skipping takes them.
  await page.routeWebSocket('**/ws/play', socket => {
    const server = socket.connectToServer();
    socket.onMessage(message => {
      if (typeof message !== 'string') { server.send(message); return; }
      const payload = JSON.parse(message);
      if (payload.type === 'new_game') payload.config.seed = 0;
      server.send(JSON.stringify(payload));
    });
  });
  await page.goto('/?hero=elemental_ur&villain=elemental_gw');
  await page.getByTestId('opponent-select').selectOption('passive');
  await page.getByRole('button', { name: 'New Game', exact: true }).click();
  await expect.poll(() => page.getByTestId('action-option').count()).toBeGreaterThan(1);
  await page.getByText('Decision log', { exact: true }).click();
  const automated = page.getByTestId('log-entry').filter({ hasText: 'Auto: Pass priority' });
  for (let input = 0; input < 6 && await automated.count() === 0; input++) {
    const revision = Number(await page.locator('main').getAttribute('data-frame-revision'));
    const labels = await page.getByTestId('action-option').allTextContents();
    const land = labels.findIndex(label => /Play (Island|Mountain)/.test(label));
    await page.getByTestId('action-option').nth(land === -1 ? labels.findIndex(label => label.includes('Pass priority')) : land).click();
    await expect.poll(async () => Number(await page.locator('main').getAttribute('data-frame-revision'))).toBeGreaterThan(revision);
    await expect.poll(() => page.getByTestId('action-option').count()).toBeGreaterThan(1);
  }
  await expect(automated.first()).toBeVisible();
  const revision = await page.locator('main').getAttribute('data-frame-revision');
  await page.reload();
  await expect(page.locator('main')).toHaveAttribute('data-frame-revision', revision!);
  await expect.poll(() => page.getByTestId('action-option').count()).toBeGreaterThan(1);
});
