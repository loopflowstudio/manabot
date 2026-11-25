/**
 * End-to-end game explanation proof (MTG-391):
 * - A real game's history indents consequences with their rule ("lethal damage"),
 *   and selecting lines draws their kept boards (cast/damage/death) while
 *   naming the cursor in the top bar.
 * - Opening the saved recording in Review draws the same history and steps into
 *   per-line boards retained on `TraceEvent.boards`.
 */
import { expect, test, type Page } from '@playwright/test';
import { selectDeck } from './deck-selection';

async function playToDeath(page: Page): Promise<boolean> {
  for (let i = 0; i < 100; i += 1) {
    await expect(page.locator('[data-testid=game][data-live=true], [data-testid=result-card]').first()).toBeVisible({ timeout: 30_000 });
    const deathLines = page.locator('[data-testid=game-log] [data-testid=history-consequence]').filter({ hasText: /died · lethal damage/ });
    if (await deathLines.count() > 0) return true;
    if (await page.getByTestId('result-card').isVisible()) break;
    const rows = page.getByTestId('action-option');
    const labels = await rows.allTextContents();
    // Prefer casting burn/creatures, targeting creatures, and attacking/blocking so combat or burn kills a creature.
    let pick = labels.findIndex(l => /Keep hand|Play (Island|Mountain)|Cast Lightning Bolt|Cast (Raging Goblin|Gray Ogre|Wind Drake)|Target (Gray Ogre|Wind Drake|Raging Goblin)|Attack with |Block /.test(l));
    if (pick < 0) pick = labels.findIndex(l => /Pass priority|Decline|do not/.test(l));
    const revision = Number(await page.locator('main').getAttribute('data-frame-revision'));
    await rows.nth(Math.max(0, pick)).click();
    await expect.poll(async () => Number(await page.locator('main').getAttribute('data-frame-revision')), { timeout: 30_000 }).toBeGreaterThan(revision);
  }
  const deathLines = page.locator('[data-testid=game-log] [data-testid=history-consequence]').filter({ hasText: /died · lethal damage/ });
  return (await deathLines.count()) > 0;
}

test('indented consequences, per-line kept boards, and Review TraceEvent.boards agree on a real game', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.routeWebSocket('**/ws/play', socket => {
    const server = socket.connectToServer();
    socket.onMessage(message => {
      if (typeof message !== 'string') { server.send(message); return; }
      const payload = JSON.parse(message);
      if (payload.type === 'new_game') payload.config.seed = 11;
      server.send(JSON.stringify(payload));
    });
  });

  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('random');
  await selectDeck(page, 'hero', 'interactive');
  await selectDeck(page, 'villain', 'interactive');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('game')).toBeVisible();

  const sawDeath = await playToDeath(page);
  expect(sawDeath).toBe(true);

  // Expand any collapsed busy-turn consequences so the death line is clickable.
  const toggles = page.locator('[data-testid=consequence-toggle][aria-expanded="false"]');
  for (let i = 0; i < await toggles.count(); i += 1) {
    await toggles.nth(0).click();
  }

  const deathButton = page.locator('[data-testid=game-log] [data-testid=history-consequence] button[data-history-seq]').filter({ hasText: /died · lethal damage/ }).first();
  await expect(deathButton).toBeVisible();
  const castButton = page.locator('[data-testid=game-log] button[data-history-seq]').filter({ hasText: /Cast Lightning Bolt, targeting Gray Ogre/ }).first();
  const damageButton = page.locator('[data-testid=game-log] [data-testid=history-consequence] button[data-history-seq]').filter({ hasText: /3 damage to Gray Ogre/ }).first();

  // 1. Cast board: Lightning Bolt on the stack, Gray Ogre still on the opponent's battlefield.
  await castButton.click();
  await expect(page.getByTestId('game-board')).toHaveAttribute('data-board', 'step');
  await expect(page.getByTestId('stack-entry').first()).toContainText('Lightning Bolt');
  await expect(page.locator('.half.opponent [data-testid=board-card]').filter({ hasText: 'Gray Ogre' })).toHaveCount(1);
  await page.screenshot({ path: testInfo.outputPath('live-cast-cursor.png') });

  // 2. Damage board: stepping → shows Lightning Bolt off the stack and Gray Ogre on the battlefield with 3 damage.
  await page.keyboard.press('ArrowRight');
  await expect(damageButton).toHaveAttribute('aria-current', 'true');
  await expect(page.getByTestId('stack-entry')).toHaveCount(0);
  await expect(page.locator('.half.opponent [data-testid=board-card]').filter({ hasText: 'Gray Ogre' })).toHaveCount(1);
  await expect(page.getByTestId('walk-caption')).toContainText('3 to Gray Ogre');
  await page.screenshot({ path: testInfo.outputPath('live-damage-cursor.png') });

  // 3. Death board: stepping → shows Gray Ogre gone, rule named in caption and top bar reading.
  await page.keyboard.press('ArrowRight');
  await expect(deathButton).toHaveAttribute('aria-current', 'true');
  await expect(page.locator('.half.opponent [data-testid=board-card]').filter({ hasText: 'Gray Ogre' })).toHaveCount(0);
  await expect(page.getByTestId('turn-number')).toContainText('Your Lightning Bolt');
  await expect(page.getByTestId('turn-number')).toContainText('Esc for now');
  await expect(page.getByTestId('walk-caption')).toContainText('lethal damage');
  await page.screenshot({ path: testInfo.outputPath('live-death-cursor.png') });

  // Esc returns to now.
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('game-board')).toHaveAttribute('data-board', 'frame');
  await expect(page.getByTestId('turn-number')).not.toContainText('Esc for now');

  // Grab the saved recording id from sessionStorage and open it in Review.
  const recordingId = await page.evaluate(() => {
    const raw = sessionStorage.getItem('etude.gui.resume');
    return raw ? (JSON.parse(raw) as { recording_id?: string }).recording_id ?? null : null;
  });
  expect(recordingId).toBeTruthy();

  await page.goto(`/replay?trace=${encodeURIComponent(recordingId!)}&review=1`);
  await expect(page.getByTestId('game-board')).toBeVisible({ timeout: 30_000 });

  // Expand busy turns in Review if any.
  const reviewToggles = page.locator('[data-testid=consequence-toggle][aria-expanded="false"]');
  for (let i = 0; i < await reviewToggles.count(); i += 1) {
    await reviewToggles.nth(0).click();
  }
  const reviewDamage = page.locator('[data-testid=game-log] [data-testid=history-consequence] button[data-move]').filter({ hasText: /3 damage to Gray Ogre/ }).first();
  const reviewDeath = page.locator('[data-testid=game-log] [data-testid=history-consequence] button[data-move]').filter({ hasText: /died · lethal damage/ }).first();
  // Selecting damage inside the consequences transition rebuilds its per-line board via TraceEvent.boards (Ogre still on board!).
  await reviewDamage.click();
  await expect(reviewDamage).toHaveAttribute('aria-current', 'true');
  await expect(page.locator('.half.opponent [data-testid=board-card]').filter({ hasText: 'Gray Ogre' })).toHaveCount(1);
  // Selecting died inside the same transition shows Gray Ogre gone.
  await reviewDeath.click();
  await expect(reviewDeath).toHaveAttribute('aria-current', 'true');
  await expect(page.locator('.half.opponent [data-testid=board-card]').filter({ hasText: 'Gray Ogre' })).toHaveCount(0);
  await expect(page.getByTestId('turn-number')).toContainText('Esc for now');
  await page.screenshot({ path: testInfo.outputPath('review-death-cursor.png') });
});
