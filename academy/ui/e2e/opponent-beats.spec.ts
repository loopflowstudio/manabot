/**
 * The opponent's moves are read out one beat at a time over the live table:
 * each long enough to read, Space ends the reading without acting, and reduced
 * motion reads none (the recap still lists every fact).
 */
import { expect, test, type Page } from '@playwright/test';

async function startAgainstRandom(page: Page, sent: { count: number }): Promise<void> {
  page.on('websocket', socket => socket.on('framesent', frame => {
    if (String(frame.payload).includes('"type":"command"')) sent.count += 1;
  }));
  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('random');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('table')).toBeVisible();
}

/** Decide until `until` holds, preferring lands, spells and attacks so the game moves. */
async function playUntil(page: Page, until: () => Promise<boolean>, limit: number): Promise<number> {
  for (let decisions = 0; decisions < limit; decisions += 1) {
    await expect(page.locator('[data-testid=table][data-live=true], [data-testid=result-card]').first()).toBeVisible({ timeout: 30_000 });
    if (await until()) return decisions;
    if (await page.getByTestId('result-card').isVisible()) return decisions;
    const rows = page.getByTestId('action-option');
    const labels = await rows.allTextContents();
    let pick = labels.findIndex(label => /Play (Island|Mountain)|Cast |Attack with /.test(label));
    if (pick < 0) pick = labels.findIndex(label => /Pass priority|do not block|Decline/.test(label));
    const revision = Number(await page.locator('main').getAttribute('data-frame-revision'));
    await page.mouse.move(2, 2);
    await rows.nth(Math.max(0, pick)).click();
    await expect.poll(async () => Number(await page.locator('main').getAttribute('data-frame-revision')), { timeout: 30_000 }).toBeGreaterThan(revision);
  }
  return limit;
}

test('the opponent\'s moves are read out, and Space ends the reading without acting', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  // The slow pace leaves time to read a beat before skipping it.
  await page.addInitScript(() => localStorage.setItem('academy.table.beat-pace', 'slow'));
  const sent = { count: 0 };
  await startAgainstRandom(page, sent);
  const beat = page.getByTestId('beat');
  await playUntil(page, async () => await beat.count() > 0, 120);
  // Read the beat and skip it within its 1.4 s, before it ends by itself.
  const text = await page.evaluate(() => document.querySelector('[data-testid=beat]')?.textContent ?? '');
  // A beat is shown on the board it happened on, when that is not the frame's
  // own: a past board, with no card outlined (its offers belong to the frame).
  const board = page.getByTestId('game-board');
  const onStep = await board.getAttribute('data-board') === 'step';
  if (onStep) expect(await board.locator('[data-testid=table-card][data-offer-count]:not([data-offer-count="0"])').count()).toBe(0);
  const before = sent.count;
  await page.keyboard.press('Space');
  expect(text).toMatch(/ cast | played | discarded | attacked with | dealt | died| blocked /);
  expect(text).toMatch(/\d+ of \d+ · Space to skip/);
  await expect(beat).toHaveCount(0);
  // Skipping jumps to the frame's own board.
  await expect(board).toHaveAttribute('data-board', 'frame');
  expect(sent.count).toBe(before);
  await page.screenshot({ path: testInfo.outputPath('after-skip.png') });
  // Every fact read out stays in the recap.
  await expect(page.getByTestId('recent')).toBeVisible();
});

test('each beat is shown on its own board, and pressing the board skips to the frame\'s', async ({ page }) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.addInitScript(() => localStorage.setItem('academy.table.beat-pace', 'slow'));
  const sent = { count: 0 };
  await startAgainstRandom(page, sent);
  const board = page.getByTestId('game-board');
  await playUntil(page, async () => await board.getAttribute('data-board') === 'step', 160);
  await expect(board).toHaveAttribute('data-board', 'step');
  await page.screenshot({ path: test.info().outputPath('step-board.png') });
  const before = sent.count;
  // A press on a past board commits nothing: its cards had no offers when pressed.
  const card = board.getByTestId('table-card').first();
  await card.click();
  await expect(board).toHaveAttribute('data-board', 'frame');
  await expect(page.getByTestId('beat')).toHaveCount(0);
  expect(sent.count).toBe(before);
});

test('reduced motion reads out no beats', async ({ page }) => {
  test.setTimeout(180_000);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const sent = { count: 0 };
  await startAgainstRandom(page, sent);
  let opponentFacts = 0;
  await playUntil(page, async () => {
    expect(await page.getByTestId('beat').count()).toBe(0);
    opponentFacts = await page.locator('[data-testid=recent-fact].theirs').count();
    return opponentFacts > 0;
  }, 120);
  expect(opponentFacts).toBeGreaterThan(0);
  await expect(page.getByTestId('beat')).toHaveCount(0);
});

test('the Menu turns the reading off, and the browser remembers it', async ({ page }) => {
  test.setTimeout(180_000);
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  const sent = { count: 0 };
  await startAgainstRandom(page, sent);
  await page.getByTestId('table-menu').click();
  const pace = page.getByTestId('beat-pace');
  await expect(pace.getByRole('radio', { name: 'Quick' })).toBeChecked();
  await pace.getByRole('radio', { name: 'Off' }).check();
  await page.getByTestId('table-menu').click();
  let opponentFacts = 0;
  await playUntil(page, async () => {
    expect(await page.getByTestId('beat').count()).toBe(0);
    opponentFacts = await page.locator('[data-testid=recent-fact].theirs').count();
    return opponentFacts > 0;
  }, 120);
  expect(opponentFacts).toBeGreaterThan(0);
  await page.reload();
  await expect(page.getByTestId('table')).toBeVisible();
  await page.getByTestId('table-menu').click();
  await expect(page.getByTestId('beat-pace').getByRole('radio', { name: 'Off' })).toBeChecked();
});
