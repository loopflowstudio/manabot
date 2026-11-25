/**
 * The opponent's turn walks on the table one kept board stop at a time:
 * steady-height caption and scrubber in `.bottom`, pointer rest on the caption
 * holds, clicking a tick holds at that stop, "Show their turn again" replays
 * from the turn's first stop, Space and clicking the earlier board only return
 * to now without acting, the rail stays live (pressing a choice or number key
 * returns to now and commits), selecting a history line + ← → + Esc move the
 * same cursor, and reduced motion never auto-walks while still allowing manual
 * stepping.
 */
import { expect, test, type Page } from '@playwright/test';

async function startAgainstRandom(page: Page, sent: { count: number }, seed = 0): Promise<void> {
  await page.routeWebSocket('**/ws/play', socket => {
    const server = socket.connectToServer();
    socket.onMessage(message => {
      if (typeof message !== 'string') { server.send(message); return; }
      const payload = JSON.parse(message);
      if (payload.type === 'new_game') payload.config.seed = seed;
      if (payload.type === 'command') sent.count += 1;
      server.send(JSON.stringify(payload));
    });
  });
  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('random');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('game')).toBeVisible();
}

/** Decide until `until` holds, preferring lands, spells and attacks so the game moves. */
async function playUntil(page: Page, until: () => Promise<boolean>, limit: number): Promise<number> {
  for (let decisions = 0; decisions < limit; decisions += 1) {
    await expect(page.locator('[data-testid=game][data-live=true], [data-testid=result-card]').first()).toBeVisible({ timeout: 30_000 });
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

interface WalkSnapshot {
  caption: string;
  walking: boolean;
  ticks: number;
  moveTicks: number;
  railDisabled: boolean;
  onStep: boolean;
  outlined: number;
  windowMarks: number;
  topBar: string;
  walkBarHeight: number;
}

async function snapshotWhileWalking(page: Page): Promise<WalkSnapshot | null> {
  return page.evaluate(() => {
    const bar = document.querySelector<HTMLElement>('[data-testid=walk-bar][data-walking=true]');
    if (!bar) return null;
    const caption = bar.querySelector('[data-testid=walk-caption]')?.textContent?.trim() ?? '';
    const row = document.querySelector<HTMLElement>('[data-testid=action-option]');
    const board = document.querySelector('[data-testid=game-board]')!;
    const topBar = document.querySelector('[data-testid=turn-number]')?.textContent?.trim() ?? '';
    return {
      caption,
      walking: bar.getAttribute('data-walking') === 'true',
      ticks: bar.querySelectorAll('[data-testid=walk-tick]').length,
      moveTicks: bar.querySelectorAll('[data-testid=walk-tick][data-move-start=true]').length,
      railDisabled: row?.getAttribute('aria-disabled') === 'true',
      onStep: board.getAttribute('data-board') === 'step',
      outlined: board.querySelectorAll('[data-testid=board-card][data-offer-count]:not([data-offer-count="0"])').length,
      windowMarks: document.querySelectorAll('[data-testid=player-damage], [data-testid=damage-dealt], [data-testid=result-card], [data-testid=recent]').length,
      topBar,
      walkBarHeight: Math.round(bar.getBoundingClientRect().height),
    };
  });
}

test('the board walks the opponent turn with a steady-height caption and scrubber; Space returns to now and number keys act', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'no-preference', colorScheme: 'light' });
  const sent = { count: 0 };
  await startAgainstRandom(page, sent);

  const main = page.locator('main');
  const board = page.getByTestId('game-board');
  let seen: WalkSnapshot | null = null;
  let beforeSpace = 0;
  let revision: string | null = null;

  await playUntil(page, async () => {
    beforeSpace = sent.count;
    revision = await main.getAttribute('data-frame-revision');
    seen = await snapshotWhileWalking(page);
    if (!seen) return false;
    // Immediately press Space while walking: returns to now and sends nothing.
    await page.keyboard.press('Space');
    return true;
  }, 120);

  expect(seen).not.toBeNull();
  expect(seen!.walking).toBe(true);
  expect(seen!.caption).toMatch(/ cast | played | discarded | attacked with | dealt | died| blocked | activated |Trigger/i);
  expect(seen!.ticks).toBeGreaterThanOrEqual(1);
  expect(seen!.moveTicks).toBeGreaterThanOrEqual(1);
  // The rail stays live throughout the walk (no MTG-396 aria-disabled hold).
  expect(seen!.railDisabled).toBe(false);
  // A past board has no live card offers outlined, and window totals/result card wait.
  if (seen!.onStep) expect(seen!.outlined).toBe(0);
  expect(seen!.windowMarks).toBe(0);
  // The top bar names the cursor and Esc for now.
  expect(seen!.topBar).toContain('Esc for now');

  // Space returned to now without submitting any Command.
  await expect(board).toHaveAttribute('data-board', 'frame');
  await expect(page.getByTestId('walk-bar')).toHaveAttribute('data-walking', 'false');
  expect(sent.count).toBe(beforeSpace);
  await expect(main).toHaveAttribute('data-frame-revision', revision!);

  // Steady height: `.walk-bar` does not reflow when the walk ends.
  const idleHeight = Math.round((await page.getByTestId('walk-bar').boundingBox())!.height);
  expect(idleHeight).toBe(seen!.walkBarHeight);

  // "Show their turn again" replays from the turn's first stop; hovering the caption holds.
  await page.getByTestId('walk-replay').click();
  await expect(page.getByTestId('walk-bar')).toHaveAttribute('data-walking', 'true');
  await expect(board).toHaveAttribute('data-board', 'step');
  await page.emulateMedia({ colorScheme: 'light' });
  await page.getByTestId('walk-caption').hover();
  await expect(page.getByTestId('walk-caption')).toHaveAttribute('data-holding', 'true');
  await expect(page.getByTestId('walk-count')).toHaveText(/^\d+ of \d+ · holding$/);
  await page.waitForTimeout(150);
  await page.screenshot({ path: testInfo.outputPath('walk-holding-wash.png') });
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.waitForTimeout(150);
  await page.screenshot({ path: testInfo.outputPath('walk-holding-night.png') });
  await page.emulateMedia({ colorScheme: 'light' });
  await page.mouse.move(2, 2);

  // Clicking a scrubber tick puts the cursor on that stop and holds (walking = false).
  await page.getByTestId('walk-tick').first().click();
  await expect(page.getByTestId('walk-bar')).toHaveAttribute('data-walking', 'false');
  await expect(board).toHaveAttribute('data-board', 'step');
  await expect(page.getByTestId('turn-number')).toContainText('Esc for now');

  // Pressing a number key while looking back returns to now AND commits immediately.
  const beforeNum = sent.count;
  await page.keyboard.press('1');
  await expect.poll(() => sent.count, { timeout: 15_000 }).toBe(beforeNum + 1);
  await expect(page.getByTestId('recent')).toBeVisible();
});

test('clicking the earlier board returns to now; selecting a history line and arrows step kept boards', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'no-preference', colorScheme: 'light' });
  const sent = { count: 0 };
  await startAgainstRandom(page, sent);

  const board = page.getByTestId('game-board');
  await playUntil(page, async () => await board.getAttribute('data-board') === 'step', 160);
  await expect(board).toHaveAttribute('data-board', 'step');
  await page.screenshot({ path: testInfo.outputPath('step-board.png') });

  // Clicking a card on an earlier board returns to now and commits nothing.
  const beforeBoardClick = sent.count;
  await board.getByTestId('board-card').first().click();
  await expect(board).toHaveAttribute('data-board', 'frame');
  await expect(page.getByTestId('walk-bar')).toHaveAttribute('data-walking', 'false');
  expect(sent.count).toBe(beforeBoardClick);

  // Selecting a history line in the rail puts the cursor on that line's kept board.
  const lines = page.locator('[data-testid=game-log] button[data-history-seq]');
  const count = await lines.count();
  expect(count).toBeGreaterThanOrEqual(2);
  await lines.nth(0).click();
  await expect(board).toHaveAttribute('data-board', 'step');
  await expect(lines.nth(0)).toHaveAttribute('aria-current', 'true');
  await expect(page.getByTestId('turn-number')).toContainText('Esc for now');

  // → steps forward one history line; ← steps back; Esc returns to now.
  const firstSeq = await page.getByTestId('walk-bar').getAttribute('data-cursor');
  await page.keyboard.press('ArrowRight');
  const nextSeq = await page.getByTestId('walk-bar').getAttribute('data-cursor');
  expect(nextSeq).not.toBe(firstSeq);
  await page.keyboard.press('ArrowLeft');
  await expect(page.getByTestId('walk-bar')).toHaveAttribute('data-cursor', firstSeq!);
  await page.keyboard.press('Escape');
  await expect(board).toHaveAttribute('data-board', 'frame');
  await expect(page.getByTestId('turn-number')).not.toContainText('Esc for now');
});

test('reduced motion does not auto-walk, while scrubber and Show their turn again still work by hand', async ({ page }) => {
  test.setTimeout(180_000);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const sent = { count: 0 };
  await startAgainstRandom(page, sent);
  let opponentFacts = 0;
  await playUntil(page, async () => {
    expect(await page.locator('[data-testid=walk-bar][data-walking=true]').count()).toBe(0);
    opponentFacts = await page.locator('[data-testid=recent-fact].theirs').count();
    return opponentFacts > 0;
  }, 120);
  expect(opponentFacts).toBeGreaterThan(0);
  await expect(page.getByTestId('game-board')).toHaveAttribute('data-board', 'frame');
  await expect(page.getByTestId('walk-bar')).toHaveAttribute('data-walking', 'false');
  // Manual replay steps to stop 0 and holds without auto-advancing.
  await page.getByTestId('walk-replay').click();
  await expect(page.getByTestId('walk-bar')).toHaveAttribute('data-walking', 'false');
  await expect(page.getByTestId('game-board')).toHaveAttribute('data-board', 'step');
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('game-board')).toHaveAttribute('data-board', 'frame');
});
test('a reload never walks turns already shown, and the next opponent turn still walks', async ({ page }) => {
  test.setTimeout(180_000);
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  const sent = { count: 0 };
  await startAgainstRandom(page, sent);
  await playUntil(page, async () => (await page.locator('[data-testid=recent-fact].theirs').count()) > 0, 120);
  await expect(page.getByTestId('walk-bar')).toHaveAttribute('data-walking', 'false', { timeout: 30_000 });
  const shownLines = await page.locator('[data-testid=game-log] button[data-history-seq]').count();

  await page.reload();
  await expect(page.getByTestId('game')).toBeVisible();
  await expect(page.locator('[data-testid=game-log] button[data-history-seq]')).toHaveCount(shownLines);
  for (let sample = 0; sample < 8; sample += 1) {
    expect(await page.locator('[data-testid=walk-bar][data-walking=true]').count()).toBe(0);
    await expect(page.getByTestId('game-board')).toHaveAttribute('data-board', 'frame');
    await page.waitForTimeout(200);
  }

  let walked = false;
  await playUntil(page, async () => {
    walked = (await page.locator('[data-testid=walk-bar][data-walking=true]').count()) > 0;
    return walked;
  }, 120);
  expect(walked).toBe(true);
});
