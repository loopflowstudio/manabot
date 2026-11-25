/**
 * The opponent's moves are read out one beat at a time before the decision:
 * each long enough to read, the decision's choices wait beneath it, the first
 * input ends the reading without acting; damage marks summing the window and
 * the result card wait with it; and reduced motion reads none (the
 * recap still lists every fact). Playwright waits on a click at a held choice, as
 * it treats `aria-disabled` as not yet enabled.
 */
import { expect, test, type Page } from '@playwright/test';
import { showEachCreaturesChoice } from './fixtures/combat';

/**
 * Start a seeded game against the random opponent, counting the commands sent.
 * Seed 1 reaches a past board (an opponent window of several moves) at the
 * fourth decision under `playUntil`'s choices.
 */
async function startAgainstRandom(page: Page): Promise<{ count: number }> {
  const sent = { count: 0 };
  await page.routeWebSocket('**/ws/play', socket => {
    const server = socket.connectToServer();
    socket.onMessage(message => {
      if (typeof message !== 'string') { server.send(message); return; }
      const payload = JSON.parse(message);
      if (payload.type === 'new_game') payload.config.seed = 1;
      if (payload.type === 'command') sent.count += 1;
      server.send(JSON.stringify(payload));
    });
  });
  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('random');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('game')).toBeVisible();
  return sent;
}

/**
 * Decide until `until` returns a truthy result, preferring lands, spells and
 * attacks so the game moves. Throws, naming why, when the game ends or the
 * budget runs out first.
 */
async function playUntil<T>(page: Page, until: () => Promise<T | null | false>, limit: number): Promise<T> {
  for (let decisions = 0; decisions < limit; decisions += 1) {
    await expect(page.locator('[data-testid=game][data-live=true], [data-testid=result-card]').first()).toBeVisible({ timeout: 30_000 });
    const result = await until();
    if (result) return result;
    if (await page.getByTestId('result-card').isVisible()) throw new Error(`the game ended after ${decisions} decisions before the awaited state`);
    await showEachCreaturesChoice(page);
    const rows = page.getByTestId('action-option');
    const labels = await rows.allTextContents();
    let pick = labels.findIndex(label => /Play (Island|Mountain)|Cast |Attack with /.test(label));
    if (pick < 0) pick = labels.findIndex(label => /Pass priority|do not block|Decline/.test(label));
    const revision = Number(await page.locator('main').getAttribute('data-frame-revision'));
    await page.mouse.move(2, 2);
    await rows.nth(Math.max(0, pick)).click();
    await expect.poll(async () => Number(await page.locator('main').getAttribute('data-frame-revision')), { timeout: 30_000 }).toBeGreaterThan(revision);
  }
  throw new Error(`no awaited state within the ${limit}-decision budget`);
}

/** What the page showed in the instant before a press during a beat. */
interface Seen { text: string; inPanel: boolean; held: boolean; onStep: boolean; outlined: number; windowMarks: number }

/**
 * Press during a beat in one task, so the beat cannot end between reading the
 * page and the press: a held number key, activation of the first choice with
 * or without a pointer, or a pointer press on a card of the past board. Null
 * when no beat was showing (for `board`, no beat on a past board).
 */
async function pressDuringBeat(page: Page, how: 'key' | 'pointer' | 'activation' | 'board'): Promise<Seen | null> {
  return page.evaluate(how => {
    const beat = document.querySelector('[data-testid=beat]');
    if (!beat) return null;
    const row = document.querySelector<HTMLElement>('[data-testid=action-option]')!;
    const board = document.querySelector('[data-testid=game-board]')!;
    const card = board.querySelector<HTMLElement>('[data-testid=board-card]');
    if (how === 'board' && (board.getAttribute('data-board') !== 'step' || !card)) return null;
    const seen = {
      text: beat.textContent ?? '',
      inPanel: beat.closest('[data-testid=action-panel]') !== null,
      held: row.getAttribute('aria-disabled') === 'true',
      onStep: board.getAttribute('data-board') === 'step',
      outlined: board.querySelectorAll('[data-testid=board-card][data-offer-count]:not([data-offer-count="0"])').length,
      windowMarks: document.querySelectorAll('[data-testid=player-damage], [data-testid=damage-dealt], [data-testid=result-card]').length,
    };
    if (how === 'key') {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: '1', bubbles: true }));
      window.dispatchEvent(new KeyboardEvent('keydown', { key: '1', bubbles: true, repeat: true }));
    } else if (how === 'pointer' || how === 'board') {
      const target = how === 'board' ? card! : row;
      target.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0 }));
      target.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, detail: 1 }));
    } else row.click();
    return seen;
  }, how);
}

test('the opponent\'s moves are read out before the decision, and the first press ends the reading without acting', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  const sent = await startAgainstRandom(page);
  const main = page.locator('main');
  // A number key during a beat ends the reading and chooses nothing.
  const { read, before, revision } = await playUntil(page, async () => {
    const before = sent.count;
    const revision = await main.getAttribute('data-frame-revision');
    const read = await pressDuringBeat(page, 'key');
    return read && { read, before, revision };
  }, 120);
  expect(read.text).toMatch(/ cast | played | discarded | attacked with | dealt | died| blocked /);
  expect(read.text).toMatch(/\d+ of \d+ · Space to skip/);
  // The beat headed the decision block, and the choices beneath it waited, still listed.
  expect(read.inPanel).toBe(true);
  expect(read.held).toBe(true);
  // A beat is shown on the board it happened on, when that is not the frame's
  // own: a past board, with no card outlined (its offers belong to the frame).
  if (read.onStep) expect(read.outlined).toBe(0);
  // Damage marks, which add up the whole window, and the result card wait with the decision.
  expect(read.windowMarks).toBe(0);
  const board = page.getByTestId('game-board');
  await expect(page.getByTestId('beat')).toHaveCount(0);
  // Ending the reading shows the frame's own board, with the live decision.
  await expect(board).toHaveAttribute('data-board', 'frame');
  await expect(page.getByTestId('action-option').first()).not.toHaveAttribute('aria-disabled', 'true');
  expect(sent.count).toBe(before);
  await expect(main).toHaveAttribute('data-frame-revision', revision!);
  // The next press acts.
  await page.keyboard.press('1');
  await expect.poll(() => sent.count).toBe(before + 1);
  await page.screenshot({ path: testInfo.outputPath('after-skip.png') });
  // Every fact read out stays in the recap.
  await expect(page.getByTestId('recent')).toBeVisible();
});

test('pressing the board or activating a choice ends the reading without acting', async ({ page }) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  const sent = await startAgainstRandom(page);
  const board = page.getByTestId('game-board');
  // A press on a past board (cards with no offers), a pointer press on a held
  // choice, or assistive-technology activation without pointer events all end
  // the reading without acting.
  for (const how of ['board', 'pointer', 'activation'] as const) {
    const { seen, before } = await playUntil(page, async () => {
      const before = sent.count;
      const seen = await pressDuringBeat(page, how);
      return seen && { seen, before };
    }, 160);
    if (how === 'board') expect(seen.onStep).toBe(true);
    await expect(board).toHaveAttribute('data-board', 'frame');
    await expect(page.getByTestId('beat')).toHaveCount(0);
    expect(sent.count).toBe(before);
  }
});

test('reduced motion reads out no beats', async ({ page }) => {
  test.setTimeout(180_000);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await startAgainstRandom(page);
  await playUntil(page, async () => {
    expect(await page.getByTestId('beat').count()).toBe(0);
    return await page.locator('[data-testid=recent-fact].theirs').count() > 0;
  }, 120);
  await expect(page.getByTestId('beat')).toHaveCount(0);
});
