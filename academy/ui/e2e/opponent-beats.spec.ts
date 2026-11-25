/**
 * The opponent's moves are read out one beat at a time before the decision:
 * each long enough to read, the decision's choices wait beneath it, the first
 * input ends the reading without acting; damage marks summing the window and
 * the result card wait with it; and reduced motion reads none (the
 * recap still lists every fact). Playwright waits on a click at a held choice, as
 * it treats `aria-disabled` as not yet enabled.
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

/** What the page showed in the instant before a press during a beat. */
interface Seen { text: string; inPanel: boolean; held: boolean; onStep: boolean; outlined: number; windowMarks: number }

/**
 * Press during a beat in one task, so the beat cannot end between reading the
 * page and the press: a held number key, or activation of the first choice
 * with or without a pointer. Null when no beat was showing.
 */
async function pressDuringBeat(page: Page, how: 'key' | 'pointer' | 'activation'): Promise<Seen | null> {
  return page.evaluate(how => {
    const beat = document.querySelector('[data-testid=beat]');
    if (!beat) return null;
    const row = document.querySelector<HTMLElement>('[data-testid=action-option]')!;
    const board = document.querySelector('[data-testid=game-board]')!;
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
    } else if (how === 'pointer') {
      row.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
      row.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, detail: 1 }));
    } else row.click();
    return seen;
  }, how);
}

test('the opponent\'s moves are read out before the decision, and the first press ends the reading without acting', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  // The slow pace leaves time to read a beat before it ends by itself.
  await page.addInitScript(() => localStorage.setItem('academy.table.beat-pace', 'slow'));
  const sent = { count: 0 };
  await startAgainstRandom(page, sent);
  const main = page.locator('main');
  let seen: Seen | null = null;
  let before = 0;
  let revision: string | null = null;
  // A number key during a beat ends the reading and chooses nothing.
  await playUntil(page, async () => {
    before = sent.count;
    revision = await main.getAttribute('data-frame-revision');
    seen = await pressDuringBeat(page, 'key');
    return seen !== null;
  }, 120);
  const read = seen as Seen | null;
  expect(read).not.toBeNull();
  expect(read!.text).toMatch(/ cast | played | discarded | attacked with | dealt | died| blocked /);
  expect(read!.text).toMatch(/\d+ of \d+ · Space to skip/);
  // The beat headed the decision block, and the choices beneath it waited, still listed.
  expect(read!.inPanel).toBe(true);
  expect(read!.held).toBe(true);
  // A beat is shown on the board it happened on, when that is not the frame's
  // own: a past board, with no card outlined (its offers belong to the frame).
  if (read!.onStep) expect(read!.outlined).toBe(0);
  // Damage marks, which add up the whole window, and the result card wait with the decision.
  expect(read!.windowMarks).toBe(0);
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
  await page.addInitScript(() => localStorage.setItem('academy.table.beat-pace', 'slow'));
  const sent = { count: 0 };
  await startAgainstRandom(page, sent);
  const board = page.getByTestId('game-board');
  await playUntil(page, async () => await board.getAttribute('data-board') === 'step', 160);
  await expect(board).toHaveAttribute('data-board', 'step');
  await page.screenshot({ path: test.info().outputPath('step-board.png') });
  const before = sent.count;
  // A press on a past board commits nothing: its cards had no offers when pressed.
  const card = board.getByTestId('board-card').first();
  await card.click();
  await expect(board).toHaveAttribute('data-board', 'frame');
  await expect(page.getByTestId('beat')).toHaveCount(0);
  expect(sent.count).toBe(before);
  // A held choice can also be activated without pointer events by assistive technology.
  for (const how of ['pointer', 'activation'] as const) {
    let heldBefore = 0;
    let seen = false;
    await playUntil(page, async () => {
      heldBefore = sent.count;
      seen = await pressDuringBeat(page, how) !== null;
      return seen;
    }, 160);
    expect(seen).toBe(true);
    await expect(page.getByTestId('beat')).toHaveCount(0);
    await expect(board).toHaveAttribute('data-board', 'frame');
    expect(sent.count).toBe(heldBefore);
  }
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
