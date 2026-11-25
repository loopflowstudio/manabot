/**
 * Study's acceptance: a game against the Reader fixture policy, reviewed
 * against that policy's analysis, counted in Games, and played on from one of
 * its decisions; and a game with analysis shown at one decision, marked so in
 * Review and Games. Served by `tests/fixtures/policy_runtime_server.py`.
 */
import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const proof = new URL('../test-results/policy/addresses.json', import.meta.url);
const study = (): string => (JSON.parse(readFileSync(proof, 'utf8')) as { study: string }).study;
const shot = (name: string): string => fileURLToPath(new URL(`../test-results/policy/${name}.png`, import.meta.url));

/** Every socket message the page receives, parsed. */
function received(page: Page): Record<string, unknown>[] {
  const messages: Record<string, unknown>[] = [];
  page.on('websocket', socket => socket.on('framereceived', event => messages.push(JSON.parse(String(event.payload)))));
  return messages;
}

/** Choose until the game is over, preferring lands, spells and attacks so it moves.
 * `before` runs at each decision first; returning true makes it the one chosen. */
async function playToEnd(page: Page, before: (decision: number) => Promise<void> = async () => {}): Promise<void> {
  const over = page.getByTestId('result-card');
  const actions = page.getByTestId('action-option');
  for (let decision = 0; decision < 400; decision++) {
    await expect(actions.first().or(over)).toBeVisible({ timeout: 30_000 });
    if (await over.isVisible()) return;
    await expect(actions.first()).toBeEnabled();
    await before(decision);
    const seq = Number(await page.locator('main').getAttribute('data-update-seq'));
    const labels = await actions.allTextContents();
    let pick = labels.findIndex(label => /^Play |^Cast |^Attack with|^Target Reader/.test(label));
    if (pick < 0) pick = labels.findIndex(label => /^Pass priority/.test(label));
    await actions.nth(Math.max(pick, 0)).focus();
    await page.keyboard.press('Enter');
    await expect.poll(async () => Number(await page.locator('main').getAttribute('data-update-seq'))).toBeGreaterThan(seq);
  }
  throw new Error('The game did not end within 400 decisions.');
}

const carriesAnalysis = (message: Record<string, unknown>): boolean =>
  message.type === 'analysis' || 'analysis' in message || 'analysis' in ((message.update ?? {}) as object);

test('a game reviewed against the policy, counted in Games, and played on', async ({ page }) => {
  test.setTimeout(240_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const messages = received(page);
  await page.goto(study());
  await playToEnd(page);
  expect(messages.filter(carriesAnalysis)).toEqual([]);

  await page.getByTestId('game-result-recording').click();
  await expect(page).toHaveURL(/\/replay\?trace=/);
  const marks = page.getByTestId('move-mark');
  await expect(marks.first()).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('analysis-status')).toHaveCount(0, { timeout: 60_000 });

  // (a) Your hand stays yours while the opponent acts after your first move.
  for (let step = 0; step < 6; step++) {
    await expect(page.getByRole('region', { name: /^Your hand, [1-9]/ })).toBeVisible();
    await page.getByRole('button', { name: 'Next', exact: true }).click();
  }
  // (b) The opponent's moves are named.
  const log = page.getByTestId('game-log');
  await expect(log).toContainText('Reader');
  await expect(log).not.toContainText('Opponent decision');

  // (c) Your choice reads against the analysis, your move marked.
  await page.locator('[data-testid=log-entry]', { has: marks }).first().getByRole('button').click();
  await expect(page.getByTestId('choices-heading')).toContainText('Analysis by Reader');
  await expect(page.getByTestId('offer-choosers').filter({ hasText: 'you' })).toHaveCount(1);

  // (d) One line on a fixed 0–100 scale, a point per analysed decision.
  const value = page.getByTestId('strip-value');
  await expect(page.getByTestId('strip-midline')).toHaveCount(1);
  const points = Number(await value.getAttribute('data-points'));
  expect(points).toBeGreaterThan(0);
  await expect(page.getByTestId('strip-caption')).toHaveText(`Reader's expected score from your seat · ${points} of ${points} decisions`);

  // (f) The summary's agreement is the count of "same" marks, and nothing grades you.
  await page.getByTestId('game-menu').click();
  const agreement = page.getByTestId('agreement');
  await expect(agreement).toContainText(`Decisions with a choice: ${points} ·`);
  // A move's mark counts each of its decisions (a cast, its target, its payment).
  const tallies = await marks.evaluateAll(nodes => nodes.map(node => [Number(node.getAttribute('data-agreed')), Number(node.getAttribute('data-of'))]));
  expect(tallies.reduce((sum, [, of]) => sum + of, 0)).toBe(points);
  const same = tallies.reduce((sum, [agreed]) => sum + agreed, 0);
  await expect(agreement).toContainText(`would: ${same} of ${points}`);
  await expect(page.locator('body')).not.toContainText(/mistake|blunder|inaccura|accuracy|brilliant/i);
  await page.screenshot({ path: shot('study-review') });
  await page.getByTestId('game-menu').click();

  // (e) Show both hands turns the opponent's hand up.
  await page.getByLabel('Show both hands').check();
  await expect(page.getByTestId('shown-hand').first()).toBeVisible();

  // Games: the row's turns, and the record's one game.
  await page.goto('/games?scope=mine');
  await expect(page.getByTestId('game-history-numbers').first()).toContainText(/\d+ turns?/);
  await expect(page.getByTestId('my-record-row')).toHaveCount(1);
  await expect(page.getByTestId('my-record-row')).toContainText('1');
  await page.getByTestId('game-history-row').first().getByRole('link', { name: /^Replay game/ }).click();
  await expect(marks.first()).toBeVisible({ timeout: 30_000 });

  // Play on from your last analysed decision: the board is that decision, now live.
  const last = page.locator('[data-testid=log-entry]', { has: marks }).last();
  await last.getByRole('button').click();
  await expect(last.getByRole('button')).toHaveAttribute('aria-current', 'true');
  await expect(page.getByTestId('turn-number')).not.toHaveText(/^Turn 0/);
  const turn = await page.getByTestId('turn-number').textContent();
  await page.getByTestId('play-on').click();
  await expect(page.getByTestId('played-from-origin')).toBeVisible();
  await page.getByTestId('start-played-from').click();
  await expect(page.getByTestId('action-option').first()).toBeEnabled({ timeout: 30_000 });
  await expect(page.getByTestId('turn-number')).toHaveText(turn ?? '');
  await page.screenshot({ path: shot('study-played-on') });
  await playToEnd(page);
  await page.getByTestId('game-result-recording').click();
  await expect(page.getByTestId('played-from')).toContainText('Played on from');
  await page.goto('/games?scope=mine');
  await expect(page.getByTestId('game-history-numbers').first()).toContainText('played on from another game');
  await expect(page.getByTestId('my-record-row')).toHaveCount(1);
  expect(errors).toEqual([]);
});

test('a decision made with analysis shown says so in Review and Games', async ({ page }) => {
  test.setTimeout(240_000);
  const messages = received(page);
  await page.goto(study());
  const toggle = page.getByTestId('show-analysis');
  let shown = false;
  await playToEnd(page, async decision => {
    if (decision === 2) {
      await toggle.check();
      await expect(page.getByTestId('choices-heading')).toContainText('Analysis by Reader', { timeout: 30_000 });
      shown = true;
    } else if (shown) {
      await toggle.uncheck();
      await expect(page.getByTestId('choices-heading')).not.toContainText('Analysis by');
      shown = false;
    }
  });
  expect(messages.some(message => message.type === 'analysis')).toBe(true);

  await page.getByTestId('game-result-recording').click();
  await expect(page.getByTestId('analysis-shown')).toHaveCount(1, { timeout: 30_000 });
  await page.goto('/games?scope=mine');
  await expect(page.getByTestId('game-history-numbers').first()).toContainText('analysis shown at 1 decision');
  await expect(page.getByTestId('my-record-with-analysis')).toHaveText('1');
});
