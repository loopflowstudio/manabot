import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

interface Addresses { play: string; timeout: string; watch: string; watch_timeout: string;
  pending_watch: { id: string; token: string };
  saved: string; saved_ordinal: number; ids: Record<string, string>; }
const proof = new URL('../test-results/search/addresses.json', import.meta.url);
for (const theme of ['light', 'dark'] as const) {
  test(`Search Play, Watch, reload and saved evidence (${theme})`, async ({ page }) => {
    const addresses = JSON.parse(readFileSync(proof, 'utf8')) as Addresses;
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(theme => localStorage.setItem('etude.theme', theme), theme);
    await page.emulateMedia({ colorScheme: theme });
    const frames: { match_id: string; revision: number }[] = [];
    page.on('websocket', socket => socket.on('framereceived', event => {
      const message = JSON.parse(String(event.payload));
      const frame = message.update?.frame ?? message.recovery?.frame ?? message.frame;
      if (frame) frames.push(frame);
    }));
    await page.goto(addresses.play);
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    await expect(page.getByTestId('player-plate').getByText(/search 1×1/).first()).toBeVisible();
    const before = frames.at(-1)!;
    await page.reload();
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    expect(frames.at(-1)!.match_id).toBe(before.match_id);
    expect(frames.at(-1)!.revision).toBe(before.revision);
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/search/play-${theme}.png`, import.meta.url)) });

    await page.goto(addresses.watch);
    await expect(page.getByRole('combobox', { name: 'Seat 1 bot search', exact: true })).toHaveValue('search');
    await page.getByRole('combobox', { name: 'Seat 1 bot search', exact: true }).selectOption('none');
    await page.getByRole('combobox', { name: 'Seat 1 bot search', exact: true }).selectOption('search');
    await expect(page.getByRole('spinbutton', { name: 'Seat 1 bot search worlds', exact: true })).toHaveValue('1');
    await expect(page.getByRole('spinbutton', { name: 'Seat 1 bot search depth', exact: true })).toHaveValue('1');
    // A fresh deal can give the non-search seat the first decision; this seed
    // gives it to seat 1's search, whose evidence the next lines read.
    await page.getByLabel('Deal mode', { exact: true }).selectOption('repeat');
    await page.getByLabel('Seed', { exact: true }).fill('0');
    await page.getByRole('button', { name: 'Start watching' }).click();
    await page.getByRole('button', { name: 'Continue', exact: true }).click();
    // The opening decision has policy rows and no card subjects.
    const searchDetail = page.getByTestId('search-detail');
    await searchDetail.locator('summary').click();
    await expect(page.getByRole('columnheader', { name: 'Raw policy', exact: true })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Updated policy', exact: true })).toBeVisible();
    await expect(page.getByText('Uniform belief (physical deal):', { exact: false })).toBeVisible();
    await page.reload();
    await searchDetail.locator('summary').click();
    await expect(page.getByRole('columnheader', { name: 'Raw policy', exact: true })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
    await page.getByRole('columnheader', { name: 'Raw policy', exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/search/watch-${theme}.png`, import.meta.url)) });

    // Later card choices carry the searched policy on the board.
    const counter = page.getByTestId('position-counter');
    const cardPolicy = page.getByTestId('card-policy').first();
    for (let step = 0; step < 8 && !(await cardPolicy.isVisible()); step += 1) {
      const beforePosition = (await counter.textContent())!;
      await page.getByRole('button', { name: /^(Next|Continue)$/ }).click();
      await expect(counter).not.toHaveText(beforePosition);
    }
    await expect(cardPolicy).toBeVisible();

    await page.goto(addresses.saved);
    // The bot's first searched decision, reached with the transport's next-decision control.
    const perspective = page.getByTestId('replay-perspective');
    await expect(page.getByTestId('position-counter')).toBeVisible();
    for (let step = 0; step < 20 && !(await perspective.textContent())?.includes('chooser seat 2'); step += 1) {
      await page.getByRole('button', { name: 'Next decision', exact: true }).click();
      await page.waitForTimeout(150);
    }
    await expect(perspective).toContainText('chooser seat 2');
    await page.getByTestId('search-detail').locator('summary').click();
    await expect(page.getByRole('columnheader', { name: 'Updated policy', exact: true })).toBeVisible();
    await expect(page.getByTestId('recording-heading')).toHaveText('Recorded Play game');
    await page.getByRole('columnheader', { name: 'Updated policy', exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/search/replay-${theme}.png`, import.meta.url)) });
    expect(errors).toEqual([]);
  });
}

test('timeouts retain Play and initial Watch choices across reload and explicit retry', async ({ page }) => {
  const addresses = JSON.parse(readFileSync(proof, 'utf8')) as Addresses;
  const frames: { match_id: string; revision: number }[] = [];
  page.on('websocket', socket => socket.on('framereceived', event => {
    const message = JSON.parse(String(event.payload));
    const frame = message.update?.frame ?? message.recovery?.frame ?? message.frame;
    if (frame) frames.push(frame);
  }));
  await page.goto(addresses.timeout);
  const retry = page.getByRole('button', { name: 'Retry search', exact: true });
  await expect(retry.or(page.getByTestId('action-option').first()).first()).toBeVisible();
  if (await retry.count() === 0) {
    await page.getByTestId('action-option').filter({ hasText: 'Keep hand' }).first().click();
  }
  await expect(page.getByRole('button', { name: 'Retry search', exact: true })).toBeVisible();
  const pending = frames.at(-1)!;
  await page.reload();
  await expect(page.getByRole('button', { name: 'Retry search', exact: true })).toBeVisible();
  expect(frames.at(-1)!.match_id).toBe(pending.match_id);
  expect(frames.at(-1)!.revision).toBe(pending.revision);
  await page.screenshot({ path: fileURLToPath(new URL('../test-results/search/timeout.png', import.meta.url)) });
  await page.getByRole('button', { name: 'Retry search', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Retry search', exact: true })).toBeEnabled();
  expect(frames.at(-1)!.revision).toBe(pending.revision);
  await page.goto(addresses.watch_timeout);
  // A repeated deal whose first Watch decision belongs to the timed-out seat.
  await page.getByLabel('Deal mode').selectOption('repeat');
  await page.getByLabel('Seed').fill('1');
  await page.getByRole('button', { name: 'Start watching' }).click();
  await page.getByRole('button', { name: 'Continue', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Retry search', exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Retry search', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Retry search', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Retry search', exact: true })).toBeEnabled();
});

test('Watch reload retains timeout and exact position while history stays navigable', async ({ page }) => {
  const { pending_watch: watch } = JSON.parse(readFileSync(proof, 'utf8')) as Addresses;
  await page.addInitScript(watch => sessionStorage.setItem(`etude.watch.${watch.id}`, watch.token), watch);
  const steps: number[] = [];
  page.on('request', request => {
    if (request.url().endsWith(`/api/watch/${watch.id}/step`)) steps.push(request.postDataJSON().revision);
  });
  const url = `/replay?watch=1&trace=${watch.id}`;
  await page.goto(`${url}&position=latest`);
  const counter = page.getByTestId('position-counter');
  await expect(counter).toHaveText('Position 7 / 7');
  const retry = page.getByRole('button', { name: 'Retry search', exact: true });
  await expect(retry).toBeEnabled();
  await expect(page.getByRole('alert')).toContainText(/time|deadline/i);
  await page.reload();
  await expect(counter).toHaveText('Position 7 / 7');
  await expect(retry).toBeEnabled();
  await expect(page.getByRole('alert')).toContainText(/time|deadline/i);
  await page.getByRole('button', { name: 'Previous', exact: true }).click();
  await expect(counter).toHaveText('Position 6 / 7');
  await page.reload();
  await expect(counter).toHaveText('Position 6 / 7');
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(counter).toHaveText('Position 7 / 7');
  await expect(retry).toBeEnabled();
  // Reading the first choice never retries a pending search.
  await page.getByRole('slider', { name: 'Replay position' }).fill('0');
  await expect(counter).toHaveText('Position 1 / 7');
  await page.reload();
  await expect(counter).toHaveText('Position 1 / 7');
  expect(steps).toEqual([]);
  await page.goto(`${url}&position=latest`);
  await expect(retry).toBeEnabled();
  await retry.click();
  await expect(counter).toHaveText('Position 7 / 9');
  expect(steps).toEqual([3]);
  await expect(retry).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('the waiting line remains legible while a real response awaits delivery', async ({ page }) => {
  const addresses = JSON.parse(readFileSync(proof, 'utf8')) as Addresses;
  const held: (() => void)[] = [];
  let armed = false;
  let holding = false;
  // Hold delivery, not server execution: no extra compute or resource pressure
  // is needed to inspect the actual waiting state and elapsed-time indicator.
  await page.routeWebSocket('**/ws/play', socket => {
    const server = socket.connectToServer();
    server.onMessage(message => {
      const value = JSON.parse(String(message));
      // Deliver the real search start, then retain every following message in
      // order so a quick completion cannot replace the state under inspection.
      if (holding) {
        held.push(() => socket.send(message));
      } else {
        socket.send(message);
        if (armed && value.type === 'work_status' && value.work.kind === 'search' && value.work.state === 'running') holding = true;
      }
    });
  });
  try {
    await page.goto(addresses.play);
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    armed = true;
    await page.getByTestId('action-option').first().click();
    await expect.poll(() => holding).toBe(true);
    // The rail names who is searching and for how long, once.
    await expect(page.getByTestId('work-status')).toContainText('is searching');
    await expect(page.getByTestId('work-status')).toContainText(/· \d+ s/);
    await expect(page.getByTestId('work-status')).toHaveCount(1);
    await page.screenshot({ path: fileURLToPath(new URL('../test-results/search/thinking.png', import.meta.url)) });
    armed = false;
    holding = false;
    held.splice(0).forEach(send => send());
    await expect(page.getByTestId('work-status')).toHaveCount(0);
  } finally {
    armed = false;
    holding = false;
    held.splice(0).forEach(send => send());
  }
});
