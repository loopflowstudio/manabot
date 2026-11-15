import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

interface Addresses { play: string; timeout: string; watch: string; watch_timeout: string;
  pending_watch: { id: string; token: string };
  saved: string; saved_ordinal: number; ids: Record<string, string>; }
const proof = new URL('../test-results/mirror/addresses.json', import.meta.url);
for (const theme of ['light', 'dark'] as const) {
  test(`mirror Play, Watch, reload and saved evidence (${theme})`, async ({ page }) => {
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
    await expect(page.getByText(/mirror search 1×1/).first()).toBeVisible();
    const before = frames.at(-1)!;
    await page.reload();
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    expect(frames.at(-1)!.match_id).toBe(before.match_id);
    expect(frames.at(-1)!.revision).toBe(before.revision);
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/mirror/play-${theme}.png`, import.meta.url)) });

    await page.goto(addresses.watch);
    await expect(page.getByRole('combobox', { name: 'Seat 1 search', exact: true })).toHaveValue('mirror');
    await page.getByRole('combobox', { name: 'Seat 1 search', exact: true }).selectOption('none');
    await page.getByRole('combobox', { name: 'Seat 1 search', exact: true }).selectOption('mirror');
    await expect(page.getByRole('spinbutton', { name: 'Seat 1 search worlds', exact: true })).toHaveValue('1');
    await expect(page.getByRole('spinbutton', { name: 'Seat 1 search depth', exact: true })).toHaveValue('1');
    await page.getByRole('button', { name: 'Start watching' }).click();
    await expect(page.getByRole('columnheader', { name: 'Raw policy', exact: true })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Updated policy', exact: true })).toBeVisible();
    await expect(page.getByText('Uniform belief (physical deal):', { exact: false })).toBeVisible();
    await page.reload();
    await expect(page.getByRole('columnheader', { name: 'Raw policy', exact: true })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
    await page.getByRole('columnheader', { name: 'Raw policy', exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/mirror/watch-${theme}.png`, import.meta.url)) });

    await page.goto(addresses.saved);
    await page.getByRole('link', { name: 'Inspect bot search', exact: true }).click();
    await expect(page.getByText(new RegExp(`^Decision ${addresses.saved_ordinal + 1} of`))).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Updated policy', exact: true })).toBeVisible();
    await expect(page.getByText('Recorded Play game', { exact: true })).toBeVisible();
    await page.getByRole('columnheader', { name: 'Updated policy', exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/mirror/replay-${theme}.png`, import.meta.url)) });
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
    await page.getByTestId('action-option').filter({ hasText: 'Pass priority' }).first().click();
  }
  await expect(page.getByRole('button', { name: 'Retry search', exact: true })).toBeVisible();
  const pending = frames.at(-1)!;
  await page.reload();
  await expect(page.getByRole('button', { name: 'Retry search', exact: true })).toBeVisible();
  expect(frames.at(-1)!.match_id).toBe(pending.match_id);
  expect(frames.at(-1)!.revision).toBe(pending.revision);
  await page.screenshot({ path: fileURLToPath(new URL('../test-results/mirror/timeout.png', import.meta.url)) });
  await page.getByRole('button', { name: 'Retry search', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Retry search', exact: true })).toBeEnabled();
  expect(frames.at(-1)!.revision).toBe(pending.revision);
  await page.goto(addresses.watch_timeout);
  await page.getByRole('button', { name: 'Start watching' }).click();
  await expect(page.getByRole('button', { name: 'Retry search', exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Retry search', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Retry search', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Retry search', exact: true })).toBeEnabled();
});

test('Watch reload restores latest timeout and explicit retry while history stays navigable', async ({ page }) => {
  const { pending_watch: watch } = JSON.parse(readFileSync(proof, 'utf8')) as Addresses;
  await page.addInitScript(watch => localStorage.setItem(`etude.watch.${watch.id}`, watch.token), watch);
  const steps: number[] = [];
  page.on('request', request => {
    if (request.url().endsWith(`/api/watch/${watch.id}/step`)) steps.push(request.postDataJSON().revision);
  });
  const url = `/replay?watch=1&trace=${watch.id}`;
  await page.goto(url);
  await expect(page.getByText(/^Decision 3 of 3/)).toBeVisible();
  const retry = page.getByRole('button', { name: 'Retry search', exact: true });
  await expect(retry).toBeEnabled();
  await expect(page.getByRole('alert')).toContainText(/time|deadline/i);
  await page.reload();
  await expect(page.getByText(/^Decision 3 of 3/)).toBeVisible();
  await expect(retry).toBeEnabled();
  await expect(page.getByRole('alert')).toContainText(/time|deadline/i);
  await page.getByRole('button', { name: 'Previous decision', exact: true }).click();
  await expect(page.getByText(/^Decision 2 of 3/)).toBeVisible();
  await page.getByRole('button', { name: 'Step decision', exact: true }).click();
  await expect(page.getByText(/^Decision 3 of 3/)).toBeVisible();
  await expect(retry).toBeEnabled();
  // An explicit historical URL remains historical, even with a pending retry.
  await page.goto(`${url}&decision=0`);
  await expect(page.getByText(/^Decision 1 of 3/)).toBeVisible();
  await page.reload();
  await expect(page.getByText(/^Decision 1 of 3/)).toBeVisible();
  expect(steps).toEqual([]);
  await page.goto(url);
  await expect(retry).toBeEnabled();
  await retry.click();
  await expect(page.getByText(/^Decision 4 of 4/)).toBeVisible();
  expect(steps).toEqual([3]);
  await expect(retry).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('the waiting line remains legible while a real response awaits delivery', async ({ page }) => {
  const addresses = JSON.parse(readFileSync(proof, 'utf8')) as Addresses;
  const held: (() => void)[] = [];
  let holdNext = true;
  // Hold delivery, not server execution: no extra compute or resource pressure
  // is needed to inspect the actual waiting state and elapsed-time indicator.
  await page.routeWebSocket('**/ws/play', socket => {
    const server = socket.connectToServer();
    server.onMessage(message => {
      const value = JSON.parse(String(message));
      if (holdNext && value.type === 'command_outcome' && value.status === 'accepted') {
        holdNext = false;
        held.push(() => socket.send(message));
      } else socket.send(message);
    });
  });
  try {
    await page.goto(addresses.play);
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    await page.getByTestId('action-option').first().click();
    await expect.poll(() => held.length).toBe(1);
    await expect(page.getByTestId('mirror-thinking')).toContainText('Mirror search 1×1');
    await expect(page.getByTestId('mirror-thinking')).toContainText(/\d+\.\d s/);
    await page.screenshot({ path: fileURLToPath(new URL('../test-results/mirror/thinking.png', import.meta.url)) });
    held.shift()?.();
    await expect(page.getByTestId('mirror-thinking')).toHaveCount(0);
  } finally {
    held.splice(0).forEach(send => send());
  }
});
