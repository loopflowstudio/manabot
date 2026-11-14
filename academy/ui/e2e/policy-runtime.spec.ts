import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const proof = new URL('../test-results/policy/addresses.json', import.meta.url);
for (const theme of ['light', 'dark'] as const) {
  test(`independent policy Play, reload, and both Watch seats (${theme})`, async ({ page }) => {
    const addresses = JSON.parse(readFileSync(proof, 'utf8')) as {
      play: string; watch: string; ids: { independent: string; baseline: string }
    };
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(theme => localStorage.setItem('etude.theme', theme), theme);
    await page.emulateMedia({ colorScheme: theme });
    const frames: { match_id: string; revision: number }[] = [];
    page.on('websocket', socket => socket.on('framereceived', event => {
      const message = JSON.parse(String(event.payload));
      const frame = message.recovery?.frame ?? message.frame;
      if (frame) frames.push(frame);
    }));
    await page.goto(addresses.play);
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    const before = frames.at(-1)!;
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/policy/play-${theme}.png`, import.meta.url)) });
    await page.reload();
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    expect(frames.at(-1)!.match_id).toBe(before.match_id);
    expect(frames.at(-1)!.revision).toBe(before.revision);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
    await page.goto(addresses.watch);
    const selections = page.getByTestId('checkpoint-select');
    await expect(selections).toHaveCount(2);
    await expect(selections.nth(0)).toHaveValue(`policy:${addresses.ids.independent}`);
    await expect(selections.nth(1)).toHaveValue(`policy:${addresses.ids.baseline}`);
    await selections.nth(0).focus();
    await page.keyboard.press('Tab');
    await expect(page.getByText(/Mirror search unavailable/).first()).toBeVisible();
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/policy/watch-setup-${theme}.png`, import.meta.url)) });
    await page.getByRole('button', { name: 'Start watching' }).click();
    await expect(page.getByText(/^Decision 1 of 1 ·/)).toBeVisible();
    await page.getByRole('button', { name: 'Step decision', exact: true }).click();
    await expect(page.getByText(/^Decision 2 of 2 ·/)).toBeVisible();
    await page.reload();
    await expect(page.getByText(/^Decision 1 of 2 ·/)).toBeVisible();
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/policy/watch-${theme}.png`, import.meta.url)) });
    const status = page.locator('details.player-status').first();
    await status.locator('summary').click();
    await expect(status).toContainText('Runner:');
    expect(errors).toEqual([]);
  });
}
