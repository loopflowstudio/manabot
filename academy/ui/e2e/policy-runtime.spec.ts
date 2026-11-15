import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const proof = new URL('../test-results/policy/addresses.json', import.meta.url);
for (const theme of ['light', 'dark'] as const) {
  test(`independent policy Play, reload, and both Watch seats (${theme})`, async ({ page }) => {
    const addresses = JSON.parse(readFileSync(proof, 'utf8')) as {
      refused: string; play: string; watch: string; ids: { independent: string; baseline: string; "refused-config": string }
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
    await page.goto(addresses.refused);
    await expect(page.getByTestId('checkpoint-select')).toHaveValue(`policy:${addresses.ids['refused-config']}`);
    await expect(page.getByText(/restore the pinned artifact and runner/).first()).toBeVisible();
    await page.getByRole('button', { name: 'New Game', exact: true }).click();
    await expect(page.getByTestId('checkpoint-select')).toHaveValue(`policy:${addresses.ids['refused-config']}`);
    await expect(page.getByTestId('action-option')).toHaveCount(0);
    const preview = new URL(addresses.play);
    preview.search = new URLSearchParams({opponent: addresses.ids.baseline, hero: 'interactive', villain: 'interactive'}).toString();
    await page.goto(preview.href);
    await expect(page.getByTestId('policy-compatibility')).toContainText('was trained on');
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/policy/play-incompatible-${theme}.png`, import.meta.url)) });
    await page.goto(addresses.play);
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    const before = frames.at(-1)!;
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/policy/play-${theme}.png`, import.meta.url)) });
    await page.reload();
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    expect(frames.at(-1)!.match_id).toBe(before.match_id);
    expect(frames.at(-1)!.revision).toBe(before.revision);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
    // Reopen an exact selection after a server's runner has changed. Only the
    // runner differs from the retained descriptor; the ID must stay stale until clicked.
    const priorId = 'f'.repeat(64);
    await page.evaluate(async ({ id, priorId }) => {
      const catalog = await (await fetch('/api/opponents')).json();
      const current = catalog.opponents.find((p: { sha256: string }) => p.sha256 === id).policy;
      localStorage.setItem('etude.policy.selections', JSON.stringify([
        { ...current, id: priorId, runner: 'previous-runner' },
      ]));
    }, { id: addresses.ids.independent, priorId });
    const stale = new URL(addresses.watch);
    stale.searchParams.set('seat1', `policy:${priorId}`);
    await page.goto(stale.href);
    await page.reload();
    await expect(page.getByTestId('checkpoint-select').nth(0)).toHaveValue(`policy:${priorId}`);
    await expect(page.getByText('The runner changed since this was selected.', { exact: false })).toBeVisible();
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/policy/watch-reselect-${theme}.png`, import.meta.url)) });
    await page.getByRole('button', { name: /^Reselect Independent/ }).focus();
    await page.keyboard.press('Enter');
    const selections = page.getByTestId('checkpoint-select');
    await expect(selections).toHaveCount(2);
    await expect(selections.nth(0)).toHaveValue(`policy:${addresses.ids.independent}`);
    await expect(selections.nth(1)).toHaveValue(`policy:${addresses.ids.baseline}`);
    await expect(selections.nth(0)).toContainText('independent');
    await page.getByText('Other decks', { exact: true }).click();
    await page.getByTestId('browse-interactive').click();
    await page.getByTestId('assign-deck-villain').click();
    await expect(page.getByTestId('policy-compatibility').filter({ hasText: 'was trained on' })).toHaveCount(1);
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/policy/watch-incompatible-${theme}.png`, import.meta.url)) });
    await page.getByTestId('browse-gw_allies').click();
    await page.getByTestId('assign-deck-villain').click();
    await expect(page.getByTestId('policy-compatibility')).toHaveCount(0);
    await selections.nth(0).focus();
    await page.keyboard.press('Tab');
    await expect(page.getByText(/Mirror search unavailable/).first()).toBeVisible();
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/policy/watch-setup-${theme}.png`, import.meta.url)) });
    await page.getByRole('button', { name: 'Start watching' }).click();
    await expect(page.getByText(/^Decision 1 of 1 ·/)).toBeVisible();
    await page.getByRole('button', { name: 'Step decision', exact: true }).click();
    await expect(page.getByText(/^Decision 2 of 2 ·/)).toBeVisible();
    await page.reload();
    await expect(page.getByText(/^Decision 2 of 2 ·/)).toBeVisible();
    await page.getByRole('button', { name: 'Previous decision', exact: true }).click();
    await expect(page.getByText(/^Decision 1 of 2 ·/)).toBeVisible();
    await page.screenshot({ path: fileURLToPath(new URL(`../test-results/policy/watch-${theme}.png`, import.meta.url)) });
    const status = page.locator('details.player-status').first();
    await status.locator('summary').click();
    await expect(status).toContainText('Runner:');
    await expect(status).toContainText('Policy ID:');
    await expect(status).not.toContainText('SHA-256:');
    expect(errors).toEqual([]);
  });
}
