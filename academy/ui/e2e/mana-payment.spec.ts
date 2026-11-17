import { expect, test, type Page } from '@playwright/test';
import type { Command, ExperienceFrame } from '../src/lib/types';
import { finishBoardNarration } from './presentation-readiness';

declare global {
  interface Window { __mana: { frame: ExperienceFrame | null; commands: Command[] } }
}
async function track(page: Page): Promise<void> {
  await page.addInitScript(() => {
    window.__mana = { frame: null, commands: [] };
    const Native = window.WebSocket;
    class PaymentSocket extends Native {
      constructor(url: string | URL, protocols?: string | string[]) {
        super(url, protocols);
        this.addEventListener('message', event => {
          const payload = JSON.parse(String(event.data));
          const frame = payload.recovery?.frame ?? payload.update?.frame ?? payload.frame;
          if (frame) window.__mana.frame = frame;
        });
      }
      send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
        if (typeof data === 'string') {
          const message = JSON.parse(data);
          if (message.type === 'new_game') {
            message.config = { ...message.config, payment_fixture: true };
            data = JSON.stringify(message);
          }
          if (message.type === 'command') window.__mana.commands.push(message.command);
        }
        super.send(data);
      }
    }
    window.WebSocket = PaymentSocket;
  });
}
async function startPaymentGame(page: Page): Promise<void> {
  await track(page);
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('decision-surface')).toBeVisible();
  await finishBoardNarration(page);
}
async function choose(page: Page, id: number): Promise<void> {
  const before = await page.evaluate(() => window.__mana.frame!.revision);
  await page.locator(`[data-testid="action-option"][data-offer-id="${id}"]`).focus();
  await page.keyboard.press('Enter');
  await expect.poll(() => page.evaluate(() => window.__mana.frame?.revision), { timeout: 20_000 }).toBeGreaterThan(before);
  await finishBoardNarration(page);
}
for (const [width, height] of [[1280, 800], [1440, 900]]) {
  test(`preview and choose native payments at ${width}`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height });
    await startPaymentGame(page);
    const produce = page.getByTestId('action-option').filter({hasText: 'White Lotus Hideout'}).first();
    await produce.click();
    await page.getByRole('button', {name: /Add \{U\} .*Lesson/}).click();
    const floating = page.getByTestId('mana-pool').last();
    await floating.locator('summary').click();
    await expect(floating).toContainText('Lesson');
    await expect(floating).toContainText('step end');
    await floating.locator('summary').click();
    // The preceding Hideout activation brought the last page of lands into view.
    await page.getByRole('button', {name: 'Previous lands'}).click();
    const commands = await page.evaluate(() => window.__mana.commands.length);
    const cast = page.getByRole('button', { name: 'Cast Lightning Strike', exact: true });
    const panel = page.getByTestId('payment-preview');
    await cast.focus();
    await expect(panel).toContainText('Would tap');
    await expect(panel).toContainText('Lesson');
    await expect(page.locator('[data-payment-source]')).toHaveCount(2);
    expect(await page.evaluate(() => window.__mana.commands.length)).toBe(commands);
    await cast.press('Enter');
    await expect(page.getByTestId('action-panel')).toHaveAttribute('data-action-space-kind', 'CHOOSE_TARGET');
    const target = await page.evaluate(() => window.__mana.frame!.offers.find(o => o.label === 'Target Villain')!.id);
    await choose(page, target);
    const pay = page.getByTestId('action-option').filter({hasText: 'Pay the cost'});
    await expect(pay).toBeFocused();
    await expect(panel).toContainText('Would tap');
    for (const colorScheme of ['dark', 'light'] as const) {
      await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
      await expect(panel.getByRole('list', { name: 'Would tap' })).toBeVisible();
      await expect(page.getByTestId('action-option')).toHaveCount(9);
      await expect(page.locator('[data-payment-source]').first()).toBeInViewport();
      await expect(pay).toBeInViewport({ ratio: 1 });
      await expect(page.getByTestId('action-option').filter({hasText: 'Tap Island'}).first()).toBeInViewport({ ratio: 1 });
      await expect(page.getByText('History and settings', {exact: true})).toBeInViewport({ ratio: 1 });
      const leftOver = panel.getByText(/^Left over/);
      const panelBox = await panel.boundingBox();
      const remainderBox = await leftOver.boundingBox();
      expect(remainderBox!.y + remainderBox!.height).toBeLessThanOrEqual(panelBox!.y + panelBox!.height);
      await page.screenshot({path: testInfo.outputPath(`payment-${width}-${colorScheme}.png`)});
    }
    const paymentHash = await page.evaluate(() => window.__mana.frame!.frame_hash);
    await page.reload();
    await expect.poll(() => page.evaluate(() => window.__mana.frame?.frame_hash)).toBe(paymentHash);
    await expect(panel).toContainText('Would tap');
    const activate = page.getByTestId('action-option').filter({hasText: 'Activate ability: Prophetic Prism'});
    await activate.focus();
    await expect(panel).toContainText('Focus or point to a payment choice.');
    await expect(page.locator('[data-payment-source]')).toHaveCount(0);
    const tap = page.getByTestId('action-option').filter({hasText: 'Tap Island'}).first();
    await tap.focus();
    await expect(panel.getByRole('listitem').first()).toContainText('Island');
    if (width === 1280) {
      await tap.press('Enter');
    } else {
      // Hold an authentic answer for the old offer until after the source commits.
      const other = page.getByTestId('action-option').filter({hasText: 'Tap Mountain'}).first();
      let release!: () => void;
      const hold = new Promise<void>(resolve => { release = resolve; });
      let arrived!: () => void;
      const waiting = new Promise<void>(resolve => { arrived = resolve; });
      await page.route('**/payment-preview', async route => {
        const response = await route.fetch();
        arrived();
        await hold;
        await route.fulfill({response});
      }, {times: 1});
      await other.hover();
      await waiting;
      await tap.click();
      await expect.poll(() => page.evaluate(() => window.__mana.frame!.frame_hash)).not.toBe(paymentHash);
      release();
    }
    await expect.poll(() => page.evaluate(() => window.__mana.frame?.projection.agent.mana?.[1])).toBe(1);
    await expect(pay).toBeFocused();
    await expect(page.locator('[data-payment-source]')).toHaveCount(1);
    const source = await page.locator('[data-payment-source]').getAttribute('data-object-id');
    await pay.click();
    await expect(page.locator(`[data-object-id="${source}"]`)).toHaveAttribute('aria-label', /tapped/);
    const afterPayment = await page.evaluate(() => window.__mana.frame!);
    if (!afterPayment.projection.game_over) {
      const pass = afterPayment.offers.find(o => o.label === 'Pass priority')!;
      await choose(page, pass.id);
    }
    await expect.poll(() => page.evaluate(() => window.__mana.frame?.projection.game_over)).toBe(true);
    const requests: string[] = [];
    page.on('request', request => { if (request.url().endsWith('/payment-preview') && new URL(request.frame().url()).pathname === '/replay') requests.push(request.url()); });
    await page.getByRole('link', {name: 'Open replay', exact: true}).click();
    await expect(page.getByRole('slider', { name: 'Replay position' })).toBeVisible();
    await expect(panel).toHaveCount(0);
    expect(requests).toEqual([]);
  });
}
