import { expect, test, type Page } from '@playwright/test';
import type { ExperienceFrame } from '../src/lib/types';
import { finishBoardNarration } from './presentation-readiness';

declare global {
  interface Window { __triggerFrame: ExperienceFrame | null }
}

async function track(page: Page): Promise<void> {
  await page.addInitScript(() => {
    window.__triggerFrame = null;
    const Native = window.WebSocket;
    class FixtureSocket extends Native {
      constructor(url: string | URL, protocols?: string | string[]) {
        super(url, protocols);
        this.addEventListener('message', event => {
          const payload = JSON.parse(String(event.data));
          const frame = payload.recovery?.frame ?? payload.update?.frame ?? payload.frame;
          if (frame) window.__triggerFrame = frame;
        });
      }
      send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
        if (typeof data === 'string') {
          const message = JSON.parse(data);
          if (message.type === 'new_game') {
            message.config = { ...message.config, trigger_order_fixture: true };
            data = JSON.stringify(message);
          }
        }
        super.send(data);
      }
    }
    window.WebSocket = FixtureSocket;
  });
}

async function choose(page: Page, label: string): Promise<void> {
  const frame = await page.evaluate(() => window.__triggerFrame!);
  const offer = frame.offers.find(offer => offer.label.includes(label));
  expect(offer, label).toBeDefined();
  const button = page.locator(`[data-testid="action-option"][data-offer-id="${offer!.id}"]`);
  await button.focus();
  await page.keyboard.press('Enter');
  await expect.poll(() => page.evaluate(() => window.__triggerFrame?.revision)).toBeGreaterThan(frame.revision);
  await finishBoardNarration(page);
}

for (const first of ['Callous Inspector', 'Bastion of Remembrance']) {
  test(`controller chooses ${first} first and recovers the actual order decision`, async ({ page }) => {
    await track(page);
    await page.goto('/');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    await page.getByRole('button', { name: 'New Game', exact: true }).click();
    await expect(page.getByTestId('decision-surface')).toBeVisible();
    await choose(page, 'Cast Deadly Dispute');
    await choose(page, 'Sacrifice Callous Inspector');
    await expect.poll(() => page.evaluate(() => window.__triggerFrame?.action_space)).toBe('ORDER_TRIGGERS');
    await expect(page.getByTestId('decision-context')).toContainText('Later placements resolve first');
    const orderButtons = page.getByTestId('action-option');
    await expect(orderButtons.filter({ hasText: 'Callous Inspector' })).toContainText('Deal 1 damage to you. Create 1 Clue token');
    await expect(orderButtons.filter({ hasText: 'Callous Inspector' })).not.toContainText('Menace');
    await expect(orderButtons.filter({ hasText: 'Bastion of Remembrance' })).toContainText('Each opponent loses 1 life and you gain 1 life');
    await expect(orderButtons.filter({ hasText: 'Bastion of Remembrance' })).not.toContainText('enters the battlefield');
    await expect(orderButtons.first()).toContainText('Put on stack next');
    const hash = await page.evaluate(() => window.__triggerFrame!.frame_hash);
    await page.reload();
    await expect.poll(() => page.evaluate(() => window.__triggerFrame?.frame_hash)).toBe(hash);
    await expect(page.getByTestId('action-option')).toHaveCount(2);
    await choose(page, first);
    for (let count = 0; count < 4; count++) {
      const frame = await page.evaluate(() => window.__triggerFrame!);
      if (!frame.offers.some(offer => offer.label === 'Pass priority')) break;
      await choose(page, 'Pass priority');
      const projection = await page.evaluate(() => window.__triggerFrame!.projection);
      if (projection.game_over || projection.agent.life === 0) break;
      // Both trigger resolutions precede the Dispute still on the stack.
      if (count === 1) break;
    }
    const projection = await page.evaluate(() => window.__triggerFrame!.projection);
    expect(projection.agent.life).toBe(first === 'Callous Inspector' ? 1 : 0);
  });
}
