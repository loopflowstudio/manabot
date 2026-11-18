import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';
import type { Command, ExperienceFrame } from '../src/lib/types';

test.setTimeout(300_000);

declare global {
  interface Window {
    __surface: { frame: ExperienceFrame | null; commands: Command[]; outcomes: string[]; socket: WebSocket | null };
  }
}

async function track(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const state: Window['__surface'] = { frame: null, commands: [], outcomes: [], socket: null };
    const Native = window.WebSocket;
    function findFrame(value: unknown): ExperienceFrame | null {
      if (!value || typeof value !== 'object') return null;
      if ('frame_hash' in value && 'projection' in value) return value as ExperienceFrame;
      for (const child of Object.values(value)) {
        const frame = findFrame(child);
        if (frame) return frame;
      }
      return null;
    }
    class TrackedSocket extends Native {
      constructor(url: string | URL, protocols?: string | string[]) {
        super(url, protocols);
        state.socket = this;
        this.addEventListener('message', event => {
          const payload: unknown = JSON.parse(String(event.data));
          const frame = findFrame(payload);
          if (frame) state.frame = frame;
          if (payload && typeof payload === 'object' && 'status' in payload && typeof payload.status === 'string') state.outcomes.push(payload.status);
        });
      }
      send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
        if (typeof data === 'string') {
          const message = JSON.parse(data) as { type: string; command?: Command };
          if (message.type === 'command' && message.command) state.commands.push(message.command);
        }
        super.send(data);
      }
    }
    window.__surface = state;
    window.WebSocket = TrackedSocket;
  });
}

async function choose(page: Page, id: number, via?: Locator): Promise<void> {
  const revision = await page.evaluate(() => window.__surface.frame!.revision);
  const target = via ?? page.locator(`[data-testid="action-option"][data-offer-id="${id}"]`);
  await expect(page.getByTestId('table')).toHaveAttribute('data-live', 'true');
  await target.click();
  await expect.poll(() => page.evaluate(() => window.__surface.frame?.revision)).toBeGreaterThan(revision);
  const next = await page.evaluate(() => window.__surface.frame!.revision);
  await expect(page.locator('main')).toHaveAttribute('data-frame-revision', String(next));
}

async function start(page: Page, demo: string): Promise<void> {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await track(page);
  await page.goto(`/?demo=${demo}`);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', { timeout: 30_000 });
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('table')).toBeVisible({ timeout: 120_000 });
}

test('Precision on the table: cards act, inspecting commits nothing, recovery and retry', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1500, height: 1100 });
  await start(page, 'precision');
  const board = (await page.getByTestId('game-board').boundingBox())!;
  const rail = (await page.getByRole('complementary', { name: 'Stack, decision and legal choices' }).boundingBox())!;
  expect(rail.x).toBeGreaterThanOrEqual(board.x + board.width - 1);
  expect(board.y + board.height).toBeLessThanOrEqual(1100);
  await expect(page.locator('details[open]')).toHaveCount(0);
  const mine = page.getByRole('region', { name: 'Your battlefield', exact: true });
  for (const name of ['Treasure', 'Swamp', 'Callous Inspector']) await expect(mine.locator(`[data-card-name="${name}"]`).first()).toBeVisible();

  // Resting on a card opens Inspect and sends nothing.
  const before = await page.evaluate(() => ({ hash: window.__surface.frame!.frame_hash, count: window.__surface.commands.length }));
  await page.locator('.hand [data-card-name="Deadly Precision"]').first().hover();
  await expect(page.getByTestId('card-preview')).toContainText('Deadly Precision');
  expect(await page.evaluate(() => window.__surface.commands.length)).toBe(before.count);
  expect(await page.evaluate(() => window.__surface.frame!.frame_hash)).toBe(before.hash);
  await page.mouse.move(4, 4);

  const cast = await page.evaluate(() => window.__surface.frame!.offers.find(offer => offer.label === 'Cast Deadly Precision')!.id);
  await choose(page, cast);
  // The target is the creature itself: click it on the board.
  const target = await page.evaluate(() => window.__surface.frame!.offers.find(offer => offer.subjects?.some(subject => subject.object.label === 'Callous Inspector'))!);
  const victim = target.subjects![0].object.entity_id;
  await expect(page.getByTestId('stack-entry').filter({ hasText: 'Deadly Precision' })).toBeVisible();
  await choose(page, target.id, page.locator(`.board [data-testid=table-card][data-object-id="${target.subjects![0].object.display_id}"]`).first());
  const sacrifice = await page.evaluate(entity => window.__surface.frame!.offers.find(offer => offer.action_type === 'SELECT_CARD' && offer.subjects?.some(subject => subject.object.entity_id === entity))!.id, victim);
  await choose(page, sacrifice);
  // Treasure is the only mana source left: its activation, {B} and the final
  // payment are forced, so native skipping commits them without a prompt, and
  // the passive opponent lets the spell resolve.
  await expect(page.getByTestId('recent')).toContainText('You cast Deadly Precision, targeting Callous Inspector');
  await expect(mine.locator('[data-card-name="Treasure"]')).toHaveCount(0);
  await expect(page.getByTestId('game-log')).toContainText('Cast Deadly Precision');
  await page.screenshot({ path: testInfo.outputPath('precision-payment.png') });

  const paidHash = await page.evaluate(() => window.__surface.frame!.frame_hash);
  const committed = await page.evaluate(() => window.__surface.commands.at(-1)!);
  await page.reload();
  await expect.poll(() => page.evaluate(() => window.__surface.frame?.frame_hash)).toBe(paidHash);
  // A receipt lost across the reconnect can be retried with its exact original Command ID.
  await page.evaluate(command => window.__surface.socket!.send(JSON.stringify({ type: 'command', command })), committed);
  await expect.poll(() => page.evaluate(() => window.__surface.outcomes.at(-1))).toBe('duplicate');
  expect(await page.evaluate(() => window.__surface.frame?.frame_hash)).toBe(paidHash);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
});

test('Precision exposes the shared four-mana alternative', async ({ page }) => {
  await start(page, 'precision-mana');
  await choose(page, await page.evaluate(() => window.__surface.frame!.offers.find(offer => offer.label === 'Cast Deadly Precision')!.id));
  await choose(page, await page.evaluate(() => window.__surface.frame!.offers.find(offer => offer.action_type === 'CHOOSE_TARGET')!.id));
  const alternative = await page.evaluate(() => window.__surface.frame!.offers.find(offer => offer.action_type === 'PAY_COST')!);
  expect(alternative.label).toContain('{4}');
  await choose(page, alternative.id);
  await expect(page.getByTestId('recent')).toContainText('You cast Deadly Precision');
  await expect(page.getByTestId('game-log')).toContainText('Cast Deadly Precision');
});
