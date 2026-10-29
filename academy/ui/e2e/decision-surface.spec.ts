import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
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

async function choose(page: Page, id: number): Promise<void> {
  const revision = await page.evaluate(() => window.__surface.frame!.revision);
  const button = page.locator(`[data-testid="action-option"][data-offer-id="${id}"]`);
  await button.focus();
  await page.keyboard.press('Enter');
  await expect.poll(() => page.evaluate(() => window.__surface.frame?.revision)).toBeGreaterThan(revision);
  const next = await page.evaluate(() => window.__surface.frame!.revision);
  await expect(page.getByTestId('decision-surface')).toHaveAttribute('data-revision', String(next));
}

for (const width of [390, 1500]) {
  test(`Precision inspection, explicit commitment and recovery at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 1100 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await track(page);
    await page.goto('/?demo=precision');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected', { timeout: 30_000 });
    const start = page.getByRole('button', { name: 'New Game', exact: true });
    await expect(start).toHaveCount(1);
    await start.click();
    await expect(page.getByRole('button', { name: 'Preparing game…', exact: true })).toBeDisabled();
    await expect(page.getByTestId('decision-surface')).toBeVisible({ timeout: 120_000 });
    const board = await page.getByTestId('game-board').boundingBox();
    const choices = await page.getByTestId('decision-surface').boundingBox();
    expect(board).not.toBeNull();
    expect(choices!.y).toBeGreaterThanOrEqual(board!.y + board!.height);
    const tableTools = page.getByTestId('shared-table-panel');
    await expect(tableTools).not.toBeVisible();
    expect((await page.getByTestId('game-header').boundingBox())!.y).toBeLessThan(board!.y);
    await expect(page.locator('details[open]')).toHaveCount(0);
    const battlefieldHeights = await page.getByTestId('battlefield').evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height));
    expect(battlefieldHeights).toHaveLength(2);
    const tokens = page.getByTestId('battlefield').last().locator('[data-battlefield-lane="tokens"]');
    while (await tokens.locator('[data-card-name="Treasure"]').count() === 0 && await tokens.getByRole('button', { name: 'Next artifact tokens' }).isEnabled()) {
      await tokens.getByRole('button', { name: 'Next artifact tokens' }).click();
    }
    await expect(tokens).toContainText('Treasure');
    await expect(page.getByTestId('battlefield').last().locator('[data-battlefield-lane="lands"]')).toContainText('Swamp');
    await expect(page.getByTestId('battlefield').last().locator('[data-battlefield-lane="creatures"]')).toContainText('Callous Inspector');
    const before = await page.evaluate(() => ({ hash: window.__surface.frame!.frame_hash, count: window.__surface.commands.length }));
    await page.getByTestId('game-board').getByRole('button', { name: /^Deadly Precision(?: \(|$)/ }).first().click();
    await expect(page.getByRole('region', { name: 'Inspected card' })).toContainText('Deadly Precision');
    expect(await page.evaluate(() => window.__surface.commands.length)).toBe(before.count);
    expect(await page.evaluate(() => window.__surface.frame!.frame_hash)).toBe(before.hash);
    await page.getByRole('button', { name: 'Show all', exact: true }).click();
    const cast = await page.evaluate(() => window.__surface.frame!.offers.find(offer => offer.label === 'Cast Deadly Precision')!.id);
    await choose(page, cast);
    const target = await page.evaluate(() => window.__surface.frame!.offers.find(offer => offer.subjects?.some(subject => subject.object.label === 'Callous Inspector'))!);
    const victim = target.subjects![0].object.entity_id;
    await choose(page, target.id);
    await expect(page.getByTestId('decision-context')).toContainText('Selected target');
    const sacrifice = await page.evaluate(entity => window.__surface.frame!.offers.find(offer => offer.action_type === 'SELECT_CARD' && offer.subjects?.some(subject => subject.object.entity_id === entity))!.id, victim);
    await choose(page, sacrifice);
    // Treasure is the only mana source left: its activation, {B} and the final
    // payment are forced, so native skipping commits them without a prompt.
    await expect(page.getByTestId('decision-surface')).toContainText('sacrificed Treasure');
    await expect(page.getByTestId('decision-surface')).toContainText('Deadly Precision was cast');
    expect(await page.getByTestId('battlefield').evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height))).toEqual(battlefieldHeights);
    await page.screenshot({ path: testInfo.outputPath('precision-payment.png'), fullPage: true });
    const paidHash = await page.evaluate(() => window.__surface.frame!.frame_hash);
    const committed = await page.evaluate(() => window.__surface.commands.at(-1)!);
    await page.reload();
    await expect.poll(() => page.evaluate(() => window.__surface.frame?.frame_hash)).toBe(paidHash);
    // A receipt lost across the reconnect can be retried with its exact original Command ID.
    await page.evaluate(command => window.__surface.socket!.send(JSON.stringify({ type: 'command', command })), committed);
    await expect.poll(() => page.evaluate(() => window.__surface.outcomes.at(-1))).toBe('duplicate');
    expect(await page.evaluate(() => window.__surface.frame?.frame_hash)).toBe(paidHash);
    const overflow = await page.locator('body *').evaluateAll(elements => elements.filter(element => element.getBoundingClientRect().right > window.innerWidth + 1).map(element => ({ tag: element.tagName, class: element.className, width: element.getBoundingClientRect().width })).slice(0, 12));
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), JSON.stringify(overflow)).toBe(true);
    const accessibility = await new AxeBuilder({ page }).analyze();
    expect(accessibility.violations).toEqual([]);
  });
}

test('Precision exposes the shared four-mana alternative', async ({ page }) => {
  await track(page);
  await page.goto('/?demo=precision-mana');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', { timeout: 30_000 });
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  await expect(page.getByTestId('decision-surface')).toBeVisible({ timeout: 120_000 });
  await choose(page, await page.evaluate(() => window.__surface.frame!.offers.find(offer => offer.label === 'Cast Deadly Precision')!.id));
  await choose(page, await page.evaluate(() => window.__surface.frame!.offers.find(offer => offer.action_type === 'CHOOSE_TARGET')!.id));
  const alternative = await page.evaluate(() => window.__surface.frame!.offers.find(offer => offer.action_type === 'PAY_COST')!);
  expect(alternative.label).toContain('{4}');
  await choose(page, alternative.id);
  await expect(page.getByTestId('decision-surface')).toContainText('Deadly Precision was cast');
});
