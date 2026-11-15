import { expect, test, type Page } from '@playwright/test';
import type { Command, ExperienceFrame } from '../src/lib/types';
import type { PreviewResponse } from '../src/lib/payment-preview';
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
async function choose(page: Page, id: number): Promise<void> {
  const before = await page.evaluate(() => window.__mana.frame!.revision);
  await page.locator(`[data-testid="action-option"][data-offer-id="${id}"]`).focus();
  await page.keyboard.press('Enter');
  await expect.poll(() => page.evaluate(() => window.__mana.frame?.revision), { timeout: 20_000 }).toBeGreaterThan(before);
  await finishBoardNarration(page);
}
async function offer(page: Page, label: string): Promise<number> {
  const current = await page.evaluate(() => window.__mana.frame!);
  const found = current.offers.find(candidate => candidate.label.includes(label));
  expect(found, label).toBeDefined();
  return found!.id;
}

for (const [width, height] of [[1280, 800], [1440, 900]]) {
  test(`native payment preview, deliberate sources and pending output recovery at ${width}`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await track(page);
    await page.goto('/');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    await page.getByRole('button', { name: 'New Game', exact: true }).click();
    await expect(page.getByTestId('decision-surface')).toBeVisible();
    await finishBoardNarration(page);
    // Blur the automatically focused first choice so keyboard focus can ask
    // again; the query remains cached only within this current frame.
    const close = page.getByRole('button', { name: 'Close preview', exact: true });
    if (await close.isVisible()) await close.click();
    const cast = await offer(page, 'Cast Boomerang Basics');
    // An initial autofocus may already have populated the frame-local cache.
    let preview: PreviewResponse | null = null;
    const credentials = await page.evaluate(() => JSON.parse(sessionStorage.getItem('etude.gui.resume')!));
    const current = await page.evaluate(() => window.__mana.frame!);
    const response = await page.request.post(`/api/sessions/${credentials.session_id}/payment-preview`, {
      headers: { 'x-etude-participant-token': credentials.resume_token }, data: {
        match_id: current.match_id, revision: current.revision, prompt_id: current.prompt!.id,
        frame_hash: current.frame_hash, offer_id: cast, request_id: 'browser-parity',
      },
    });
    expect(response.ok()).toBe(true);
    preview = await response.json();
    await page.locator(`[data-testid="action-option"][data-offer-id="${cast}"]`).focus();
    await expect(page.getByTestId('payment-preview')).toContainText('Exact automatic source plan');
    expect(preview!.preview.sources.length).toBeGreaterThan(0);
    const proposed = preview!.preview.sources.map(source => source.source.kind === 'object' ? source.source.id.entity : -1);
    await expect(page.locator('[data-payment-source]')).toHaveCount(proposed.length);
    const sourceBounds = await page.locator('[data-payment-source]').first().boundingBox();
    const sourceRegion = await page.getByRole('region', { name: 'You battlefield and zones', exact: true }).boundingBox();
    expect(sourceBounds!.y + sourceBounds!.height).toBeLessThanOrEqual(sourceRegion!.y + sourceRegion!.height + 1);
    for (const theme of ['light', 'dark'] as const) {
      await page.emulateMedia({ colorScheme: theme });
      await page.screenshot({ path: testInfo.outputPath(`payment-${width}-${theme}.png`), fullPage: true });
    }
    const count = await page.evaluate(() => window.__mana.commands.length);
    await page.getByRole('button', { name: 'Close preview', exact: true }).click();
    await expect(page.locator('[data-payment-source]')).toHaveCount(0);
    expect(await page.evaluate(() => window.__mana.commands.length)).toBe(count);
    const tap = current.offers.find(candidate => candidate.label.startsWith('Tap Island for')
      && candidate.source?.kind === 'object' && !proposed.includes(candidate.source.id.entity));
    expect(tap).toBeDefined();
    await choose(page, tap!.id);
    const floated = await page.evaluate(() => window.__mana.frame!);
    expect(floated.projection.agent.mana![1]).toBe(1);
    const nextCast = await offer(page, 'Cast Boomerang Basics');
    await page.locator(`[data-testid="action-option"][data-offer-id="${nextCast}"]`).hover();
    await expect(page.getByTestId('payment-preview')).toContainText('No automatic source taps needed');
    // Modal production is a committed activation, followed by a real native choice.
    await choose(page, await offer(page, 'Activate ability: Tarnished Citadel'));
    const pending = await page.evaluate(() => window.__mana.frame!);
    expect(pending.action_space).toBe('MODAL');
    await page.reload();
    await expect.poll(() => page.evaluate(() => window.__mana.frame?.frame_hash)).toBe(pending.frame_hash);
    await finishBoardNarration(page);
    await choose(page, await offer(page, 'Add {U}'));
    expect((await page.evaluate(() => window.__mana.frame!)).projection.agent.mana![1]).toBe(2);
    const sourceFrame = await page.evaluate(() => window.__mana.frame!);
    const restrictedSource = sourceFrame.offers.find(candidate => candidate.label.includes('White Lotus Hideout')
      && candidate.label.includes('Lesson') && candidate.label.includes('pay {0}'));
    expect(restrictedSource).toBeDefined();
    await choose(page, restrictedSource!.id);
    await choose(page, await offer(page, 'Add {U}'));
    await page.getByTestId('mana-pool').last().locator('summary').click();
    await expect(page.getByTestId('mana-pool').last()).toContainText('Lesson');
    await expect(page.getByTestId('mana-pool').last()).toContainText('included in the total');
    await page.getByTestId('mana-pool').last().locator('summary').click();
    // Target-independent costs remain exact before a target is chosen.
    await choose(page, await offer(page, 'Cast Boomerang Basics'));
    const targeting = await page.evaluate(() => window.__mana.frame!);
    expect(targeting.action_space).toBe('CHOOSE_TARGET');
    await choose(page, targeting.offers[0].id);
    await choose(page, await offer(page, 'Pay the cost'));
    const spending = await page.evaluate(() => window.__mana.frame!);
    const restrictedSpend = spending.offers.find(candidate => candidate.label.includes('Spend') && candidate.label.includes('Lesson'));
    expect(restrictedSpend).toBeDefined();
    await choose(page, restrictedSpend!.id);
    expect((await page.evaluate(() => window.__mana.frame!)).projection.agent.mana![1]).toBe(2);
    const board = await page.getByTestId('game-board').boundingBox();
    expect(board!.y + board!.height).toBeLessThanOrEqual(height);
    await page.getByRole('link', { name: 'Open replay', exact: true }).click();
    await expect(page.getByText('Automatic payment preview was not recorded.', { exact: false })).toBeVisible();
    await expect(page.locator('[data-payment-source]')).toHaveCount(0);
    await expect(page.getByTestId('action-option').first()).toBeDisabled();
  });
}

test('a delayed preview response cannot restore source marks after a Command', async ({ page }) => {
  await track(page);
  let release!: () => void;
  let captured = false;
  const delayed = new Promise<void>(done => release = done);
  await page.route('**/payment-preview', async route => {
    if (captured) { await route.continue(); return; }
    captured = true;
    const response = await route.fetch();
    await delayed;
    await route.fulfill({ response }).catch(() => {}); // The browser may have aborted the old request.
  });
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'New Game', exact: true }).click();
  await expect(page.getByTestId('decision-surface')).toBeVisible();
  await finishBoardNarration(page);
  await page.locator(`[data-testid="action-option"][data-offer-id="${await offer(page, 'Cast Boomerang Basics')}"]`).hover();
  await expect.poll(() => captured).toBe(true);
  await choose(page, await offer(page, 'Tap Island for'));
  release();
  await expect(page.locator('[data-payment-source]')).toHaveCount(0);
  expect((await page.evaluate(() => window.__mana.frame!)).projection.agent.mana![1]).toBe(1);
});
