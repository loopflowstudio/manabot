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
    // Native `mana` excludes restricted mana; the summary counts both.
    await expect(page.getByTestId('mana-pool').last().locator('summary')).toHaveText('You floating mana: 3 {U}');
    await expect(page.getByTestId('mana-pool').last()).toContainText('part of that total');
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


test('activation fee preview stops before output and resumes the outer spell payment', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await track(page);
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'New Game', exact: true }).click();
  await expect(page.getByTestId('decision-surface')).toBeVisible();
  await finishBoardNarration(page);
  const preview = page.getByTestId('payment-preview');
  const prism = await offer(page, 'Activate ability: Prophetic Prism');
  await page.locator(`[data-testid="action-option"][data-offer-id="${prism}"]`).focus();
  await expect(preview).toContainText('Activation fee: {1}');
  await expect(preview).toContainText('Pool after fee, before ability output');
  await expect(preview).toContainText("Choose the ability's output after activation");
  await expect(preview).not.toContainText('The outer payment still follows');
  await choose(page, await offer(page, 'Cast Boomerang Basics'));
  const targeting = await page.evaluate(() => window.__mana.frame!);
  await choose(page, targeting.offers[0].id);
  const outer = await page.evaluate(() => window.__mana.frame!);
  expect(outer.offers.some(candidate => candidate.label === 'Pay the cost')).toBe(true);
  const nestedPrism = await offer(page, 'Activate ability: Prophetic Prism');
  await page.locator(`[data-testid="action-option"][data-offer-id="${nestedPrism}"]`).focus();
  await expect(preview).toContainText('Activation fee: {1}');
  await expect(preview).toContainText('The outer payment still follows');
  for (const theme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme: theme });
    await page.screenshot({ path: testInfo.outputPath(`activation-fee-${theme}.png`), fullPage: true });
  }
  await choose(page, nestedPrism);
  const pending = await page.evaluate(() => window.__mana.frame!);
  expect(pending.action_space).toBe('MODAL');
  expect(pending.offers.some(candidate => candidate.label.includes('Add {U}'))).toBe(true);
  expect(pending.projection.agent.mana!.reduce((a, b) => a + b, 0)).toBe(0);
  await page.reload();
  await expect.poll(() => page.evaluate(() => window.__mana.frame?.frame_hash)).toBe(pending.frame_hash);
  await finishBoardNarration(page);
  expect((await page.evaluate(() => window.__mana.frame!)).projection.agent.mana).toEqual(pending.projection.agent.mana);
  await choose(page, await offer(page, 'Add {U}'));
  expect((await page.evaluate(() => window.__mana.frame!)).projection.agent.mana![1]).toBe(1);
  const pay = await offer(page, 'Pay the cost');
  await page.locator(`[data-testid="action-option"][data-offer-id="${pay}"]`).focus();
  await expect(preview).toContainText('Spell cost proposal');
  await expect(preview).toContainText('Pool after cost, before spell effects');
  await expect(preview).not.toContainText('The outer payment still follows');
  await choose(page, pay);
  expect((await page.evaluate(() => window.__mana.frame!)).projection.agent.mana![1]).toBe(0);
});

for (const viewport of [{ width: 1280, height: 800 }, { width: 1440, height: 900 }]) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`moving the pointer between offers never resizes the offer list ${viewport.width} ${colorScheme}`, async ({ page }, testInfo) => {
      await page.setViewportSize(viewport);
      await page.emulateMedia({ colorScheme });
      await track(page);
      await page.goto('/');
      await expect(page.getByTestId('connection-badge')).toHaveText('connected');
      await page.getByRole('button', { name: 'New Game', exact: true }).click();
      await expect(page.getByTestId('decision-surface')).toBeVisible();
      await finishBoardNarration(page);
      const cast = page.locator(`[data-testid="action-option"][data-offer-id="${await offer(page, 'Cast Boomerang Basics')}"]`);
      const pass = page.locator(`[data-testid="action-option"][data-offer-id="${await offer(page, 'Pass priority')}"]`);
      // Layout, not viewport position: Playwright scrolls an offer into view to hover it.
      const layout = () => page.evaluate(() => {
        const offer = document.querySelector<HTMLElement>('[data-testid="action-option"]')!;
        let list: HTMLElement | null = offer.parentElement;
        while (list && list.scrollHeight <= list.clientHeight) list = list.parentElement;
        return { listHeight: list?.clientHeight, top: offer.getBoundingClientRect().top - (list?.getBoundingClientRect().top ?? 0) + (list?.scrollTop ?? 0) };
      });
      await cast.hover();
      await expect(page.getByTestId('payment-preview')).toContainText('Cast Boomerang Basics');
      const anchored = await layout();
      // With the preview open, keep multiple complete ordinary offers in view.
      // No source is collapsed or substituted to make this fit.
      const firstRows = await page.getByTestId('action-option').evaluateAll(offers =>
        offers.slice(0, 2).reduce((height, offer) => height + offer.getBoundingClientRect().height, 8));
      expect(anchored.listHeight).toBeGreaterThanOrEqual(firstRows);
      await page.screenshot({ path: testInfo.outputPath('payment-choices-density.png') });
      // An action without a payment keeps the slot open rather than collapsing it,
      // so the offer under the pointer cannot change by itself.
      await pass.hover();
      await expect(page.getByTestId('payment-preview')).toBeVisible();
      expect(await layout()).toEqual(anchored);
      await cast.hover();
      expect(await layout()).toEqual(anchored);
    });

  }
}

for (const viewport of [{ width: 1280, height: 800 }, { width: 1440, height: 900 }]) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`payment is immediately reachable and previews on arrival ${viewport.width} ${colorScheme}`, async ({ page }, testInfo) => {
      await page.setViewportSize(viewport);
      await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
      await track(page);
      await page.goto('/');
      await expect(page.getByTestId('connection-badge')).toHaveText('connected');
      await page.getByRole('button', { name: 'New Game', exact: true }).click();
      await expect(page.getByTestId('decision-surface')).toBeVisible();
      await finishBoardNarration(page);
      await choose(page, await offer(page, 'Cast Boomerang Basics'));
      await choose(page, (await page.evaluate(() => window.__mana.frame!)).offers[0].id);
      const frame = await page.evaluate(() => window.__mana.frame!);
      const payId = await offer(page, 'Pay the cost');
      const pay = page.locator(`[data-testid="action-option"][data-offer-id="${payId}"]`);
      const preview = page.getByTestId('payment-preview');
      // No locator.focus/hover/scroll here: arrival itself must reveal payment.
      await expect(pay).toBeFocused();
      await expect(page.getByTestId('action-option').first()).toHaveAttribute('data-offer-id', String(payId));
      await expect(preview).toContainText('Pay the cost');
      await expect(preview).toContainText('Exact automatic source plan');
      const commands = await page.evaluate(() => window.__mana.commands.length);
      const bounds = await pay.boundingBox();
      const list = await page.getByRole('group', { name: 'Legal choices', exact: true }).boundingBox();
      expect(bounds!.y).toBeGreaterThanOrEqual(list!.y);
      expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(list!.y + list!.height);
      expect(bounds!.y + bounds!.height).toBeLessThan(viewport.height);
      // Navigation retains every native source with its original identity.
      expect((await page.getByTestId('action-option').evaluateAll(rows => rows.map(row => Number(row.getAttribute('data-offer-id'))))).sort((a, b) => a - b))
        .toEqual(frame.offers.map(candidate => candidate.id).sort((a, b) => a - b));
      const firstSource = frame.offers.find(candidate => candidate.action_type === 'PRIORITY_ACTIVATE_ABILITY')!;
      expect(firstSource).toBeDefined();
      await page.keyboard.press('Tab');
      await expect(page.locator(`[data-testid="action-option"][data-offer-id="${firstSource.id}"]`)).toBeFocused();
      await page.keyboard.press('Shift+Tab');
      await expect(pay).toBeFocused();
      await expect(preview).toContainText('Pay the cost');
      expect(await page.evaluate(() => window.__mana.commands.length)).toBe(commands);
      await page.screenshot({ path: testInfo.outputPath('payment-on-arrival.png') });
      if (colorScheme === 'light') {
        // A real pointer reaches the visible payment without locator auto-scroll.
        await page.mouse.move(bounds!.x + bounds!.width / 2, bounds!.y + bounds!.height / 2);
        await expect(preview).toContainText('Pay the cost');
        await page.mouse.click(bounds!.x + bounds!.width / 2, bounds!.y + bounds!.height / 2);
      } else {
        await page.keyboard.press('Enter');
      }
      await expect.poll(() => page.evaluate(() => window.__mana.commands.length)).toBe(commands + 1);
      expect((await page.evaluate(() => window.__mana.commands.at(-1)!)).offer_id).toBe(payId);
      await expect.poll(() => page.evaluate(() => window.__mana.frame!.revision)).toBeGreaterThan(frame.revision);
    });
  }
}
